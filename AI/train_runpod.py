#!/usr/bin/env python3
"""
GPU-optimized AlphaZero training for 9x9 Torus Go.
Designed for RunPod RTX 4090 or similar CUDA GPUs.

Self-play runs directly on GPU for fast inference (~0.2ms/position vs ~5ms on CPU).
This is faster than CPU-parallel multiprocessing for CUDA GPUs.

Usage:
    # Full training run (recommended for RTX 4090)
    python train_runpod.py --cycles 100 --games-per-cycle 200 --mcts-sims 200

    # Faster initial training (first 24h)
    python train_runpod.py --cycles 100 --games-per-cycle 200 --mcts-sims 100

    # Quick dry-run test
    python train_runpod.py --dry-run --cycles 1 --games-per-cycle 2 --mcts-sims 10

    # Resume from checkpoint
    python train_runpod.py --resume --cycles 100 --games-per-cycle 200 --mcts-sims 200

    # Force CPU-parallel mode (e.g. for Mac MPS where GPU self-play has issues)
    python train_runpod.py --cpu-selfplay --workers 8 --cycles 100 --games-per-cycle 200 --mcts-sims 100
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time
import os
import json
import argparse
import multiprocessing as mp
from network import TorusGoNet
from selfplay import play_game, play_game_for_worker, augment_torus_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOARD_SIZE = 9
CHANNELS = 128
RES_BLOCKS = 5
IN_CHANNELS = 4          # 4-channel input: own stones, opp stones, color, move_num
MAX_BUFFER = 150_000      # Replay buffer max positions
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training_log.json"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GoDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = torch.tensor(np.array(states), dtype=torch.float32)
        self.policies = torch.tensor(np.array(policies), dtype=torch.float32)
        self.values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

# ---------------------------------------------------------------------------
# GPU Self-Play: runs games sequentially with GPU inference (FAST)
# ---------------------------------------------------------------------------
def gpu_selfplay(model, device, num_games, mcts_sims, in_channels):
    """Run self-play games sequentially on GPU. Each MCTS inference is a single
    GPU forward pass (~0.2ms on RTX 4090), making this faster than CPU-parallel."""
    all_states, all_policies, all_values = [], [], []
    game_results = []

    model.eval()
    for g in range(num_games):
        states, policies, values = play_game(
            model, mcts_simulations=mcts_sims, temperature=1.0,
            device=device, size=BOARD_SIZE, add_noise=True, in_channels=in_channels
        )
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)

        # Track game result (values[0] = result from Black's perspective)
        game_result = values[0] if len(values) > 0 else 0.0
        game_results.append(game_result)

        if (g + 1) % max(1, num_games // 10) == 0:
            print(f"  ... {g+1}/{num_games} games done", flush=True)

    return all_states, all_policies, all_values, game_results

# ---------------------------------------------------------------------------
# CPU Self-Play: runs games in parallel with multiprocessing (for non-CUDA)
# ---------------------------------------------------------------------------
def cpu_selfplay(model, num_games, mcts_sims, in_channels, workers):
    """Run self-play games in parallel on CPU workers."""
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    worker_args = [
        (model_state, mcts_sims, BOARD_SIZE, in_channels)
        for _ in range(num_games)
    ]

    all_states, all_policies, all_values = [], [], []
    game_results = []
    completed = 0

    with mp.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(play_game_for_worker, worker_args):
            states, policies, values, game_result = result
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
            game_results.append(game_result)
            completed += 1
            if completed % max(1, num_games // 10) == 0:
                print(f"  ... {completed}/{num_games} games done", flush=True)

    return all_states, all_policies, all_values, game_results

# ---------------------------------------------------------------------------
# Evaluation: pit current model vs. previous best
# ---------------------------------------------------------------------------
def evaluate_models(current_model, best_model, device, num_games=40, mcts_sims=100):
    """Play num_games between current and best model. Returns win rate of current."""
    from torusgo import TorusGo
    from mcts import MCTS

    wins = 0
    for g in range(num_games):
        game = TorusGo(size=BOARD_SIZE)
        current_is_black = (g % 2 == 0)

        mcts_current = MCTS(current_model, num_simulations=mcts_sims, device=device)
        mcts_best = MCTS(best_model, num_simulations=mcts_sims, device=device)
        move_number = 0

        while not game.game_over:
            is_current_turn = (game.current_player == 1) == current_is_black
            mcts_obj = mcts_current if is_current_turn else mcts_best
            action_probs = mcts_obj.get_action_prob(game, temperature=0.0, move_number=move_number)
            action = int(np.argmax(action_probs))
            game.step(action)
            move_number += 1
            if move_number > 200:
                break

        reward = game.get_reward()
        if current_is_black and reward > 0:
            wins += 1
        elif not current_is_black and reward < 0:
            wins += 1

    return wins / num_games

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def train(args):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Decide self-play mode
    use_gpu_selfplay = (device.type == 'cuda') and not args.cpu_selfplay
    if use_gpu_selfplay:
        print(f"[train] Using device: {device} (GPU self-play — fast inference)")
    else:
        print(f"[train] Using device: {device} (CPU-parallel self-play, {args.workers} workers)")

    # Model
    model = TorusGoNet(
        size=BOARD_SIZE, channels=CHANNELS,
        num_res_blocks=RES_BLOCKS, in_channels=IN_CHANNELS
    ).to(device)

    # Best model (for evaluation gating)
    best_model = TorusGoNet(
        size=BOARD_SIZE, channels=CHANNELS,
        num_res_blocks=RES_BLOCKS, in_channels=IN_CHANNELS
    ).to(device)
    best_model.load_state_dict(model.state_dict())

    # Optimizer + Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.cycles, eta_min=args.lr / 20)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # Replay buffer
    all_states, all_policies, all_values = [], [], []

    # Logging
    log_entries = []
    start_cycle = 0

    # Resume from checkpoint
    if args.resume:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(ckpt_path):
            print(f"[train] Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            best_model.load_state_dict(ckpt['best_model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_cycle = ckpt['cycle'] + 1
            all_states = ckpt.get('buffer_states', [])
            all_policies = ckpt.get('buffer_policies', [])
            all_values = ckpt.get('buffer_values', [])
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE) as f:
                    log_entries = json.load(f)
            print(f"[train] Resumed at cycle {start_cycle}, buffer has {len(all_states)} positions")
        else:
            print(f"[train] No checkpoint found at {ckpt_path}, starting fresh")

    total_start = time.time()

    for cycle in range(start_cycle, args.cycles):
        cycle_start = time.time()
        lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle+1}/{args.cycles}  |  LR: {lr:.6f}  |  Buffer: {len(all_states)} positions")
        print(f"{'='*60}", flush=True)

        # -------------------------------------------------------------------
        # 1. Self-play
        # -------------------------------------------------------------------
        model.eval()
        print(f"[selfplay] Generating {args.games_per_cycle} games, {args.mcts_sims} MCTS sims "
              f"({'GPU' if use_gpu_selfplay else f'CPU x{args.workers}'})...", flush=True)
        sp_start = time.time()

        if use_gpu_selfplay:
            cycle_states, cycle_policies, cycle_values, game_results = gpu_selfplay(
                model, device, args.games_per_cycle, args.mcts_sims, IN_CHANNELS
            )
        else:
            cycle_states, cycle_policies, cycle_values, game_results = cpu_selfplay(
                model, args.games_per_cycle, args.mcts_sims, IN_CHANNELS, args.workers
            )

        sp_time = time.time() - sp_start
        avg_game_len = len(cycle_states) / max(1, args.games_per_cycle)

        # Win rate stats
        black_wins = sum(1 for r in game_results if r > 0)
        white_wins = sum(1 for r in game_results if r < 0)
        draws = sum(1 for r in game_results if r == 0)
        total_games = len(game_results)
        black_pct = black_wins / total_games * 100 if total_games > 0 else 0
        white_pct = white_wins / total_games * 100 if total_games > 0 else 0
        draw_pct = draws / total_games * 100 if total_games > 0 else 0

        print(f"[selfplay] {args.games_per_cycle} games in {sp_time:.1f}s ({sp_time/args.games_per_cycle:.2f}s/game, avg {avg_game_len:.0f} moves)")
        print(f"[selfplay] Results: Black {black_wins}/{total_games} ({black_pct:.1f}%) | White {white_wins}/{total_games} ({white_pct:.1f}%) | Draw {draws}/{total_games} ({draw_pct:.1f}%)", flush=True)

        # -------------------------------------------------------------------
        # 2. Torus data augmentation
        # -------------------------------------------------------------------
        if args.augment > 0:
            before = len(cycle_states)
            cycle_states, cycle_policies, cycle_values = augment_torus_data(
                cycle_states, cycle_policies, cycle_values,
                size=BOARD_SIZE, num_augments=args.augment
            )
            print(f"[augment] {before} → {len(cycle_states)} positions ({args.augment} torus shifts per sample)")

        # -------------------------------------------------------------------
        # 3. Update replay buffer
        # -------------------------------------------------------------------
        all_states = (all_states + cycle_states)[-MAX_BUFFER:]
        all_policies = (all_policies + cycle_policies)[-MAX_BUFFER:]
        all_values = (all_values + cycle_values)[-MAX_BUFFER:]

        # -------------------------------------------------------------------
        # 4. Train network
        # -------------------------------------------------------------------
        dataset = GoDataset(all_states, all_policies, all_values)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=(device.type == 'cuda')
        )

        model.train()
        print(f"[train] Training on {len(dataset)} positions for {args.train_epochs} epochs...")

        for epoch in range(args.train_epochs):
            total_loss, total_p, total_v, steps = 0, 0, 0, 0
            for states_b, policies_b, values_b in dataloader:
                states_b = states_b.to(device)
                policies_b = policies_b.to(device)
                values_b = values_b.to(device)

                optimizer.zero_grad()
                pred_pi, pred_v = model(states_b)

                p_loss = policy_loss_fn(pred_pi, policies_b)
                v_loss = value_loss_fn(pred_v, values_b)
                loss = p_loss + v_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_p += p_loss.item()
                total_v += v_loss.item()
                steps += 1

            print(f"  Epoch {epoch+1}/{args.train_epochs} | Loss: {total_loss/steps:.4f} (Pi: {total_p/steps:.4f}, V: {total_v/steps:.4f})")

        scheduler.step()

        # -------------------------------------------------------------------
        # 5. Evaluation & model gating (every eval_interval cycles)
        # -------------------------------------------------------------------
        if (cycle + 1) % args.eval_interval == 0 and not args.dry_run:
            print(f"[eval] Evaluating current vs best ({args.eval_games} games)...")
            model.eval()
            win_rate = evaluate_models(
                model, best_model, device,
                num_games=args.eval_games, mcts_sims=args.eval_mcts_sims
            )
            print(f"[eval] Current model win rate: {win_rate:.1%}")
            if win_rate >= 0.55:
                print(f"[eval] ✅ New best model! (win rate {win_rate:.1%} >= 55%)")
                best_model.load_state_dict(model.state_dict())
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_9x9_best.pt"))
            else:
                print(f"[eval] ❌ Keeping previous best (win rate {win_rate:.1%} < 55%)")
        elif (cycle + 1) % args.eval_interval == 0 and args.dry_run:
            best_model.load_state_dict(model.state_dict())

        # -------------------------------------------------------------------
        # 6. Checkpoint
        # -------------------------------------------------------------------
        cycle_time = time.time() - cycle_start
        entry = {
            "cycle": cycle + 1,
            "loss": total_loss / steps if steps > 0 else 0,
            "pi_loss": total_p / steps if steps > 0 else 0,
            "v_loss": total_v / steps if steps > 0 else 0,
            "buffer_size": len(all_states),
            "games": args.games_per_cycle,
            "avg_game_len": avg_game_len,
            "cycle_time_s": cycle_time,
            "lr": lr,
            "black_wins": black_wins,
            "white_wins": white_wins,
            "draws": draws,
            "black_win_pct": round(black_pct, 1),
            "white_win_pct": round(white_pct, 1),
        }
        log_entries.append(entry)

        if (cycle + 1) % args.save_interval == 0 or (cycle + 1) == args.cycles:
            ckpt = {
                'cycle': cycle,
                'model': model.state_dict(),
                'best_model': best_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'buffer_states': all_states[-50000:],
                'buffer_policies': all_policies[-50000:],
                'buffer_values': all_values[-50000:],
            }
            ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
            torch.save(ckpt, ckpt_path)
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_9x9_c{cycle+1}.pt"))
            print(f"[ckpt] Saved checkpoint at cycle {cycle+1}")

        with open(LOG_FILE, 'w') as f:
            json.dump(log_entries, f, indent=2)

        total_elapsed = (time.time() - total_start) / 3600
        est_remaining = (total_elapsed / (cycle - start_cycle + 1)) * (args.cycles - cycle - 1)
        print(f"[time] Cycle: {cycle_time:.0f}s | Total: {total_elapsed:.1f}h | Est. remaining: {est_remaining:.1f}h", flush=True)

    # Save final models
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_9x9_final.pt"))
    torch.save(best_model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_9x9_best.pt"))
    total_time = (time.time() - total_start) / 3600
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — {total_time:.1f} hours")
    print(f"  Best model: {CHECKPOINT_DIR}/model_9x9_best.pt")
    print(f"{'='*60}")


def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Train 9x9 Torus Go on GPU")
    parser.add_argument('--cycles', type=int, default=100, help='Number of training cycles')
    parser.add_argument('--games-per-cycle', type=int, default=200, help='Self-play games per cycle')
    parser.add_argument('--mcts-sims', type=int, default=200, help='MCTS simulations per move')
    parser.add_argument('--workers', type=int, default=8, help='CPU workers (only used with --cpu-selfplay)')
    parser.add_argument('--cpu-selfplay', action='store_true', help='Force CPU-parallel self-play instead of GPU')
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--train-epochs', type=int, default=4, help='Training epochs per cycle')
    parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate')
    parser.add_argument('--augment', type=int, default=4, help='Torus shift augmentations per sample (0 to disable)')
    parser.add_argument('--save-interval', type=int, default=5, help='Save checkpoint every N cycles')
    parser.add_argument('--eval-interval', type=int, default=10, help='Evaluate model every N cycles')
    parser.add_argument('--eval-games', type=int, default=40, help='Games for model evaluation')
    parser.add_argument('--eval-mcts-sims', type=int, default=100, help='MCTS sims for evaluation games')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--dry-run', action='store_true', help='Quick test with minimal settings')
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] Using minimal settings for testing")
        args.train_epochs = 1
        args.eval_interval = 1
        args.save_interval = 1

    print(f"[config] cycles={args.cycles}, games/cycle={args.games_per_cycle}, "
          f"mcts_sims={args.mcts_sims}, "
          f"selfplay={'CPU x' + str(args.workers) if args.cpu_selfplay else 'GPU'}, "
          f"batch={args.batch_size}, lr={args.lr}, augment={args.augment}")

    train(args)


if __name__ == '__main__':
    main()

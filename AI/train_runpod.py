#!/usr/bin/env python3
"""
GPU-optimized AlphaZero training for 9x9 Torus Go.
Designed for RunPod RTX 4090 or similar CUDA GPUs.

Optimizations:
1. Threaded Self-Play: Multiple games run in parallel on CPU threads while sharing the same GPU model.
2. Fast Go Logic: Optimized TorusGo implementation.
3. Fast Tensor Conversion: Uses torch.from_numpy and minimal copies.

Usage:
    # Full training run (standard for RTX 4090)
    python train_runpod.py --cycles 100 --games-per-cycle 200 --mcts-sims 100 --threads 8

    # Resume from checkpoint
    python train_runpod.py --resume --cycles 100 --games-per-cycle 200 --mcts-sims 100
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
from concurrent.futures import ThreadPoolExecutor
from network import TorusGoNet
from selfplay import play_game, play_game_for_worker, augment_torus_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOARD_SIZE = 9
CHANNELS = 128
RES_BLOCKS = 5
IN_CHANNELS = 4          # 4-channel input
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
# Threaded GPU Self-Play
# ---------------------------------------------------------------------------
def gpu_selfplay(model, device, num_games, mcts_sims, in_channels, num_threads):
    all_states, all_policies, all_values = [], [], []
    game_results = []
    completed = 0

    def play_one_game(_):
        nonlocal completed
        s, p, v = play_game(
            model, mcts_simulations=mcts_sims, temperature=1.0,
            device=device, size=BOARD_SIZE, add_noise=True, in_channels=in_channels
        )
        res = v[0] if len(v) > 0 else 0.0
        completed += 1
        if completed % max(1, num_games // 20) == 0:
            print(f"  ... {completed}/{num_games} games done", flush=True)
        return s, p, v, res

    model.eval()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(play_one_game, range(num_games)))

    for s, p, v, res in results:
        all_states.extend(s)
        all_policies.extend(p)
        all_values.extend(v)
        game_results.append(res)

    return all_states, all_policies, all_values, game_results

# ---------------------------------------------------------------------------
# CPU Self-Play (Fallback)
# ---------------------------------------------------------------------------
def cpu_selfplay(model, num_games, mcts_sims, in_channels, workers):
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    worker_args = [(model_state, mcts_sims, BOARD_SIZE, in_channels) for _ in range(num_games)]
    all_states, all_policies, all_values, game_results = [], [], [], []
    completed = 0
    with mp.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(play_game_for_worker, worker_args):
            s, p, v, res = result
            all_states.extend(s); all_policies.extend(p); all_values.extend(v)
            game_results.append(res)
            completed += 1
            if completed % max(1, num_games // 10) == 0:
                print(f"  ... {completed}/{num_games} games done", flush=True)
    return all_states, all_policies, all_values, game_results

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_models(current_model, best_model, device, num_games=40, mcts_sims=100):
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
            if move_number > 200: break
        reward = game.get_reward()
        if (current_is_black and reward > 0) or (not current_is_black and reward < 0):
            wins += 1
    return wins / num_games

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def train(args):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    use_gpu_selfplay = (device.type == 'cuda') and not args.cpu_selfplay
    print(f"[train] Device: {device} | Selfplay: {'GPU (Threaded x' + str(args.threads) + ')' if use_gpu_selfplay else 'CPU (Parallel x' + str(args.workers) + ')'}")

    model = TorusGoNet(size=BOARD_SIZE, channels=CHANNELS, num_res_blocks=RES_BLOCKS, in_channels=IN_CHANNELS).to(device)
    best_model = TorusGoNet(size=BOARD_SIZE, channels=CHANNELS, num_res_blocks=RES_BLOCKS, in_channels=IN_CHANNELS).to(device)
    best_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.cycles, eta_min=args.lr / 20)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    all_states, all_policies, all_values, log_entries = [], [], [], []
    start_cycle = 0

    if args.resume:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model']); best_model.load_state_dict(ckpt['best_model'])
            optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['scheduler'])
            start_cycle = ckpt['cycle'] + 1
            all_states, all_policies, all_values = ckpt.get('buffer_states', []), ckpt.get('buffer_policies', []), ckpt.get('buffer_values', [])
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE) as f: log_entries = json.load(f)
            print(f"[train] Resumed at cycle {start_cycle}")

    total_start = time.time()
    for cycle in range(start_cycle, args.cycles):
        cycle_start = time.time()
        lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*60}\n  CYCLE {cycle+1}/{args.cycles} | LR: {lr:.6f} | Buffer: {len(all_states)}\n{'='*60}", flush=True)

        model.eval()
        sp_start = time.time()
        if use_gpu_selfplay:
            s, p, v, res = gpu_selfplay(model, device, args.games_per_cycle, args.mcts_sims, IN_CHANNELS, args.threads)
        else:
            s, p, v, res = cpu_selfplay(model, args.games_per_cycle, args.mcts_sims, IN_CHANNELS, args.workers)
        
        sp_time = time.time() - sp_start
        all_states = (all_states + s)[-MAX_BUFFER:]
        all_policies = (all_policies + p)[-MAX_BUFFER:]
        all_values = (all_values + v)[-MAX_BUFFER:]

        b_wins, w_wins = sum(1 for r in res if r > 0), sum(1 for r in res if r < 0)
        total_g = len(res)
        print(f"[selfplay] {total_g} games in {sp_time:.1f}s | B: {b_wins/total_g:.1%} W: {w_wins/total_g:.1%}", flush=True)

        if args.augment > 0:
            s, p, v = augment_torus_data(s, p, v, size=BOARD_SIZE, num_augments=args.augment)

        dataset = GoDataset(all_states, all_policies, all_values)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == 'cuda'))
        
        model.train()
        print(f"[train] Training on {len(dataset)} positions...")
        for epoch in range(args.train_epochs):
            t_loss, t_p, t_v, steps = 0, 0, 0, 0
            for sb, pb, vb in dataloader:
                sb, pb, vb = sb.to(device), pb.to(device), vb.to(device)
                optimizer.zero_grad(); pr_pi, pr_v = model(sb)
                loss = policy_loss_fn(pr_pi, pb) + value_loss_fn(pr_v, vb)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                t_loss += loss.item(); t_p += (loss-value_loss_fn(pr_v, vb)).item(); t_v += value_loss_fn(pr_v, vb).item(); steps += 1
            print(f"  Epoch {epoch+1}/{args.train_epochs} | Loss: {t_loss/steps:.4f}")

        scheduler.step()
        if (cycle + 1) % args.eval_interval == 0 and not args.dry_run:
            wr = evaluate_models(model, best_model, device, args.eval_games, args.eval_mcts_sims)
            print(f"[eval] Win rate: {wr:.1%}")
            if wr >= 0.55:
                best_model.load_state_dict(model.state_dict())
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_9x9_best.pt"))

        log_entries.append({"cycle": cycle+1, "loss": t_loss/steps if steps>0 else 0, "black_win_pct": b_wins/total_g*100})
        if (cycle + 1) % args.save_interval == 0:
            torch.save({'cycle': cycle, 'model': model.state_dict(), 'best_model': best_model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'buffer_states': all_states[-50000:], 'buffer_policies': all_policies[-50000:], 'buffer_values': all_values[-50000:]}, os.path.join(CHECKPOINT_DIR, "latest.pt"))

        with open(LOG_FILE, 'w') as f: json.dump(log_entries, f, indent=2)
        print(f"[time] Total: {(time.time()-total_start)/3600:.1f}h", flush=True)

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_9x9_final.pt"))

def main():
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=100)
    parser.add_argument('--games-per-cycle', type=int, default=200)
    parser.add_argument('--mcts-sims', type=int, default=100)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cpu-selfplay', action='store_true')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--train-epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--augment', type=int, default=4)
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--eval-games', type=int, default=40)
    parser.add_argument('--eval-mcts-sims', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run: args.train_epochs, args.eval_interval, args.save_interval = 1, 1, 1
    train(args)

if __name__ == '__main__': main()

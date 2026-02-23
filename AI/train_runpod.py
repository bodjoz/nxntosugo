#!/usr/bin/env python3
"""
GPU-optimized AlphaZero training for 9x9 Torus Go.
Optimized for RunPod RTX 4090: Multiprocessing + Hyper-optimized Go logic.

This version uses multiple CPU workers to bypass the Python GIL,
each running games on the shared GPU.
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
from selfplay import play_game, augment_torus_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOARD_SIZE = 9
CHANNELS = 128
RES_BLOCKS = 5
IN_CHANNELS = 4
MAX_BUFFER = 150_000
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training_log.json"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class GoDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = torch.from_numpy(np.array(states)).float()
        self.policies = torch.from_numpy(np.array(policies)).float()
        self.values = torch.from_numpy(np.array(values)).float().unsqueeze(1)

    def __len__(self): return len(self.states)
    def __getitem__(self, idx): return self.states[idx], self.policies[idx], self.values[idx]

# ---------------------------------------------------------------------------
# Worker Function
# ---------------------------------------------------------------------------
def selfplay_worker(worker_args):
    """Plays one game on the specified device."""
    model_state_dict, mcts_sims, device_str, in_channels = worker_args
    device = torch.device(device_str)
    
    # Reload model on device
    model = TorusGoNet(size=BOARD_SIZE, channels=128, num_res_blocks=5, in_channels=in_channels).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    states, policies, values = play_game(
        model, mcts_simulations=mcts_sims, temperature=1.0,
        device=device, size=BOARD_SIZE, add_noise=True, in_channels=in_channels
    )
    
    res = values[0] if len(values) > 0 else 0.0
    return states, policies, values, res

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_models(current_model, best_model, device, num_games=40, mcts_sims=100):
    from torusgo import TorusGo
    from mcts import MCTS
    wins = 0
    current_model.eval()
    best_model.eval()
    for g in range(num_games):
        game = TorusGo(size=BOARD_SIZE)
        current_is_black = (g % 2 == 0)
        mcts_current = MCTS(current_model, num_simulations=mcts_sims, device=device)
        mcts_best = MCTS(best_model, num_simulations=mcts_sims, device=device)
        while not game.game_over:
            is_current_turn = (game.current_player == 1) == current_is_black
            action_probs = (mcts_current if is_current_turn else mcts_best).get_action_prob(game, temperature=0.0)
            game.step(int(np.argmax(action_probs)))
        reward = game.get_reward()
        if (current_is_black and reward > 0) or (not current_is_black and reward < 0): wins += 1
    return wins / num_games

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def train(args):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[train] Device: {device} | Mode: Parallel Multiprocessing ({args.workers} workers)")

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
            all_states = ckpt.get('buffer_states', [])
            all_policies = ckpt.get('buffer_policies', [])
            all_values = ckpt.get('buffer_values', [])
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE) as f: log_entries = json.load(f)
            print(f"[train] Resumed from cycle {start_cycle}")

    total_start = time.time()
    for cycle in range(start_cycle, args.cycles):
        cycle_start = time.time()
        print(f"\n{'='*60}\n  CYCLE {cycle+1}/{args.cycles} | Buffer: {len(all_states)} | LR: {optimizer.param_groups[0]['lr']:.6f}\n{'='*60}", flush=True)

        # 1. Self-play (Parallel)
        model.eval()
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        worker_args = [(cpu_state_dict, args.mcts_sims, str(device), IN_CHANNELS) for _ in range(args.games_per_cycle)]
        
        cycle_states, cycle_policies, cycle_values, game_results = [], [], [], []
        print(f"[selfplay] Playing {args.games_per_cycle} games with {args.workers} workers...", flush=True)
        completed = 0
        with mp.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(selfplay_worker, worker_args):
                s, p, v, res = result
                cycle_states.extend(s); cycle_policies.extend(p); cycle_values.extend(v); game_results.append(res)
                completed += 1
                if completed % 10 == 0:
                    print(f"  ... {completed}/{args.games_per_cycle} games done", flush=True)

        sp_time = time.time() - cycle_start
        b_wins = sum(1 for r in game_results if r > 0)
        total_g = len(game_results)
        print(f"[selfplay] Finished games in {sp_time/60:.1f}m | B: {b_wins/total_g:.1%} W: {(total_g-b_wins)/total_g:.1%}")

        # 2. Augment & Buffer
        if args.augment > 0:
            cycle_states, cycle_policies, cycle_values = augment_torus_data(cycle_states, cycle_policies, cycle_values, size=BOARD_SIZE, num_augments=args.augment)
        
        all_states = (all_states + cycle_states)[-MAX_BUFFER:]
        all_policies = (all_policies + cycle_policies)[-MAX_BUFFER:]
        all_values = (all_values + cycle_values)[-MAX_BUFFER:]

        # 3. Train
        dataset = GoDataset(all_states, all_policies, all_values)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))
        model.train()
        print(f"[train] Training on {len(dataset)} positions...", flush=True)
        for epoch in range(args.train_epochs):
            t_loss, steps = 0, 0
            for sb, pb, vb in dataloader:
                sb, pb, vb = sb.to(device), pb.to(device), vb.to(device)
                optimizer.zero_grad()
                pred_pi, pred_v = model(sb)
                loss = policy_loss_fn(pred_pi, pb) + value_loss_fn(pred_v, vb)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                t_loss += loss.item(); steps += 1
            print(f"  Epoch {epoch+1}/{args.train_epochs} | Loss: {t_loss/steps:.4f}")

        scheduler.step()
        if (cycle + 1) % args.eval_interval == 0 and not args.dry_run:
            wr = evaluate_models(model, best_model, device, args.eval_games, args.eval_mcts_sims)
            print(f"[eval] Win rate vs best: {wr:.1%}")
            if wr >= 0.55:
                best_model.load_state_dict(model.state_dict())
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_9x9_best.pt"))

        checkpoint = {
            'cycle': cycle, 'model': model.state_dict(), 'best_model': best_model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'buffer_states': all_states[-20000:], 'buffer_policies': all_policies[-20000:], 'buffer_values': all_values[-20000:]
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "latest.pt"))
        print(f"[cycle] Cycle finished in {(time.time()-cycle_start)/60:.1f}m. Total: {(time.time()-total_start)/3600:.1f}h")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_9x9_final.pt"))

def main():
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=100)
    parser.add_argument('--games-per-cycle', type=int, default=200)
    parser.add_argument('--mcts-sims', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--train-epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--augment', type=int, default=4)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--eval-games', type=int, default=40)
    parser.add_argument('--eval-mcts-sims', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run: args.games_per_cycle, args.train_epochs, args.eval_interval = 2, 1, 1
    train(args)

if __name__ == '__main__': main()

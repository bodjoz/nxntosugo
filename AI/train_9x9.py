import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from network import TorusGoNet
from selfplay import play_game

class GoDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = torch.tensor(np.array(states), dtype=torch.float32)
        self.policies = torch.tensor(np.array(policies), dtype=torch.float32)
        self.values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 9x9 needs more capacity
    model = TorusGoNet(size=9, channels=128, num_res_blocks=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Lower LR for stability
    
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    
    # Configured to finish in ~30-60min on M1 Air
    epochs = 5 
    games_per_epoch = 20
    mcts_simulations = 40 
    batch_size = 64
    training_epochs_per_cycle = 4
    
    all_states, all_policies, all_values = [], [], []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.eval()
        print(f"\n--- 9x9 Cycle {epoch+1}/{epochs} ---")
        
        print(f"Generating {games_per_epoch} self-play games for 9x9...")
        cycle_states, cycle_policies, cycle_values = [], [], []
        
        for g in range(games_per_epoch):
            s, p, v = play_game(model, mcts_simulations=mcts_simulations, device=device, size=9)
            cycle_states.extend(s)
            cycle_policies.extend(p)
            cycle_values.extend(v)
            if (g+1) % 5 == 0:
                print(f"  Game {g+1}/{games_per_epoch} completed.")
                
        max_buffer = 15000
        all_states = (all_states + cycle_states)[-max_buffer:]
        all_policies = (all_policies + cycle_policies)[-max_buffer:]
        all_values = (all_values + cycle_values)[-max_buffer:]
        
        dataset = GoDataset(all_states, all_policies, all_values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training on {len(all_states)} positions for {training_epochs_per_cycle} epochs...")
        model.train()
        
        for p_epoch in range(training_epochs_per_cycle):
            total_loss = 0
            total_p_loss = 0
            total_v_loss = 0
            
            for states, policies, values in dataloader:
                states, policies, values = states.to(device), policies.to(device), values.to(device)
                
                optimizer.zero_grad()
                pred_policies, pred_values = model(states)
                
                p_loss = policy_loss_fn(pred_policies, policies)
                v_loss = value_loss_fn(pred_values, values)
                loss = p_loss + v_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_p_loss += p_loss.item()
                total_v_loss += v_loss.item()
                
            steps = len(dataloader)
            print(f"  Update {p_epoch+1} | Loss: {total_loss/steps:.4f} (Pi: {total_p_loss/steps:.4f}, V: {total_v_loss/steps:.4f})")
            
        torch.save(model.state_dict(), f"model_9x9_checkpoint_c{epoch+1}.pt")
        
    end_time = time.time()
    print(f"\n9x9 Training completed in {(end_time - start_time)/60:.2f} minutes.")
    torch.save(model.state_dict(), "model_9x9_final.pt")

if __name__ == "__main__":
    train()

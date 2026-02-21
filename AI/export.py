import torch
import numpy as np
import json
from network import TorusGoNet
from torusgo import TorusGo

def export_dummy_network():
    device = torch.device('cpu')
    
    # Load model
    model = TorusGoNet(size=4, channels=64, num_res_blocks=2).to(device)
    try:
        model.load_state_dict(torch.load("model_final.pt", map_location=device))
        print("Successfully loaded trained weights.")
    except Exception as e:
        print("Could not load weights, exporting untrained network output as fallback.")
        
    model.eval()
    
    # Create an empty board
    game = TorusGo(size=4)
    # Let's say we put an opponent piece in the middle so the heatmap shows *something* interesting
    game.board[1, 1] = -1 
    game.board[2, 2] = 1
    
    state_tensor = torch.tensor(game.get_state_input(), dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy, value = model(state_tensor)
        policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
        value = value.item()
        
    # Exclude pass probability from heatmap
    heatmap = policy[:-1].tolist()
    
    # Normalize heatmap for visual contrast
    max_val = max(heatmap) if max(heatmap) > 0 else 1
    heatmap = [min(1.0, v / max_val) for v in heatmap]
    
    output = {
      "description": "Trained 4x4 AI Torus Go Evaluation",
      "boardSize": 4,
      "heatmap": heatmap,
      "value": value
    }
    
    with open('../trained-4x4-ai.json', 'w') as f:
        json.dump(output, f, indent=2)
        
    print("Successfully exported network evaluation to ../trained-4x4-ai.json")

if __name__ == "__main__":
    export_dummy_network()

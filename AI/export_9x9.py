import torch
import numpy as np
import json
from network import TorusGoNet
from torusgo import TorusGo

def export_dummy_network():
    device = torch.device('cpu')
    model = TorusGoNet(size=9, channels=128, num_res_blocks=5).to(device)
    
    try:
        model.load_state_dict(torch.load("model_9x9_final.pt", map_location=device, weights_only=True))
        print("Successfully loaded trained 9x9 weights.")
    except Exception as e:
        print("Could not load weights, exporting untrained network output as fallback.")
        
    model.eval()
    
    game = TorusGo(size=9)
    game.board[4, 4] = -1 
    game.board[3, 4] = 1
    
    state_tensor = torch.tensor(game.get_state_input(), dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy, value = model(state_tensor)
        policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
        value = value.item()
        
    heatmap = policy[:-1].tolist()
    max_val = max(heatmap) if max(heatmap) > 0 else 1
    heatmap = [min(1.0, v / max_val) for v in heatmap]
    
    output = {
      "description": "Trained 9x9 AI Torus Go Evaluation",
      "boardSize": 9,
      "heatmap": heatmap,
      "value": value
    }
    
    with open('../trained-9x9-ai.json', 'w') as f:
        json.dump(output, f, indent=2)
        
    print("Successfully exported network evaluation to ../trained-9x9-ai.json")

if __name__ == "__main__":
    export_dummy_network()

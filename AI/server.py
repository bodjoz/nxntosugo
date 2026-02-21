from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np

from network import TorusGoNet
from torusgo import TorusGo
from mcts import MCTS

app = Flask(__name__)
target_size = 4
CORS(app) # Allow React frontend to access

# Initialize and load model globally
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Loading trained model on {device}...")
model = TorusGoNet(size=target_size, channels=64, num_res_blocks=2).to(device)

try:
    model.load_state_dict(torch.load("model_final.pt", map_location=device))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model_final.pt: {e}")
    
def parse_board(state_data):
    """Converts 1D JS array back to Python TorusGo state."""
    if len(state_data['board']) != target_size * target_size:
        raise ValueError(f"Board size must be {target_size}x{target_size}")
        
    game = TorusGo(size=target_size)
    # JS array is [0, 1, -1] representing Empty, Black, White
    board_1d = np.array(state_data['board'], dtype=np.int8)
    game.board = board_1d.reshape((target_size, target_size))
    game.current_player = int(state_data['currentPlayer'])
    return game

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Returns the pure Neural Network evaluation (Heatmap & Value) for a given state."""
    try:
        data = request.json
        game = parse_board(data)
        
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
        
        return jsonify({
            "heatmap": heatmap,
            "value": value
        })
        
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/play', methods=['POST'])
def play():
    """Uses MCTS + Neural Network to select the best move for the given board state."""
    try:
        data = request.json
        game = parse_board(data)
        
        # We only need ~40 simulations for decent pseudo-instant 4x4 self-play
        mcts = MCTS(model, num_simulations=40, device=device)
        
        # Temperature=0 for competitive greedy play
        action_probs = mcts.get_action_prob(game, temperature=0.0)
        
        best_action = int(np.argmax(action_probs))
        
        is_pass = best_action == target_size * target_size
        
        return jsonify({
            "action": best_action,
            "isPass": is_pass,
            "x": None if is_pass else best_action % target_size,
            "y": None if is_pass else best_action // target_size
        })
        
    except Exception as e:
        print(f"Play Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print(f"Starting Torus Go AI inference server on port 5001...")
    app.run(host='0.0.0.0', port=5001)

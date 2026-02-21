from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np

from network import TorusGoNet
from torusgo import TorusGo
from mcts import MCTS

app = Flask(__name__)
CORS(app) # Allow React frontend to access

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
models = {}

def get_model(size):
    if size in models:
        return models[size]
        
    print(f"Loading {size}x{size} trained model on {device}...")
    if size == 9:
        model = TorusGoNet(size=size, channels=128, num_res_blocks=5).to(device)
        model_path = "model_9x9_final.pt"
    else:
        model = TorusGoNet(size=size, channels=64, num_res_blocks=2).to(device)
        model_path = "model_final.pt"
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"Model {model_path} loaded successfully.")
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        
    models[size] = model
    return model

def parse_board(state_data):
    """Converts 1D JS array back to Python TorusGo state."""
    size = int(np.sqrt(len(state_data['board'])))
    if size not in [4, 9]:
        raise ValueError("Only 4x4 and 9x9 board sizes are supported for AI.")
        
    game = TorusGo(size=size)
    board_1d = np.array(state_data['board'], dtype=np.int8)
    game.board = board_1d.reshape((size, size))
    game.current_player = int(state_data['currentPlayer'])
    return game

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.json
        game = parse_board(data)
        model = get_model(game.size)
        
        state_tensor = torch.tensor(game.get_state_input(), dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            policy, value = model(state_tensor)
            policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
            value = value.item()
            
        heatmap = policy[:-1].tolist()
        max_val = max(heatmap) if max(heatmap) > 0 else 1
        heatmap = [min(1.0, v / max_val) for v in heatmap]
        
        return jsonify({"heatmap": heatmap, "value": value})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/play', methods=['POST'])
def play():
    try:
        data = request.json
        game = parse_board(data)
        model = get_model(game.size)
        
        # 40 simulations for decent self-play / inference speed
        mcts = MCTS(model, num_simulations=40, device=device)
        action_probs = mcts.get_action_prob(game, temperature=0.0)
        best_action = int(np.argmax(action_probs))
        
        is_pass = best_action == game.size * game.size
        
        return jsonify({
            "action": best_action,
            "isPass": is_pass,
            "x": None if is_pass else best_action % game.size,
            "y": None if is_pass else best_action // game.size
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print(f"Starting Torus Go AI inference server on port 5001...")
    app.run(host='0.0.0.0', port=5001)

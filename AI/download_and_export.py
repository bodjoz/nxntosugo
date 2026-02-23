#!/usr/bin/env python3
"""
Download a RunPod-trained checkpoint and set it up for local inference.
Handles the 4-channel → 2-channel adapter for the existing server.py.

Usage:
    python download_and_export.py checkpoints/model_9x9_best.pt
"""
import sys
import os
import torch
import numpy as np
import json
from network import TorusGoNet
from torusgo import TorusGo

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_and_export.py <path_to_checkpoint.pt>")
        print("  e.g.: python download_and_export.py checkpoints/model_9x9_best.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found")
        sys.exit(1)

    device = torch.device('cpu')

    # Load the 4-channel model
    model = TorusGoNet(size=9, channels=128, num_res_blocks=5, in_channels=4).to(device)
    
    # PyTorch 2.6+ defaults to weights_only=True, which can fail if the checkpoint 
    # contains certain types (like numpy scalars). We'll allow the necessary global.
    # Note: Using np._core to avoid DeprecationWarning in newer NumPy versions.
    if hasattr(torch.serialization, 'add_safe_globals'):
        try:
            torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
        except AttributeError:
            # Fallback for older NumPy versions
            torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle full training checkpoints which wrap the model in a 'model' key
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        print("📦 Detected full training checkpoint, extracting model state...")
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Loaded 4-channel model from {checkpoint_path}")

    # Save as the standard model file for the updated server
    output_path = "model_9x9_trained.pt"
    torch.save(model.state_dict(), output_path)
    print(f"✅ Saved model to {output_path}")

    # Quick sanity check — run inference on a sample position
    game = TorusGo(size=9)
    game.board[4 * 9 + 4] = -1
    game.board[3 * 9 + 4] = 1

    state = game.get_state_input(in_channels=4, move_number=5)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        policy, value = model(state_tensor)
        policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
        value = value.item()

    # Export heatmap JSON for the UI
    heatmap = policy[:-1].tolist()
    max_val = max(heatmap) if max(heatmap) > 0 else 1
    heatmap = [min(1.0, v / max_val) for v in heatmap]

    output_json = {
        "description": "RunPod-trained 9x9 Torus Go AI",
        "boardSize": 9,
        "model_file": output_path,
        "in_channels": 4,
        "channels": 128,
        "res_blocks": 5,
        "heatmap": heatmap,
        "value": value,
    }

    json_path = '../trained-9x9-ai.json'
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"✅ Exported sample heatmap to {json_path}")

    # Print top 5 moves
    board_policy = policy[:-1].reshape(9, 9)
    top_indices = np.argsort(policy[:-1])[::-1][:5]
    print(f"\n📊 Value: {value:.3f} ({'Black advantage' if value > 0 else 'White advantage'})")
    print("📍 Top 5 moves:")
    for idx in top_indices:
        r, c = divmod(idx, 9)
        print(f"   ({r},{c}) = {policy[idx]:.4f}")

    print(f"\n🎮 Model ready! Update server.py to use in_channels=4 and model file '{output_path}'")
    print("   Then run: python server.py")


if __name__ == '__main__':
    main()

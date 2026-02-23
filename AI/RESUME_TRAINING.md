# 🔄 How to Resume & Continue Training

If your training stalled or you want to continue from where you left off, follow these steps.

## 1. Prepare the Optimized Script
I have updated `train_runpod.py` to be more efficient when saving checkpoints.
1.  **Re-upload** the updated `AI/train_runpod.py` to your RunPod instance.
2.  Make sure your `latest.pt` file is in the `checkpoints/` folder on RunPod.

## 2. Run with the Resume Flag
Use the `--resume` flag to tell the script to load the previous state (optimizer, scheduler, and experience buffer) instead of starting fresh.

### Command:
```bash
python3 train_runpod.py --resume --workers 8 --games-per-cycle 200
```

### What happens next?
- The script will load `checkpoints/latest.pt`.
- It will show: `[train] Resumed from cycle X`.
- It will continue exactly where it left off, including current learning rates.

---

## 🚀 Pro Tips for Faster Training
If you find the current speed too slow on a high-end GPU like the RTX 4090:

1.  **Increase Workers**: You can try `--workers 12` or `--workers 16` if your CPU has enough cores (RunPod usually gives you plenty).
2.  **Adjust Cycle Length**: 
    - Fewer games per cycle (`--games-per-cycle 100`) means the model gets updated more frequently.
    - More games per cycle (`--games-per-cycle 400`) means the "Value" evaluation will be more stable.
3.  **Check GPU Utilization**: Run `nvidia-smi` in your RunPod terminal to see if the GPU is staying busy. If it's below 50%, increase `--workers`.

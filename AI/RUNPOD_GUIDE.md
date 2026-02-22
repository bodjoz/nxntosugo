# RunPod Training Guide — 9x9 Torus Go

Step-by-step instructions to train your 9x9 Torus Go AI on RunPod GPU cloud.

## Prerequisites

- A RunPod account ([runpod.io](https://runpod.io))
- ~$25-30 in credits loaded
- This project's code (the `AI/` directory)

---

## Step 1: Create a RunPod Account & Add Credits

1. Go to [runpod.io](https://runpod.io) and sign up
2. Go to **Billing** → **Add Credits**
3. Add **$25-30** (training will cost ~$10-20, extra for safety)

---

## Step 2: Create a GPU Pod

1. Go to **Pods** → **+ Deploy**
2. Select **Community Cloud** (cheaper)
3. Choose GPU: **RTX 4090** (best price/performance, ~$0.39/hr)
   - Alternative: RTX 3090 (~$0.22/hr, ~50% slower)
4. Template: **RunPod Pytorch 2.4** (or any PyTorch CUDA template)
5. Container Disk: **20 GB** (default is fine)
6. Volume Disk: **20 GB** (for persistent storage of checkpoints)
7. Click **Deploy**

Wait ~1 minute for the pod to start.

---

## Step 3: Upload Your Code

Click **Connect** → **Web Terminal** on your pod.

### Option A: Git clone (if your repo is on GitHub)
```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/nxntorusgo.git
cd nxntorusgo/AI
```

### Option B: Upload manually via runpodctl
On your **local Mac**:
```bash
# Install runpodctl (one-time)
brew install runpod/runpodctl/runpodctl

# Get your pod ID from the RunPod dashboard
# Upload the AI directory
runpodctl send AI/ --podId YOUR_POD_ID
```

### Option C: Copy-paste via web terminal
In the RunPod web terminal:
```bash
cd /workspace
mkdir -p torusgo && cd torusgo

# Create each file by pasting contents
# (use nano or cat << 'EOF' > filename ... EOF for each file)
```

---

## Step 4: Install Dependencies

```bash
cd /workspace/nxntorusgo/AI   # or wherever you put your code
pip install numpy tensorboard
```

> PyTorch + CUDA is already installed in the pod template.

---

## Step 5: Start Training

### Recommended full training run (~24-48 hours):
```bash
nohup python -u train_runpod.py \
    --cycles 100 \
    --games-per-cycle 500 \
    --mcts-sims 200 \
    --workers 12 \
    --batch-size 256 \
    --augment 4 \
    > training.log 2>&1 &
```

The `nohup` ensures training continues even if you close the terminal.

### Quick test (verify everything works, ~2 minutes):
```bash
python train_runpod.py --dry-run --cycles 1 --games-per-cycle 2 --mcts-sims 10 --workers 2
```

---

## Step 6: Monitor Progress

### Watch live output:
```bash
tail -f training.log
```

### Check training metrics:
```bash
python -c "import json; d=json.load(open('training_log.json')); [print(f'C{e[\"cycle\"]:3d} | Loss:{e[\"loss\"]:.4f} Pi:{e[\"pi_loss\"]:.4f} V:{e[\"v_loss\"]:.4f} | {e[\"cycle_time_s\"]/60:.1f}min') for e in d]"
```

### What to look for:
- **Policy loss** should decrease from ~4.0 to ~2.0-2.5 over training
- **Value loss** should decrease from ~0.5 to ~0.1-0.2
- **Cycle time** should be ~15-25 min per cycle on RTX 4090
- **Eval wins** (every 10 cycles) should show >55% win rate for model updates

---

## Step 7: Download the Trained Model

When training is done (or strong enough based on loss values):

### From RunPod web terminal:
```bash
ls -la checkpoints/
# You should see: model_9x9_best.pt, model_9x9_final.pt, latest.pt
```

### Download to your Mac:

**Option A: runpodctl**
```bash
# On your Mac:
runpodctl receive checkpoints/model_9x9_best.pt --podId YOUR_POD_ID
```

**Option B: RunPod file browser**
Use the RunPod dashboard → your pod → File Manager to download `checkpoints/model_9x9_best.pt`

**Option C: SCP**
```bash
# Get SSH details from RunPod dashboard → Connect → SSH
scp -P PORT root@IP:/workspace/nxntorusgo/AI/checkpoints/model_9x9_best.pt ./AI/checkpoints/
```

---

## Step 8: Set Up Locally

On your Mac, in the `AI/` directory:

```bash
# Place the downloaded model
cp checkpoints/model_9x9_best.pt ./

# Run the export script
python download_and_export.py checkpoints/model_9x9_best.pt

# Start the server
python server.py
```

The server will auto-detect the 4-channel trained model. Play against it in the browser!

---

## Step 9: Stop the Pod!

> ⚠️ **Don't forget to stop or terminate your pod when training is done!**

Go to RunPod dashboard → your pod → **Stop** (or **Terminate** if you've downloaded everything).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `--batch-size` to 128 or `--workers` to 8 |
| `ConnectionError` on workers | Reduce `--workers` to 4-8 |
| Training too slow | Increase `--workers`, decrease `--mcts-sims` to 100 |
| Want to stop early | `Ctrl+C` or `kill %1`, model is checkpointed every 5 cycles |
| Resume after crash | Just add `--resume` flag and rerun the same command |

---

## Cost Guide

| Duration | RTX 4090 Cost | Expected Strength |
|----------|---------------|-------------------|
| 12 hours (~25 cycles) | ~$5 | Basic: captures, simple patterns |
| 24 hours (~50 cycles) | ~$10 | Moderate: territory sense, life/death basics |
| 48 hours (~100 cycles) | ~$20 | Strong: ~10-15 kyu equivalent |
| 72 hours (~150 cycles) | ~$28 | Stronger: approaching single-digit kyu |

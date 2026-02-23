# 🎯 9x9 Torus Go: Complete Deployment & Update Guide

This guide covers everything from downloading a new training result to playing against it in your browser.

---

## 📋 1. Prerequisites
Ensure you have these terminal windows ready:
1.  **Backend Terminal**: For running `server.py` (the AI logic).
2.  **Frontend Terminal**: For running `npm run dev` (the web UI).

---

## 📥 2. How to Download from RunPod
If you see `checkpoints/latest.pt` on your RunPod instance:
1.  Download it to your local computer.
2.  Move it into your local project folder at:
    `AI/checkpoints/latest.pt`
    *(Delete the old one first if you want to be sure)*

> [!WARNING]
> **If your RunPod log has not updated for hours:**
> This usually means the training loop stalled during the "Save" or "Evaluation" phase (a known issue with large data buffers). The `latest.pt` file on RunPod is likely still valid and contains the weights from the **previous** completed cycle. You can safely download and use it.

---

## 🛠️ 3. How to "Unpack" (Export) the Model
The `latest.pt` file you download is a "Full Checkpoint" (120MB+). It contains training data you don't need for playing. You must export just the "Brain" (Weights).

1.  Open your **Backend Terminal**.
2.  Go to the AI folder: `cd AI`
3.  Run the export command:
    ```bash
    ./venv/bin/python download_and_export.py checkpoints/latest.pt
    ```
4.  **Confirm success**: You should see:
    - 📦 `Detected full training checkpoint, extracting model state...`
    - ✅ `Saved model to model_9x9_trained.pt`
    - ✅ `Exported sample heatmap to ../trained-9x9-ai.json`

---

## 🖥️ 4. How to Start the Backend (AI Server)
The backend is a Flask server that "runs" the AI and answers requests from the browser.

1.  In the **Backend Terminal** (still in the `AI` folder):
2.  Run:
    ```bash
    ./venv/bin/python server.py
    ```
3.  **Verify**: Look for this line:
    `✅ Model .../model_9x9_trained.pt loaded successfully.`
    *(If you see an error here, it means you skipped Step 3)*

---

## 🌐 5. How to Initiate the UI & Play
1.  Open your **Frontend Terminal** (in the root `nxntorusgo` folder).
2.  Run:
    ```bash
    npm run dev
    ```
3.  Open your browser to the URL shown (usually `http://localhost:3000`).
4.  **In the Browser**:
    - Look at the sidebar on the right.
    - Under **AI Opponent**, change "None" to **Black** or **White**.
    - **Refresh the page** if the AI doesn't move immediately.
    - You should see a colorful "Heatmap" on the board showing the AI's thoughts!

---

## ❓ FAQ & Troubleshooting
- **"I see a CORS error in the console"**: This means your `server.py` is NOT running. Go back to Step 4.
- **"The AI takes forever to move"**: The AI uses MCTS simulations. If it's too slow, we can reduce the `num_simulations` in `server.py` (it's currently set to 40).
- **"The board is empty but AI won't play"**: Make sure you selected the AI color. Also, verify `server.py` shows it loaded the `in_channels=4` model.

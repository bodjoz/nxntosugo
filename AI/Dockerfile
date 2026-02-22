FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements_gpu.txt .
RUN pip install --no-cache-dir -r requirements_gpu.txt

# Copy training code
COPY network.py torusgo.py mcts.py selfplay.py train_runpod.py ./

# Create checkpoint directory
RUN mkdir -p checkpoints

# Default: full training run with recommended settings
ENTRYPOINT ["python", "-u", "train_runpod.py"]
CMD ["--cycles", "100", "--games-per-cycle", "500", "--mcts-sims", "200", "--workers", "12"]

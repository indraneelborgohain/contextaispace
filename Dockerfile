# ── GPT-OSS 20B — RunPod Serverless Docker Image ─────────────────────────
#
# Build:
#   docker build -t gptoss-runpod .
#
# Run locally (for testing):
#   docker run --gpus all -p 8000:8000 gptoss-runpod
#
# With HuggingFace model download at startup:
#   docker run --gpus all -e HF_REPO_ID="your-org/gptoss-20b" \
#              -e HF_TOKEN="hf_xxx" -p 8000:8000 gptoss-runpod
#
# With pre-baked weights (mount volume):
#   docker run --gpus all -v /path/to/weights:/app/model/gpt-oss-20b/original \
#              -p 8000:8000 gptoss-runpod
# ──────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System dependencies ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ─────────────────────────────────────────────────
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install runpod

# ── Copy application code ───────────────────────────────────────────────
COPY architecture/ architecture/
COPY app/ app/
COPY training/ training/
COPY inference.py .
COPY system_generator.py .
COPY hf_gptoss_loader.py .
COPY transfer_weights.py .
COPY rp_handler.py .

# ── Default environment variables ────────────────────────────────────────
# MODEL_DIR: path inside the container where checkpoint shards live
# HF_REPO_ID: set this to auto-download weights from HuggingFace on first run
ENV MODEL_DIR="model/gpt-oss-20b/original/" \
    HF_REPO_ID="" \
    HF_LOCAL_DIR="models/gptoss-20b"

# ── Expose port (used only when testing locally, RunPod manages networking) ─
EXPOSE 8000

# ── Entry point ──────────────────────────────────────────────────────────
CMD ["python3", "rp_handler.py"]

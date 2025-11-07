# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ARG TASK=question-answering
ARG MODEL_ID=deepset/minilm-uncased-squad2
ARG GEN_MODEL_ID=google/flan-t5-small

# --- Runtime env (keep OFFLINE flags unset until after downloads) ---
ENV TASK=${TASK} \
    MODEL_ID=${MODEL_ID} \
    GEN_MODEL_ID=${GEN_MODEL_ID} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=9090 \
    UVICORN_LOG_LEVEL=info

# OS deps (libgomp fixes PyTorch runtime on slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libgomp1 libstdc++6 tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# Model cache location
RUN mkdir -p /models

# Download/collapse models into the image (ONLINE during build)
COPY download_model.py .

# 1) Extractive QA model
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    TASK=${TASK} MODEL_ID=${MODEL_ID} \
    python /app/download_model.py

# 2) Abstractive (text2text-generation) model
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    TASK=text2text-generation MODEL_ID=${GEN_MODEL_ID} \
    python /app/download_model.py

# App code
COPY app.py .

# After models are baked, lock runtime to OFFLINE
ENV OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 9090

# Use tini for clean signal handling; run uvicorn directly (no extra entrypoint file required)
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9090"]

# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models

# System deps: ffmpeg for MP4, libgomp for torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl ca-certificates libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Make pip tooling solid, then install torch (CPU) first
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# Diffusers stack (pinned for compatibility)
RUN pip install --no-cache-dir \
      diffusers==0.30.0 \
      transformers==4.44.2 \
      accelerate==0.34.2 \
      safetensors==0.4.5 \
      imageio==2.34.1 \
      imageio-ffmpeg==0.4.9

WORKDIR /app
ENV OUTPUT_DIR=/outputs
COPY hello_video_gen.py .

CMD ["python", "hello_video_gen.py"]

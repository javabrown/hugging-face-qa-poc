#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="hf-qa-poc:cpu"
CONTAINER_NAME="hf-qa-poc"
TASK="question-answering"
MODEL_ID="deepset/minilm-uncased-squad2"
HOST_PORT="${PORT:-9090}"

echo "[run.sh] TASK=$TASK MODEL_ID=$MODEL_ID PORT=$HOST_PORT"

# Clean prior container
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# Rebuild fresh to be safe (comment out to speed up)
docker rmi -f "$IMAGE_TAG" >/dev/null 2>&1 || true
docker build \
  --build-arg TASK="$TASK" \
  --build-arg MODEL_ID="$MODEL_ID" \
  -t "$IMAGE_TAG" .

# Foreground run so you see logs; Ctrl+C stops it
docker run --name "$CONTAINER_NAME" \
  -e TASK="$TASK" \
  -e MODEL_ID="$MODEL_ID" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e UVICORN_PORT=9090 \
  -p "${HOST_PORT}:9090" \
  "$IMAGE_TAG"

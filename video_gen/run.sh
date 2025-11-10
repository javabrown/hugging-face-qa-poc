#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="hf-video-hello"
CONTAINER_NAME="hf-video-hello-container"
OUT_DIR="$(pwd)/outputs"

PROMPT="${PROMPT:-a cat playing piano, cinematic lighting, 4k}"
VIDEO_MODEL_ID="${VIDEO_MODEL_ID:-damo-vilab/text-to-video-ms-1.7b}"

echo "[run.sh] Clean old container..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "[run.sh] Remove old image (if exists)..."
docker rmi -f "$IMAGE_TAG" >/dev/null 2>&1 || true

echo "[run.sh] Build image..."
docker build -t "$IMAGE_TAG" .

# Use GPU if available
GPU_FLAG=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_FLAG="--gpus all"
  echo "[run.sh] NVIDIA GPU detected; enabling --gpus all"
else
  echo "[run.sh] No GPU detected; running on CPU (very slow)"
fi

mkdir -p "$OUT_DIR"

echo "[run.sh] Run container (output -> $OUT_DIR/output_video.mp4)..."
docker run --name "$CONTAINER_NAME" --rm \
  $GPU_FLAG \
  -e PROMPT="$PROMPT" \
  -e VIDEO_MODEL_ID="$VIDEO_MODEL_ID" \
  -e OUTPUT_DIR=/outputs \
  -e NUM_STEPS="${NUM_STEPS:-25}" \
  -e FPS="${FPS:-8}" \
  -v "$OUT_DIR":/outputs \
  "$IMAGE_TAG"

echo "[run.sh] Done. Check $OUT_DIR/output_video.mp4"

#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="hf-image_gen"
CONTAINER_NAME="hf-image_gen-container"
OUT_DIR="$(pwd)/outputs"

echo "[run.sh] Cleaning up old container..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "[run.sh] Removing old image if exists..."
docker rmi -f "$IMAGE_TAG" >/dev/null 2>&1 || true

echo "[run.sh] Building image..."
docker build -t "$IMAGE_TAG" .

echo "[run.sh] Running container (saving to $OUT_DIR)..."
mkdir -p "$OUT_DIR"
docker run --name "$CONTAINER_NAME" --rm \
  -e OUTPUT_DIR=/outputs \
  -e PROMPT="${PROMPT:-a cute robot holding a flower, digital art}" \
  -v "$OUT_DIR":/outputs \
  "$IMAGE_TAG"

echo "[run.sh] Done. Check: $OUT_DIR/output.png"

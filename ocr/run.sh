#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="hf-ocr-hello"
CONTAINER_NAME="hf-ocr-hello-container"
IN_DIR="$(pwd)/inputs"
OUT_DIR="$(pwd)/outputs"
IN_FILE="$IN_DIR/sample.jpg"
OUT_FILE="$OUT_DIR/ocr.txt"

echo "[run.sh] Preparing folders…"
mkdir -p "$IN_DIR" "$OUT_DIR"

if [[ ! -f "$IN_FILE" ]]; then
  echo "[run.sh] ERROR: Input image not found: $IN_FILE"
  echo "          Put your image at ./inputs/sample.jpg and rerun."
  exit 1
fi

echo "[run.sh] Cleaning old container…"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "[run.sh] Removing old image…"
docker rmi -f "$IMAGE_TAG" >/dev/null 2>&1 || true

echo "[run.sh] Building image…"
docker build -t "$IMAGE_TAG" .

echo "[run.sh] Running OCR (input -> $IN_FILE, output -> $OUT_FILE)…"
docker run --name "$CONTAINER_NAME" --rm \
  -v "$IN_DIR":/inputs \
  -v "$OUT_DIR":/outputs \
  -e INPUT_IMAGE=/inputs/sample.jpg \
  -e OUTPUT_FILE=/outputs/ocr.txt \
  "$IMAGE_TAG"

echo "[run.sh] Done. See $OUT_FILE"

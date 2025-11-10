#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="hf-hello"
CONTAINER_NAME="hf-hello-container"

echo "[run.sh] Cleaning up old container..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "[run.sh] Removing old image if exists..."
docker rmi -f "$IMAGE_TAG" >/dev/null 2>&1 || true

echo "[run.sh] Building image..."
docker build -t "$IMAGE_TAG" .

echo "[run.sh] Running container..."
docker run --name "$CONTAINER_NAME" --rm "$IMAGE_TAG"

echo "[run.sh] Done."

#!/usr/bin/env bash
set -euo pipefail

# Convert potential CRLF -> LF just in case (no-op on Linux)
if command -v sed >/dev/null 2>&1; then
  sed -i 's/\r$//' /app/entrypoint.sh || true
fi

echo "============================================================"
echo "HF QA Service – Entrypoint"
echo "Model_ID: ${MODEL_ID:-unset}"
echo "Task    : ${TASK:-unset}"
echo "HF_HOME : ${HF_HOME:-unset}"
echo "Offline : HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}"
echo "Models  :"
ls -lh /models || true
echo "============================================================"

# Quick import/version sanity check with explicit logging
python - <<'PY'
import sys, os, traceback
def log(k,v): print(f"[check] {k}: {v}")
try:
    import torch, transformers
    log("torch", torch.__version__)
    log("transformers", transformers.__version__)
    print("[check] imports OK")
except Exception as e:
    print("[FATAL] Python import failed:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(12)
PY

echo "[entrypoint] launching uvicorn..."
exec python -m uvicorn app:app \
  --host "${UVICORN_HOST:-0.0.0.0}" \
  --port "${UVICORN_PORT:-9090}" \
  --log-level "${UVICORN_LOG_LEVEL:-info}"

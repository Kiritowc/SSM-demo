#!/bin/bash
set -euo pipefail
SSM_ROOT="${SSM_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export SSM_ROOT
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
VLM_HOST="${VLM_HOST:-127.0.0.1}"
VLM_PORT="${VLM_PORT:-8080}"
MODEL_PATH="$SSM_ROOT/vlm/artifacts/models/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH="$SSM_ROOT/vlm/artifacts/models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
LOG_PATH="$SSM_ROOT/vlm/artifacts/llama-server.log"
MODEL_TAG="Qwen3VL-2B-Instruct-Q4_K_M.gguf"
mkdir -p "$(dirname "$LOG_PATH")"

missing=()
for f in "$MODEL_PATH" "$MMPROJ_PATH"; do
  if [[ ! -f "$f" ]]; then
    missing+=("$f")
  fi
done
if ((${#missing[@]} > 0)); then
  echo "error: VLM model files not found:" >&2
  for f in "${missing[@]}"; do
    echo "  - $f" >&2
  done
  echo "Place gguf/mmproj under vlm/artifacts/models/ (see README)." >&2
  exit 1
fi

if ! command -v "$LLAMA_SERVER" >/dev/null 2>&1; then
  echo "error: llama-server not found: $LLAMA_SERVER" >&2
  echo "Set LLAMA_SERVER to the binary path." >&2
  exit 1
fi

pkill -f "$MODEL_TAG" 2>/dev/null || true
nohup "$LLAMA_SERVER" \
  --host "$VLM_HOST" --port "$VLM_PORT" --parallel 1 \
  --ctx-size 1024 --batch-size 64 --ubatch-size 16 \
  --threads 6 --threads-batch 6 \
  --gpu-layers 99 --flash-attn on \
  --no-cache-prompt --cache-ram 0 --image-max-tokens 64 \
  -m "$MODEL_PATH" --mmproj "$MMPROJ_PATH" \
  > "$LOG_PATH" 2>&1 &
echo "llama-server started at http://${VLM_HOST}:${VLM_PORT}/"

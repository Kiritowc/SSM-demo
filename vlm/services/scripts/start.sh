#!/bin/bash
set -euo pipefail
SSM_ROOT="${SSM_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export SSM_ROOT
VLM_HOST="${VLM_HOST:-127.0.0.1}"
VLM_PORT="${VLM_PORT:-8080}"
MODEL_PATH="$SSM_ROOT/vlm/artifacts/models/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH="$SSM_ROOT/vlm/artifacts/models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
LOG_PATH="$SSM_ROOT/vlm/artifacts/llama-server.log"
MODEL_TAG="Qwen3VL-2B-Instruct-Q4_K_M.gguf"
mkdir -p "$(dirname "$LOG_PATH")"

if [[ -z "${LLAMA_SERVER:-}" ]]; then
  if [[ -x "$HOME/llama.cpp/build/bin/llama-server" ]]; then
    LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
  else
    LLAMA_SERVER=llama-server
  fi
fi

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
  echo "Set LLAMA_SERVER to the binary path (see configs/platform.yaml)." >&2
  exit 1
fi

if command -v ss >/dev/null 2>&1; then
  if ss -tln | grep -q ":${VLM_PORT} "; then
    echo "error: port ${VLM_PORT} already in use; stop the existing service first:" >&2
    ss -tlnp | grep ":${VLM_PORT} " >&2 || true
    exit 1
  fi
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

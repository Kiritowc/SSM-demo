#!/bin/bash
set -euo pipefail
SSM_ROOT="${SSM_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
export SSM_ROOT
LLAMA_SERVER="${LLAMA_SERVER:-/home/sunshink/llama.cpp/build/bin/llama-server}"
MODEL_PATH="$SSM_ROOT/artifacts/vlm/models/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH="$SSM_ROOT/artifacts/vlm/models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
LOG_PATH="$SSM_ROOT/artifacts/vlm/llama-server.log"
MODEL_TAG="Qwen3VL-2B-Instruct-Q4_K_M.gguf"
mkdir -p "$(dirname "$LOG_PATH")"
pkill -f "$MODEL_TAG" 2>/dev/null || true
nohup "$LLAMA_SERVER" \
  --host 127.0.0.1 --port 8080 --parallel 1 \
  --ctx-size 1024 --batch-size 64 --ubatch-size 16 \
  --threads 6 --threads-batch 6 \
  --gpu-layers 99 --flash-attn on \
  --no-cache-prompt --cache-ram 0 --image-max-tokens 64 \
  -m "$MODEL_PATH" --mmproj "$MMPROJ_PATH" \
  > "$LOG_PATH" 2>&1 &
echo "llama-server started at http://127.0.0.1:8080/"

#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/sunshink/ssdet VLM"
LLAMA_SERVER="/home/sunshink/llama.cpp/build/bin/llama-server"
MODEL_PATH="$REPO_ROOT/vlm/Qwen3VL/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH="$REPO_ROOT/vlm/Qwen3VL/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
LOG_PATH="$REPO_ROOT/vlm/logs/llama-server.log"
MODEL_TAG="Qwen3VL-2B-Instruct-Q4_K_M.gguf"

mkdir -p "$(dirname "$LOG_PATH")"
pkill -f "$MODEL_TAG" 2>/dev/null || true

if [[ ! -f "$LLAMA_SERVER" ]]; then
  echo "未找到 llama-server，路径为: $LLAMA_SERVER" >&2
  echo "请先在 llama.cpp 目录编译生成该文件，或修改本脚本里的 LLAMA_SERVER。" >&2
  exit 1
fi
if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server 存在但不可执行: $LLAMA_SERVER" >&2
  echo "可尝试: chmod +x \"$LLAMA_SERVER\"" >&2
  exit 1
fi

nohup "$LLAMA_SERVER" \
  --host 127.0.0.1 --port 8080 --parallel 1 \
  --ctx-size 1024 --batch-size 64 --ubatch-size 16 \
  --threads 6 --threads-batch 6 \
  --gpu-layers 99 --flash-attn on \
  --no-cache-prompt \
  --cache-ram 0 \
  --image-max-tokens 64 \
  -m "$MODEL_PATH" \
  --mmproj "$MMPROJ_PATH" \
  > "$LOG_PATH" 2>&1 &

echo "llama-server started at http://127.0.0.1:8080/"

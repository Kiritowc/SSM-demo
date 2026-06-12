#!/bin/bash
set -euo pipefail

MODEL_TAG="Qwen3VL-2B-Instruct-Q4_K_M.gguf"

# 只杀加载该模型的 llama-server，避免整条路径里有空格导致 pkill 正则踩坑。
pkill -f "$MODEL_TAG" 2>/dev/null || true

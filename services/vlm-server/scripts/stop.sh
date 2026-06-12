#!/bin/bash
pkill -f "Qwen3VL-2B-Instruct-Q4_K_M.gguf" 2>/dev/null || true
echo "llama-server stopped"

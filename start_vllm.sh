#!/usr/bin/env bash
# Start vLLM serving Qwen2.5-Coder-7B-Instruct-AWQ (4-bit quantized)
# Fits in 8GB VRAM (RTX 3070 Ti)
#
# Usage: ./start_vllm.sh
# The model will be downloaded on first run (~4.5GB).

set -e

MODEL="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
PORT=8000
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.85

echo "[vllm] Starting Qwen2.5-Coder-7B-Instruct-AWQ on port ${PORT}..."
echo "[vllm] GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "[vllm] Max context length: ${MAX_MODEL_LEN}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --quantization awq \
    --dtype half \
    --enforce-eager \
    --trust-remote-code

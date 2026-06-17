#!/usr/bin/env bash
# Serve a HuggingFace model with vLLM over an OpenAI-compatible API, so the
# sandbox can reach it via a `openai_compatible` model-key in models.yaml.
#
# Crucially: --served-model-name is set to the HF id so the OpenAI `model`
# field the sandbox sends (the `model:` value in models.yaml) matches what
# vLLM expects.
#
# Usage:
#   scripts/serve_vllm.sh <hf-model> [port]
#   scripts/serve_vllm.sh Qwen/Qwen3.5-4B  8000
#   scripts/serve_vllm.sh Qwen/Qwen3.5-9B  8001
#   scripts/serve_vllm.sh Qwen/Qwen3.5-27B 8002
#
# Env overrides:
#   VLLM_API_KEY   (default: EMPTY)         must match api_key_env in models.yaml
#   MAX_MODEL_LEN  (default: 32768)
#   GPU_MEM_UTIL   (default: 0.90)
#   TP_SIZE        (default: 1)             tensor-parallel size for big models
set -euo pipefail

MODEL="${1:?usage: serve_vllm.sh <hf-model> [port]}"
PORT="${2:-8000}"

exec vllm serve "$MODEL" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --api-key "${VLLM_API_KEY:-EMPTY}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --dtype auto

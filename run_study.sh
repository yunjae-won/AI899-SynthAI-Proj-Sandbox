#!/usr/bin/env bash
# Sweep the persona-measurement study across model-keys, then aggregate into a
# single model × axis table.
#
#   ./run_study.sh                      # default pilot: local Qwen ladder
#   ./run_study.sh gpt-mini claude-sonnet qwen3.5-9b
#   SEEDS=5 MAX_TURNS=12 ./run_study.sh qwen3.5-4b qwen3.5-9b qwen3.5-27b
#
# For Qwen keys, serve each model FIRST (separate env) on the port in models.yaml:
#   scripts/serve_vllm.sh Qwen/Qwen3.5-4B  8000
#   scripts/serve_vllm.sh Qwen/Qwen3.5-9B  8001
#   scripts/serve_vllm.sh Qwen/Qwen3.5-27B 8002
# For hosted keys, export OPENAI_API_KEY / ANTHROPIC_API_KEY.
set -euo pipefail
cd "$(dirname "$0")"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(qwen3.5-4b qwen3.5-9b)   # cheap local pilot
fi

SEEDS="${SEEDS:-3}"
MAX_TURNS="${MAX_TURNS:-12}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # e.g. EXTRA_ARGS="--judge --swap-trials 2"

for mk in "${MODELS[@]}"; do
  echo "================ study: ${mk} ================"
  python -m sandbox.run study --model-key "${mk}" \
    --seeds "${SEEDS}" --max-turns "${MAX_TURNS}" ${EXTRA_ARGS}
done

echo "================ aggregate ================"
python -m sandbox.run aggregate

#!/usr/bin/env bash
# Drive the persona-measurement study across Qwen3.5 targets:
# free GPU -> serve (in its own process group) -> wait ready -> run study
# (seeds=3, turns=12, conc=16) -> kill the WHOLE server group -> free GPU ->
# next model -> aggregate. One master log for monitoring.
#
# Usage: run_all_qwen.sh [model-key ...]   (default: 4b 9b 35b-a3b-int4)
set -u
cd /home/t-hyunlee/AI899-SynthAI-Proj-Sandbox
PY=python3.11
LOG=run_all_qwen.master.log
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

declare -A MODEL PORT UTIL
MODEL[qwen3.5-4b]="Qwen/Qwen3.5-4B";                  PORT[qwen3.5-4b]=8000;            UTIL[qwen3.5-4b]=0.85
MODEL[qwen3.5-9b]="Qwen/Qwen3.5-9B";                  PORT[qwen3.5-9b]=8001;            UTIL[qwen3.5-9b]=0.85
MODEL[qwen3.5-35b-a3b-int4]="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"; PORT[qwen3.5-35b-a3b-int4]=8003; UTIL[qwen3.5-35b-a3b-int4]=0.90

# Hard GPU free: kill every CUDA compute app until <3 GiB is used. (We are the
# only GPU tenant now, so this is safe and bulletproof against orphaned engines.)
free_gpu(){
  local u
  for i in $(seq 1 60); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "${u:-9999}" -lt 3000 ] && return 0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
    sleep 2
  done
}

serve_and_run(){
  local key=$1 model=${MODEL[$1]} port=${PORT[$1]} util=${UTIL[$1]}
  local slog="serve_${key}.log"; rm -f "$slog"
  say "=== ${key}: freeing GPU then serving ${model} on :${port} (util=${util}) ==="
  free_gpu
  # setsid -> server is its own process-group leader (pgid == spid), so we can
  # later kill the ENTIRE group (api server + EngineCore subprocess).
  setsid $PY -m vllm.entrypoints.openai.api_server \
      --model "$model" --served-model-name "$model" \
      --port "$port" --host 127.0.0.1 \
      --max-model-len 16384 --gpu-memory-utilization "$util" \
      --tensor-parallel-size 1 > "$slog" 2>&1 &
  local spid=$! ready=0
  for i in $(seq 1 480); do   # up to 40 min (35B-Int4 ~20GB download)
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then ready=1; say "${key}: READY after ~$((i*5))s"; break; fi
    if grep -qE "Engine core initialization failed|ValueError: Free memory|CUDA out of memory|OutOfMemoryError|No module named|Cannot find any of" "$slog" 2>/dev/null; then
      say "${key}: FATAL serve error"; grep -iE "error|free memory|out of memory|cannot find" "$slog" | tail -8 | tee -a "$LOG"; break; fi
    if ! kill -0 "$spid" 2>/dev/null; then say "${key}: server died early"; tail -8 "$slog" | tee -a "$LOG"; break; fi
    sleep 5
  done
  if [ "$ready" -eq 1 ]; then
    say "=== ${key}: RUNNING STUDY (seeds=3, turns=12, conc=16) ==="
    $PY -m sandbox.run study --model-key "$key" --seeds 3 --max-turns 12 --concurrency 16 --use-recommended 2>&1 | tee -a "$LOG"
    say "=== ${key}: STUDY DONE ==="
  else
    say "=== ${key}: SKIPPED (server not ready) ==="
  fi
  kill -9 -- -"$spid" 2>/dev/null   # kill the whole process group
  free_gpu
  say "${key}: server group stopped; gpu used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')MiB"
}

KEYS=("$@"); [ ${#KEYS[@]} -eq 0 ] && KEYS=(qwen3.5-4b qwen3.5-9b qwen3.5-35b-a3b-int4)
T0=$(date +%s)
say "models to run: ${KEYS[*]}"
for k in "${KEYS[@]}"; do serve_and_run "$k"; done

say "=== AGGREGATE (model x axis, all study runs on disk) ==="
$PY -m sandbox.run aggregate 2>&1 | tee -a "$LOG"
say "=== ALL DONE in $(( ($(date +%s)-T0)/60 )) min ==="

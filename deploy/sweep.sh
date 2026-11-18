#!/usr/bin/env bash
# Sweep vLLM tuning knobs for the LLM service and benchmark each setting. Run on the GPU box from the repo root.
#   deploy/sweep.sh data/audio/<call>.mp3 [max_seconds]
# Each line of SWEEPS = extra env overrides for one configuration. Results: runs/sweep-<label>/bench.csv
set -euo pipefail

AUDIO=${1:?usage: deploy/sweep.sh <audio> [max_seconds]}
MAX_SECONDS=${2:-600}
COMPOSE="docker compose -f deploy/docker-compose.rocm.yml --env-file deploy/.env"
BASE_EXTRA="--no-enable-prefix-caching"

# label|VAR|VALUE (one override per configuration; everything else from deploy/.env)
SWEEPS=(
  "baseline||"
  "seqs32|LLM_MAX_NUM_SEQS|32"
  "seqs128|LLM_MAX_NUM_SEQS|128"
  "batched4k|LLM_MAX_NUM_BATCHED_TOKENS|4096"
  "batched16k|LLM_MAX_NUM_BATCHED_TOKENS|16384"
  "mem0.30|LLM_GPU_MEM|0.30"
  "no-chunked-prefill|LLM_EXTRA_ARGS|$BASE_EXTRA --no-enable-chunked-prefill"
  "tunableop|PYTORCH_TUNABLEOP_ENABLED|1"
  "fp8|LLM_EXTRA_ARGS|$BASE_EXTRA --quantization fp8 --kv-cache-dtype fp8"
)

wait_healthy() {
  for _ in $(seq 1 120); do
    curl -sf "http://localhost:8000/v1/models" >/dev/null && return 0
    sleep 5
  done
  echo "LLM did not become healthy" >&2; return 1
}

for entry in "${SWEEPS[@]}"; do
  IFS='|' read -r label var value <<<"$entry"
  envs=("LLM_EXTRA_ARGS=$BASE_EXTRA")
  [[ -n $var ]] && envs+=("$var=$value")   # later assignment wins
  echo "=== $label: ${envs[*]} ==="
  env "${envs[@]}" $COMPOSE up -d --force-recreate llm
  wait_healthy
  precision=fp16; [[ $label == fp8 ]] && precision=fp8
  env "${envs[@]}" VLLM_FLAGS="${envs[*]}" LLM_PRECISION="$precision" \
    python eval/bench.py --audio "$AUDIO" --concurrency 1 5 20 --max-seconds "$MAX_SECONDS" --run-id "sweep-$label"
done

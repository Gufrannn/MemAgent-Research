#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/data/cw/memagent_work}
PROJ_ROOT=${PROJ_ROOT:-$WORK_ROOT/code/MemAgent}
DATA=${HALUMEM_DATA:-$WORK_ROOT/datasets/HaluMem/HaluMem-Medium.jsonl}
BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8001}
MODEL=${VLLM_SERVED_MODEL:-qwen25-7b}
RUN_NAME=${RUN_NAME:-riskmem_halumem_n50}

source "$WORK_ROOT/.venv/bin/activate"
cd "$PROJ_ROOT"
mkdir -p "$WORK_ROOT/logs/riskmem"

python experiments/riskmem/riskmem_headroom_vllm.py \
  --data "$DATA" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --max-users "${MAX_USERS:-1}" \
  --max-questions "${MAX_QUESTIONS:-50}" \
  --retrieve-k "${RETRIEVE_K:-8}" \
  --gate-candidates "${GATE_CANDIDATES:-20}" \
  --concurrency "${CONCURRENCY:-16}" \
  --resume \
  --output "$WORK_ROOT/logs/riskmem/$RUN_NAME.jsonl" \
  2>&1 | tee "$WORK_ROOT/logs/riskmem/$RUN_NAME.log"

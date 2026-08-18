#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/data/cw/memagent_work}
PROJ_ROOT=${PROJ_ROOT:-$WORK_ROOT/code/MemAgent-Research}
DATA_ROOT=${MEM2ACT_ROOT:-$WORK_ROOT/datasets/Mem2ActBench}
BASE_URL=${BASE_URL:-http://127.0.0.1:8001}
SERVED_MODEL=${SERVED_MODEL:-qwen25-7b}
NUM_SAMPLES=${NUM_SAMPLES:-50}
CONCURRENCY=${CONCURRENCY:-8}
RUN_NAME=${RUN_NAME:-optionmem_v3_ledger_n${NUM_SAMPLES}}
CACHE_PATH=${CACHE_PATH:-$WORK_ROOT/logs/optionmem/mem2act_v3_memory_cache.json}
CONDITIONS=${CONDITIONS:-no_memory,full_history,summary,ledger_all,ledger_retrieval,ledger_recency,ledger_oracle}

source "$WORK_ROOT/.venv/bin/activate"
cd "$PROJ_ROOT"

if [[ ! -f "$DATA_ROOT/toolmembench_small/qa_dataset.jsonl" ]]; then
  echo "Mem2ActBench not found at $DATA_ROOT" >&2
  exit 2
fi

timeout 8 curl --fail --silent --show-error "$BASE_URL/v1/models" >/dev/null
mkdir -p "$WORK_ROOT/logs/optionmem"

python experiments/optionmem/mem2act_headroom_vllm.py \
  --data-dir "$DATA_ROOT/toolmembench_small" \
  --base-url "$BASE_URL" \
  --served-model "$SERVED_MODEL" \
  --num-samples "$NUM_SAMPLES" \
  --concurrency "$CONCURRENCY" \
  --conditions "$CONDITIONS" \
  --memory-max-tokens 512 \
  --ledger-max-tokens 2048 \
  --max-ledger-items 12 \
  --cache "$CACHE_PATH" \
  --output "$WORK_ROOT/logs/optionmem/${RUN_NAME}.jsonl" \
  2>&1 | tee "$WORK_ROOT/logs/optionmem/${RUN_NAME}.log"

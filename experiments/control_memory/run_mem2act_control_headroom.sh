#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/data/cw/memagent_work}
REPO_ROOT=${REPO_ROOT:-$WORK_ROOT/code/MemAgent}
BENCH_ROOT=${BENCH_ROOT:-$WORK_ROOT/datasets/Mem2ActBench}
BASE_URL=${BASE_URL:-http://127.0.0.1:8001}
SERVED_MODEL=${SERVED_MODEL:-qwen25-7b}
NUM_SAMPLES=${NUM_SAMPLES:-50}
CONCURRENCY=${CONCURRENCY:-8}
MEMORY_TOKENS=${MEMORY_TOKENS:-512}

source "$WORK_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

QA_FILE="$BENCH_ROOT/toolmembench_small/qa_dataset.jsonl"
CONVERSATION_FILE="$BENCH_ROOT/toolmembench_small/toolmem_conversation.jsonl"
RUN_DIR="$WORK_ROOT/logs/control_memory/mem2act_headroom"
mkdir -p "$RUN_DIR/cache"

test -f "$QA_FILE"
test -f "$CONVERSATION_FILE"
curl -fsS "$BASE_URL/v1/models" >/dev/null

python experiments/control_memory/mem2act_control_headroom_vllm.py \
  --qa-jsonl "$QA_FILE" \
  --conversation-jsonl "$CONVERSATION_FILE" \
  --base-url "$BASE_URL" \
  --served-model "$SERVED_MODEL" \
  --conditions no_memory,full_history,summary,state,control \
  --num-samples "$NUM_SAMPLES" \
  --memory-tokens "$MEMORY_TOKENS" \
  --concurrency "$CONCURRENCY" \
  --cache-dir "$RUN_DIR/cache" \
  --output "$RUN_DIR/n${NUM_SAMPLES}_m${MEMORY_TOKENS}.jsonl" \
  2>&1 | tee "$RUN_DIR/n${NUM_SAMPLES}_m${MEMORY_TOKENS}.log"

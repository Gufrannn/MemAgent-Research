#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/data/cw/memagent_work}
REPO_ROOT=${REPO_ROOT:-$WORK_ROOT/code/MemAgent-control}
BENCH_ROOT=${BENCH_ROOT:-$WORK_ROOT/datasets/Mem2ActBench}
BASE_URL=${BASE_URL:-http://127.0.0.1:8001}
SERVED_MODEL=${SERVED_MODEL:-qwen25-7b}
NUM_SAMPLES=${NUM_SAMPLES:-50}
LEVELS=${LEVELS:-L1,L2,L3,L4}
CONCURRENCY=${CONCURRENCY:-8}
LEDGER_BASE_TOKENS=${LEDGER_BASE_TOKENS:-384}
DYNAMICS_TOKENS=${DYNAMICS_TOKENS:-128}
TOTAL_TOKENS=${TOTAL_TOKENS:-512}

source "$WORK_ROOT/.venv/bin/activate"
cd "$REPO_ROOT"

RUN_DIR="$WORK_ROOT/logs/control_memory/mem2act_nested"
mkdir -p "$RUN_DIR/cache"
curl -fsS "$BASE_URL/v1/models" >/dev/null

RUN_TAG="n${NUM_SAMPLES}_levels-${LEVELS//,/-}_l${LEDGER_BASE_TOKENS}_d${DYNAMICS_TOKENS}"

python experiments/control_memory/mem2act_control_headroom_vllm.py \
  --qa-jsonl "$BENCH_ROOT/toolmembench_small/qa_dataset.jsonl" \
  --conversation-jsonl "$BENCH_ROOT/toolmembench_small/toolmem_conversation.jsonl" \
  --base-url "$BASE_URL" \
  --served-model "$SERVED_MODEL" \
  --conditions full_history,summary,ledger384,ledger512,control_nested \
  --levels "$LEVELS" \
  --num-samples "$NUM_SAMPLES" \
  --memory-tokens "$TOTAL_TOKENS" \
  --ledger-base-tokens "$LEDGER_BASE_TOKENS" \
  --dynamics-tokens "$DYNAMICS_TOKENS" \
  --concurrency "$CONCURRENCY" \
  --cache-dir "$RUN_DIR/cache" \
  --output "$RUN_DIR/${RUN_TAG}.jsonl" \
  2>&1 | tee "$RUN_DIR/${RUN_TAG}.log"

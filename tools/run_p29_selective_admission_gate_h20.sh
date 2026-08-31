#!/usr/bin/env bash
set -euo pipefail

# P29 H20 read-only learned static admission gate.
#
# This runner trains no generator and performs no inference.  It consumes the
# frozen P27 five-policy dev80 matrix plus the STOP response/trace only to test
# whether legal online state features can select KEEP vs READMIT policies.
#
# Not allowed here:
#   - changing prompts/operators;
#   - regenerating P27 outputs;
#   - using question_type/gold/answer/judge labels as online features;
#   - opening confirm324.

PROJECT_DIR="${PROJECT_DIR:-/data/cw/memagent_work/UMA-BudgetedEvidenceMemory-20260830}"
P27_RUN_ROOT="${P27_RUN_ROOT:-/data/cw/memagent_work/code/UMA-P27-AdmissionBaselines-H20/runs/longmemeval_p27_admission_policy_baselines_Qwen2.5-7B-Instruct_dev80_k20}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/runs/p29_selective_admission_gate_from_p27_dev80_k20}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

MATRIX="${P27_RUN_ROOT}/matrix/longmemeval_operation_value_wide.csv"
STOP_RESPONSES="${P27_RUN_ROOT}/stop/responses_stop.jsonl"
STOP_TRACE="${P27_RUN_ROOT}/stop/trace.jsonl"

if [[ ! -f "${MATRIX}" ]]; then
  echo "Missing P27 wide matrix: ${MATRIX}" >&2
  exit 2
fi
if [[ ! -f "${STOP_RESPONSES}" ]]; then
  echo "Missing STOP responses: ${STOP_RESPONSES}" >&2
  exit 2
fi
if [[ ! -f "${STOP_TRACE}" ]]; then
  echo "Missing STOP trace: ${STOP_TRACE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

POLICIES=(
  "repack_lexical_bm25"
  "repack_tfidf_jaccard"
  "repack_graph_bridge"
  "repack_temporal_session"
)

run_metric() {
  local metric="$1"
  local out_dir="${OUTPUT_ROOT}/${metric}"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" tools/train_p29_selective_admission_gate.py \
    --wide-matrix "${MATRIX}" \
    --stop-responses "${STOP_RESPONSES}" \
    --stop-trace "${STOP_TRACE}" \
    --metric "${metric}" \
    --policy repack_lexical_bm25 \
    --policy repack_tfidf_jaccard \
    --policy repack_graph_bridge \
    --policy repack_temporal_session \
    --readmit-policy repack_tfidf_jaccard \
    --readmit-policy repack_graph_bridge \
    --readmit-policy repack_lexical_bm25 \
    --readmit-policy repack_temporal_session \
    --feature-set stats_text \
    --hash-dim 256 \
    --alphas 0.01,0.1,1,10,100,1000 \
    --thresholds=-0.025,0,0.025,0.05,0.1 \
    --inner-folds 5 \
    --eps 0.1 \
    --tie-eps 0.01 \
    --bootstrap-samples 2000 \
    --seed 20260831 \
    --output-dir "${out_dir}"
}

run_metric reward
run_metric proxy_utility_context

sha256sum \
  "${OUTPUT_ROOT}/reward/p29_selector_report_reward.json" \
  "${OUTPUT_ROOT}/reward/p29_selector_summary_reward.csv" \
  "${OUTPUT_ROOT}/reward/p29_selector_per_qid_reward.csv" \
  "${OUTPUT_ROOT}/proxy_utility_context/p29_selector_report_proxy_utility_context.json" \
  "${OUTPUT_ROOT}/proxy_utility_context/p29_selector_summary_proxy_utility_context.csv" \
  "${OUTPUT_ROOT}/proxy_utility_context/p29_selector_per_qid_proxy_utility_context.csv"

echo "P29 selective admission gate completed: ${OUTPUT_ROOT}"

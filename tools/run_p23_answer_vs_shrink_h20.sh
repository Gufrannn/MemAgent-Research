#!/usr/bin/env bash
set -euo pipefail

# P23 H20 ANSWER-vs-SHRINK scale check.
#
# Purpose:
#   Run only two frozen paths on the preregistered dev80 manifest:
#     stop   = retrieve -> answer
#     refine = retrieve -> shrink/refine -> answer
#
# This is inference-only.  It must not tune prompts, operators, retrieval
# protocol, or RL.  Gold evidence labels are consumed only after generation for
# offline mechanism audit.

PROJECT_DIR="${PROJECT_DIR:-/data/cw/memagent_work/UMA-BudgetedEvidenceMemory-20260830}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the served model name, e.g. Qwen2.5-7B-Instruct}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/runs/longmemeval_manifests_p21_h2/longmemeval_p21_h2_dev80_manifest.json}"
RAW_LONGMEMEVAL="${RAW_LONGMEMEVAL:-${PROJECT_DIR}/data/raw/longmemeval_s.json}"
AMC_TOP_K="${AMC_TOP_K:-20}"
AMC_EXPAND_K="${AMC_EXPAND_K:-20}"
AMC_MAX_PROMPT_CHARS="${AMC_MAX_PROMPT_CHARS:-32768}"
AMC_MAX_TOKENS="${AMC_MAX_TOKENS:-2048}"
CONCURRENCY="${CONCURRENCY:-1}"
TAG_SAFE_MODEL="$(printf '%s' "${MODEL_NAME}" | tr '/: ' '___')"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/longmemeval_p23_answer_vs_shrink_${TAG_SAFE_MODEL}_dev80_k${AMC_TOP_K}}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing P23 dev80 manifest: ${MANIFEST}" >&2
  exit 2
fi
if [[ ! -f "${RAW_LONGMEMEVAL}" ]]; then
  echo "Missing raw LongMemEval-S file: ${RAW_LONGMEMEVAL}" >&2
  exit 2
fi

run_memory_sequence() {
  local agent_id="$1"
  local sequence_name="$2"
  local output_dir="${RUN_ROOT}/${agent_id}"
  mkdir -p "${output_dir}"
  rm -f "${output_dir}/responses_${agent_id}.jsonl" "${output_dir}/trace.jsonl"

  export AMC_SEQUENCE="${sequence_name}"
  export AMC_TRACE_PATH="${output_dir}/trace.jsonl"
  export AMC_TRACE_STATE_TEXT="${AMC_TRACE_STATE_TEXT:-1}"
  export AMC_TOP_K
  export AMC_EXPAND_K
  export AMC_EXPAND_MODE="${AMC_EXPAND_MODE:-bm25_tail}"
  export AMC_FILTER_MODE="${AMC_FILTER_MODE:-graph_bridge}"
  export AMC_GRAPH_MAX_SENTENCES="${AMC_GRAPH_MAX_SENTENCES:-320}"
  export AMC_MAX_PROMPT_CHARS
  export AMC_MAX_TOKENS
  export UMA_TEMPERATURE="${UMA_TEMPERATURE:-0}"
  export UMA_TOP_P="${UMA_TOP_P:-1}"

  "${PYTHON_BIN}" tools/run_uma_generation_light.py \
    --task longmemeval \
    --agent memory_sequence \
    --agent-id "${agent_id}" \
    --model "${MODEL_NAME}" \
    --output-dir "${output_dir}" \
    --concurrency "${CONCURRENCY}" \
    --qid-manifest "${MANIFEST}" \
    --preserve-manifest-order \
    --force-overwrite

  "${PYTHON_BIN}" tools/convert_uma_responses_to_longmemeval_hypothesis.py \
    --responses "${output_dir}/responses_${agent_id}.jsonl" \
    --output "${output_dir}/hypothesis_${agent_id}_official_longmemeval.jsonl"
}

run_memory_sequence stop stop
run_memory_sequence refine refine

"${PYTHON_BIN}" tools/build_longmemeval_operation_value_matrix.py \
  --raw-longmemeval "${RAW_LONGMEMEVAL}" \
  --manifest "${MANIFEST}" \
  --baseline-operation stop \
  --operation "stop=${RUN_ROOT}/stop/responses_stop.jsonl:${RUN_ROOT}/stop/trace.jsonl" \
  --operation "refine=${RUN_ROOT}/refine/responses_refine.jsonl:${RUN_ROOT}/refine/trace.jsonl" \
  --reward-source surrogate_f1 \
  --cost-field context_kchars \
  --lambda-cost 0.02 \
  --output-dir "${RUN_ROOT}/matrix"

"${PYTHON_BIN}" tools/analyze_p23_answer_vs_shrink.py \
  --wide-matrix "${RUN_ROOT}/matrix/longmemeval_operation_value_wide.csv" \
  --raw-longmemeval "${RAW_LONGMEMEVAL}" \
  --manifest "${MANIFEST}" \
  --model-label "${MODEL_NAME}_dev80_k${AMC_TOP_K}" \
  --baseline-operation stop \
  --shrink-operation refine \
  --stop-response "${RUN_ROOT}/stop/responses_stop.jsonl" \
  --stop-trace "${RUN_ROOT}/stop/trace.jsonl" \
  --output-dir "${RUN_ROOT}/p23_answer_vs_shrink_audit"

echo "P23 answer-vs-shrink completed: ${RUN_ROOT}"

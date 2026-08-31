#!/usr/bin/env bash
set -euo pipefail

# P26 H20 Admission-vs-Transformation Factorization.
#
# Purpose:
#   Split the ambiguous legacy REFINE behavior into two explicit contracts:
#
#     stop              = retrieve C0 -> admit W0 -> answer
#     shrink_visible    = retrieve C0 -> admit W0 -> transform only W0 -> answer
#     repack_candidates = retrieve C0 -> reselect/repack from fixed C0 -> answer
#
# This is inference-only.  It must not tune prompts, operators, retrieval
# protocol, judge, split, model parameters, or RL.  Gold evidence labels are
# consumed only after generation for offline decomposition.

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
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/longmemeval_p26_admission_transformation_${TAG_SAFE_MODEL}_dev80_k${AMC_TOP_K}}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing P26 dev80 manifest: ${MANIFEST}" >&2
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
run_memory_sequence shrink_visible shrink_visible
run_memory_sequence repack_candidates repack_candidates

"${PYTHON_BIN}" tools/build_longmemeval_operation_value_matrix.py \
  --raw-longmemeval "${RAW_LONGMEMEVAL}" \
  --manifest "${MANIFEST}" \
  --baseline-operation stop \
  --operation "stop=${RUN_ROOT}/stop/responses_stop.jsonl:${RUN_ROOT}/stop/trace.jsonl" \
  --operation "shrink_visible=${RUN_ROOT}/shrink_visible/responses_shrink_visible.jsonl:${RUN_ROOT}/shrink_visible/trace.jsonl" \
  --operation "repack_candidates=${RUN_ROOT}/repack_candidates/responses_repack_candidates.jsonl:${RUN_ROOT}/repack_candidates/trace.jsonl" \
  --reward-source surrogate_f1 \
  --cost-field context_kchars \
  --lambda-cost 0.02 \
  --output-dir "${RUN_ROOT}/matrix"

"${PYTHON_BIN}" tools/analyze_p26_admission_transformation_factorization.py \
  --wide-matrix "${RUN_ROOT}/matrix/longmemeval_operation_value_wide.csv" \
  --baseline-operation stop \
  --operation shrink_visible \
  --operation repack_candidates \
  --eps 0.1 \
  --output-dir "${RUN_ROOT}/p26_admission_transformation_audit"

"${PYTHON_BIN}" tools/export_p25_5_compact_trace_parity.py \
  --operation "stop=${RUN_ROOT}/stop/responses_stop.jsonl:${RUN_ROOT}/stop/trace.jsonl" \
  --operation "shrink_visible=${RUN_ROOT}/shrink_visible/responses_shrink_visible.jsonl:${RUN_ROOT}/shrink_visible/trace.jsonl" \
  --operation "repack_candidates=${RUN_ROOT}/repack_candidates/responses_repack_candidates.jsonl:${RUN_ROOT}/repack_candidates/trace.jsonl" \
  --output "${RUN_ROOT}/p26_compact_trace_parity.csv"

sha256sum "${RUN_ROOT}/matrix/longmemeval_operation_value_wide.csv" \
  "${RUN_ROOT}/p26_compact_trace_parity.csv" \
  "${RUN_ROOT}/p26_admission_transformation_audit/p26_admission_transformation_report.json"

echo "P26 admission-vs-transformation factorization completed: ${RUN_ROOT}"

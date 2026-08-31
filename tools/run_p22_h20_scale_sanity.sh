#!/usr/bin/env bash
set -euo pipefail

# P22 H20 Scale Sanity Experiment
#
# Inference-only structural replication. No RL training, no prompt tuning, no
# operator retuning. The only intended knobs are MODEL_NAME and AMC_TOP_K.
#
# Required:
#   * An OpenAI-compatible endpoint already serving MODEL_NAME at
#     http://127.0.0.1:8000/v1, or edit config.py/API_CONFIG_LOCAL beforehand.
#   * LongMemEval raw files already placed under ${PROJECT_DIR}/data/raw.
#   * The fixed sequence36 manifest already placed at ${MANIFEST}.

PROJECT_DIR="${PROJECT_DIR:-/data/cw/memagent_work/UMA-BudgetedEvidenceMemory-20260830}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the served model name, e.g. Qwen2.5-7B-Instruct}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/runs/longmemeval_manifests_p14/longmemeval_sequence36_manifest.json}"
RAW_LONGMEMEVAL="${RAW_LONGMEMEVAL:-${PROJECT_DIR}/data/raw/longmemeval_s.json}"
RAW_LONGMEMEVAL_ORACLE="${RAW_LONGMEMEVAL_ORACLE:-${PROJECT_DIR}/data/raw/longmemeval_oracle.json}"
AMC_TOP_K="${AMC_TOP_K:-20}"
AMC_EXPAND_K="${AMC_EXPAND_K:-20}"
AMC_MAX_PROMPT_CHARS="${AMC_MAX_PROMPT_CHARS:-32768}"
AMC_MAX_TOKENS="${AMC_MAX_TOKENS:-2048}"
RUN_ORACLE="${RUN_ORACLE:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
TAG_SAFE_MODEL="$(printf '%s' "${MODEL_NAME}" | tr '/: ' '___')"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/longmemeval_p22_sequence36_${TAG_SAFE_MODEL}_k${AMC_TOP_K}}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_ROOT}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 2
fi
if [[ ! -f "${RAW_LONGMEMEVAL}" ]]; then
  echo "Missing raw LongMemEval-S file: ${RAW_LONGMEMEVAL}" >&2
  exit 2
fi

run_memory_sequence() {
  local agent_id="$1"
  local sequence_name="$2"
  local expand_mode="$3"
  local output_dir="${RUN_ROOT}/${agent_id}"
  mkdir -p "${output_dir}"
  rm -f "${output_dir}/responses_${agent_id}.jsonl" "${output_dir}/trace.jsonl"

  export AMC_SEQUENCE="${sequence_name}"
  export AMC_TRACE_PATH="${output_dir}/trace.jsonl"
  export AMC_TRACE_STATE_TEXT="${AMC_TRACE_STATE_TEXT:-1}"
  export AMC_TOP_K
  export AMC_EXPAND_K
  export AMC_EXPAND_MODE="${expand_mode}"
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

run_concat_oracle() {
  if [[ "${RUN_ORACLE}" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "${RAW_LONGMEMEVAL_ORACLE}" ]]; then
    echo "Skipping oracle diagnostic because missing: ${RAW_LONGMEMEVAL_ORACLE}" >&2
    return 0
  fi
  local agent_id="oracle_d1"
  local output_dir="${RUN_ROOT}/${agent_id}"
  mkdir -p "${output_dir}"
  rm -f "${output_dir}/responses_${agent_id}.jsonl" "${output_dir}/trace.jsonl"
  export UMA_TEMPERATURE="${UMA_TEMPERATURE:-0}"
  export UMA_TOP_P="${UMA_TOP_P:-1}"
  export AMC_TRACE_PATH="${output_dir}/trace.jsonl"
  "${PYTHON_BIN}" tools/run_uma_generation_light.py \
    --task longmemeval_oracle \
    --agent concat_single \
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

run_memory_sequence stop stop bm25_tail
run_memory_sequence refine refine bm25_tail
run_memory_sequence expand_q expand bm25_tail
run_memory_sequence refine_expand refine_expand bm25_tail
run_memory_sequence expand_refine expand_refine bm25_tail
run_concat_oracle

operation_args=(
  --operation "stop=${RUN_ROOT}/stop/responses_stop.jsonl:${RUN_ROOT}/stop/trace.jsonl"
  --operation "refine=${RUN_ROOT}/refine/responses_refine.jsonl:${RUN_ROOT}/refine/trace.jsonl"
  --operation "expand_q=${RUN_ROOT}/expand_q/responses_expand_q.jsonl:${RUN_ROOT}/expand_q/trace.jsonl"
  --operation "refine_expand=${RUN_ROOT}/refine_expand/responses_refine_expand.jsonl:${RUN_ROOT}/refine_expand/trace.jsonl"
  --operation "expand_refine=${RUN_ROOT}/expand_refine/responses_expand_refine.jsonl:${RUN_ROOT}/expand_refine/trace.jsonl"
)
if [[ "${RUN_ORACLE}" == "1" && -f "${RUN_ROOT}/oracle_d1/responses_oracle_d1.jsonl" && -f "${RAW_LONGMEMEVAL_ORACLE}" ]]; then
  operation_args+=(--operation "oracle_d1=${RUN_ROOT}/oracle_d1/responses_oracle_d1.jsonl:${RUN_ROOT}/oracle_d1/trace.jsonl:${RAW_LONGMEMEVAL_ORACLE}")
fi

"${PYTHON_BIN}" tools/build_longmemeval_operation_value_matrix.py \
  --raw-longmemeval "${RAW_LONGMEMEVAL}" \
  --manifest "${MANIFEST}" \
  --baseline-operation stop \
  "${operation_args[@]}" \
  --reward-source surrogate_f1 \
  --cost-field context_kchars \
  --lambda-cost 0.02 \
  --output-dir "${RUN_ROOT}/matrix"

"${PYTHON_BIN}" tools/analyze_p14_sequence_mdp_evidence.py \
  --wide-matrix "${RUN_ROOT}/matrix/longmemeval_operation_value_wide.csv" \
  --output-dir "${RUN_ROOT}/mdp_evidence_analysis" \
  --expand-operation expand_q \
  --metric reward \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260831

"${PYTHON_BIN}" tools/analyze_p14_sequence_mdp_evidence.py \
  --wide-matrix "${RUN_ROOT}/matrix/longmemeval_operation_value_wide.csv" \
  --output-dir "${RUN_ROOT}/mdp_evidence_analysis" \
  --expand-operation expand_q \
  --metric utility \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260831

"${PYTHON_BIN}" tools/audit_within_type_operation_heterogeneity.py \
  --wide-matrix "${RUN_ROOT}/matrix/longmemeval_operation_value_wide.csv" \
  --output-dir "${RUN_ROOT}/heterogeneity_reward_legal" \
  --baseline-operation stop \
  --operations refine expand_q refine_expand expand_refine \
  --operation-access legal_online_operation \
  --metric reward \
  --thresholds 0.05 0.1

"${PYTHON_BIN}" tools/audit_within_type_operation_heterogeneity.py \
  --wide-matrix "${RUN_ROOT}/matrix/longmemeval_operation_value_wide.csv" \
  --output-dir "${RUN_ROOT}/heterogeneity_utility_legal" \
  --baseline-operation stop \
  --operations refine expand_q refine_expand expand_refine \
  --operation-access legal_online_operation \
  --metric utility \
  --thresholds 0.05 0.1

if [[ -f "${RUN_ROOT}/oracle_d1/responses_oracle_d1.jsonl" ]]; then
  "${PYTHON_BIN}" tools/audit_within_type_operation_heterogeneity.py \
    --wide-matrix "${RUN_ROOT}/matrix/longmemeval_operation_value_wide.csv" \
    --output-dir "${RUN_ROOT}/heterogeneity_reward_oracle_diagnostic" \
    --baseline-operation stop \
    --operations oracle_d1 \
    --operation-access privileged_upper_bound \
    --metric reward \
    --thresholds 0.05 0.1
fi

echo "P22 H20 scale sanity completed: ${RUN_ROOT}"

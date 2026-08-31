#!/usr/bin/env bash
set -euo pipefail

# P30 Semantic Selective Admission over frozen P27/P29.5 artifacts.
#
# This runner performs no generation and no reader inference.  It consumes a
# frozen P27 five-policy matrix plus STOP response/trace, verifies the W0
# state_text contract, reconstructs C0/W0 texts from raw LongMemEval sessions,
# and trains the SSA v1 Default + Override selector with a frozen semantic
# encoder.
#
# Not allowed here:
#   - changing prompt/operator/split/metric/generation protocol;
#   - opening confirm324;
#   - using question_type/gold/answer/judge/outcome as online features;
#   - using raw source-index statistics as features;
#   - oversampling exceptions for final realized policy evaluation;
#   - silently falling back to hashed/TF-IDF features when semantic encoder is unavailable.

PROJECT_DIR="${PROJECT_DIR:-/data/cw/memagent_work/UMA-BudgetedEvidenceMemory-20260830}"
P27_RUN_ROOT="${P27_RUN_ROOT:?Set P27_RUN_ROOT to a frozen P27/P27.5 run root}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/runs/longmemeval_manifests_p21_h2/longmemeval_p21_h2_dev80_manifest.json}"
RAW_LONGMEMEVAL="${RAW_LONGMEMEVAL:-${PROJECT_DIR}/data/raw/longmemeval_s.json}"
ENCODER_MODEL="${ENCODER_MODEL:?Set ENCODER_MODEL to a local or already-downloadable sentence-transformers encoder}"
ENCODER_BACKEND="${ENCODER_BACKEND:-sentence_transformers}"
ENCODER_MAX_LENGTH="${ENCODER_MAX_LENGTH:-512}"
CONTEXT_LABEL="${CONTEXT_LABEL:-dev80_k20_semantic_ssa}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/runs/p30_semantic_selective_admission_from_p27_dev80_k20}"
EMBEDDING_CACHE_ROOT="${EMBEDDING_CACHE_ROOT:-${OUTPUT_ROOT}/embedding_cache}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/tools:${PYTHONPATH:-}"

MATRIX="${P27_RUN_ROOT}/matrix/longmemeval_operation_value_wide.csv"
STOP_RESPONSES="${P27_RUN_ROOT}/stop/responses_stop.jsonl"
STOP_TRACE="${P27_RUN_ROOT}/stop/trace.jsonl"

for required in "${MATRIX}" "${STOP_RESPONSES}" "${STOP_TRACE}" "${MANIFEST}" "${RAW_LONGMEMEVAL}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required frozen/input file: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${EMBEDDING_CACHE_ROOT}"

"${PYTHON_BIN}" tools/verify_trace_state_text_contract.py \
  --responses "${STOP_RESPONSES}" \
  --trace "${STOP_TRACE}" \
  --output-csv "${OUTPUT_ROOT}/p30_stop_trace_state_text_contract.csv" \
  --output-json "${OUTPUT_ROOT}/p30_stop_trace_state_text_contract.json" \
  --fail-on-violation

run_metric() {
  local metric="$1"
  local out_dir="${OUTPUT_ROOT}/${metric}"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" tools/train_p30_semantic_selective_admission.py \
    --wide-matrix "${MATRIX}" \
    --stop-responses "${STOP_RESPONSES}" \
    --stop-trace "${STOP_TRACE}" \
    --raw-longmemeval "${RAW_LONGMEMEVAL}" \
    --manifest "${MANIFEST}" \
    --metric "${metric}" \
    --policy repack_lexical_bm25 \
    --policy repack_tfidf_jaccard \
    --policy repack_graph_bridge \
    --policy repack_temporal_session \
    --encoder-model "${ENCODER_MODEL}" \
    --encoder-backend "${ENCODER_BACKEND}" \
    --embedding-cache "${EMBEDDING_CACHE_ROOT}/p30_${metric}.npz" \
    --encoder-batch-size "${ENCODER_BATCH_SIZE:-32}" \
    --encoder-max-length "${ENCODER_MAX_LENGTH}" \
    --alphas "${SSA_ALPHAS:-0.01,0.1,1,10,100,1000}" \
    --thresholds "${SSA_THRESHOLDS:-0,0.025,0.05,0.075,0.1,0.15}" \
    --inner-folds "${SSA_INNER_FOLDS:-5}" \
    --eps "${SSA_EPS:-0.1}" \
    --margin-eps "${SSA_MARGIN_EPS:-0.1}" \
    --tie-eps "${SSA_TIE_EPS:-0.01}" \
    --bootstrap-samples "${SSA_BOOTSTRAP_SAMPLES:-2000}" \
    --seed "${SSA_SEED:-20260901}" \
    --context-label "${CONTEXT_LABEL}" \
    --output-dir "${out_dir}"
}

run_metric reward
run_metric proxy_utility_context

sha256sum \
  "${OUTPUT_ROOT}/p30_stop_trace_state_text_contract.json" \
  "${OUTPUT_ROOT}/p30_stop_trace_state_text_contract.csv" \
  "${OUTPUT_ROOT}/reward/p30_ssa_report_reward.json" \
  "${OUTPUT_ROOT}/reward/p30_ssa_method_table_reward.csv" \
  "${OUTPUT_ROOT}/reward/p30_ssa_per_qid_reward.csv" \
  "${OUTPUT_ROOT}/reward/p30_ssa_semantic_feature_audit.csv" \
  "${OUTPUT_ROOT}/proxy_utility_context/p30_ssa_report_proxy_utility_context.json" \
  "${OUTPUT_ROOT}/proxy_utility_context/p30_ssa_method_table_proxy_utility_context.csv" \
  "${OUTPUT_ROOT}/proxy_utility_context/p30_ssa_per_qid_proxy_utility_context.csv" \
  "${OUTPUT_ROOT}/proxy_utility_context/p30_ssa_semantic_feature_audit.csv"

echo "P30 semantic selective admission completed: ${OUTPUT_ROOT}"

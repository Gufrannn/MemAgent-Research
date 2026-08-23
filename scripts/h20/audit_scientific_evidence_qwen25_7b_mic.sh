#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/mic_common.sh"
mic_require_checkout

readonly MIC_SCIENCE_OUTPUT=${MEMAGENT_MIC_SCIENCE_OUTPUT:-$MIC_CERT/scientific_evidence_audit.json}
readonly MIC_TRAIN_PARQUET=$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet
readonly MIC_S128_PARQUET=$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet
readonly MIC_ROLLOUT_SEEDS=$MIC_OUTPUT/rollout_seed_audit.jsonl
readonly MIC_T25_CRITIC=$MIC_CRITIC_ROOT/global_step_25/critic.json

for path in "$MIC_MANIFEST" "$MIC_P0" "$MIC_E0" "$MIC_TRAIN_PARQUET" "$MIC_S128_PARQUET" \
  "$MIC_CURVE_RESOLVED" "$MIC_LEDGER" "$MIC_ROLLOUT_SEEDS" "$MIC_T25_CRITIC" \
  "$MIC_WEIGHT_LEDGER" "$MIC_PAPER_REVIEW" \
  "$MEMAGENT_MIC_REPO_DIR/docs/papers/mic_adaptive_use_disclosure_20260823.json" \
  "$MIC_CERT/t5_audit.json" "$MIC_CERT/t10_audit.json" "$MIC_CERT/t15_audit.json" \
  "$MIC_CERT/t20_audit.json" "$MIC_CERT/t25_audit.json"; do
  [[ -f $path ]] || { echo "MIC_NO_GO: scientific evidence file absent: $path" >&2; exit 91; }
done

PYTHONPATH=$MEMAGENT_MIC_REPO_DIR "$MIC_PYTHON" \
  "$MEMAGENT_MIC_REPO_DIR/tools/h20/audit_mic_scientific_evidence.py" \
  --manifest "$MIC_MANIFEST" --p0 "$MIC_P0" --e0 "$MIC_E0" \
  --run-id "$MEMAGENT_MIC_RUN_ID" \
  --train-parquet "$MIC_TRAIN_PARQUET" --s128-parquet "$MIC_S128_PARQUET" \
  --s128-resolved "$MIC_CURVE_RESOLVED" --mic-ledger "$MIC_LEDGER" \
  --weight-ledger "$MIC_WEIGHT_LEDGER" --paper-review "$MIC_PAPER_REVIEW" \
  --rollout-seed-audit "$MIC_ROLLOUT_SEEDS" --critic-checkpoint "$MIC_T25_CRITIC" \
  --health-audits "$MIC_CERT/t5_audit.json" "$MIC_CERT/t10_audit.json" \
    "$MIC_CERT/t15_audit.json" "$MIC_CERT/t20_audit.json" "$MIC_CERT/t25_audit.json" \
  --adaptive-disclosure \
    "$MEMAGENT_MIC_REPO_DIR/docs/papers/mic_adaptive_use_disclosure_20260823.json" \
  --output "$MIC_SCIENCE_OUTPUT"

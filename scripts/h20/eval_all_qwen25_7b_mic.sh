#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/mic_common.sh"
mic_require_checkout; mic_require_training_gates
export MEMAGENT_MIC_BASELINE_INVENTORY=$MIC_BASELINE_INVENTORY
export MEMAGENT_MIC_BASELINE_AUTHORITY_SHA256
MEMAGENT_MIC_BASELINE_AUTHORITY_SHA256=$(
  "$MIC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["original_curve_report_sha256"])' \
    "$MIC_P0"
)
for step in 5 10 15 20 25; do
  bash "$SCRIPT_DIR/eval_audit_qwen25_7b_mic.sh" "$step"
done
"$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" final-eval-audit \
  --baseline "$MIC_BASELINE" --p0 "$MIC_P0" --health-root "$MIC_CERT" \
  --eval-root "$MIC_ROOT" --output-root "$MIC_OUTPUT" \
  --checkpoint-authority "$MIC_CHECKPOINT_AUTHORITY" \
  --checkpoint-authority-certificate "$MIC_CHECKPOINT_AUTHORITY_CERT" \
  --weight-ledger "$MIC_WEIGHT_LEDGER" --mic-ledger "$MIC_LEDGER" \
  --e0 "$MIC_E0" --paper-review "$MIC_PAPER_REVIEW" \
  --output "$MIC_CERT/final_eval_audit.json"

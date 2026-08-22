#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'usage: audit_health_qwen25_7b_mic.sh STEP' >&2; exit 64; }
STEP=$1
case "$STEP" in 5|10|15|20|25) ;; *) echo 'MIC_NO_GO: invalid anchor' >&2; exit 65;; esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/mic_common.sh"
mic_require_checkout; mic_require_training_gates
[[ -d $MIC_OUTPUT/global_step_${STEP}/actor && -f $MIC_OUTPUT/global_step_${STEP}/data.pt ]] || {
  echo 'MIC_NO_GO: checkpoint incomplete' >&2; exit 80;
}
"$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" audit \
  --p0 "$MIC_P0" --e0 "$MIC_E0" --paper-review "$MIC_PAPER_REVIEW" \
  --ledger "$MIC_LEDGER" --weight-ledger "$MIC_WEIGHT_LEDGER" --target-step "$STEP" \
  --output "$MIC_CERT/t${STEP}_audit.json"
if [[ $STEP -eq 5 ]]; then
  "$MIC_PYTHON" - "$MIC_CERT/t5_audit.json" "$MIC_CERT/t5_health.json" <<'PY'
import json, pathlib, sys
source, target = map(pathlib.Path, sys.argv[1:])
if target.exists():
    raise SystemExit("MIC_NO_GO: refusing to overwrite T5 health certificate")
audit = json.loads(source.read_text())
if audit.get("status") != "PASS" or audit.get("decision") != "MIC_T5_AUDIT_PASS":
    raise SystemExit("MIC_NO_GO: T5 audit is not healthy")
target.write_text(json.dumps({
    "schema": "memagent.mic.v1", "status": "PASS",
    "decision": "MIC_T5_TRAINING_HEALTH_PASS",
    "criterion": "finite_training_checkpoint_gradient_ledger_weight_sync",
    "t5_audit_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
}, indent=2, sort_keys=True) + "\n")
PY
fi

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=scripts/h20/serialization_credit_pilots_common.sh
source "$SCRIPT_DIR/serialization_credit_pilots_common.sh"

serial_credit_sanitize_inherited_environment
serial_credit_require_checkout
serial_credit_require_p0
export CUDA_VISIBLE_DEVICES=$SERIAL_CREDIT_GPUS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=0
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SERIAL_CREDIT_REPO_DIR
export TOKENIZERS_PARALLELISM=false
[[ -f $SERIAL_CREDIT_SMSB_REPORT ]] || {
  echo 'SERIAL_CREDIT_NO_GO:TETRAD SMSB adjudication is missing' >&2; exit 76;
}
"$SERIAL_CREDIT_PYTHON" \
  "$SERIAL_CREDIT_REPO_DIR/tools/h20/audit_qwen25_7b_serialization_credit.py" \
  --manifest "$SERIAL_CREDIT_MANIFEST" --smsb-gate-only >/dev/null || {
    echo 'SERIAL_CREDIT_NO_GO:TETRAD authenticated SMSB ledger/report gate failed' >&2
    exit 84
  }

serial_credit_acquire_lock
serial_credit_require_idle
[[ ! -e $SERIAL_CREDIT_TETRAD_ROOT ]] || {
  echo 'SERIAL_CREDIT_NO_GO:TETRAD append-only Tetrad root already exists' >&2; exit 77;
}
mkdir -p "$SERIAL_CREDIT_TETRAD_RESULTS" "$SERIAL_CREDIT_TETRAD_CREDENTIALS" \
  "$SERIAL_CREDIT_TETRAD_RECEIPTS" "$SERIAL_CREDIT_TETRAD_CHILD_LOGS"
readonly TETRAD_LOG=$SERIAL_CREDIT_LOG_ROOT/tetrad4.log
readonly RUNNER=$SERIAL_CREDIT_REPO_DIR/tools/h20/run_qwen25_7b_serialization_credit.py
readonly LAUNCHER=$SERIAL_CREDIT_REPO_DIR/tools/h20/launch_qwen25_7b_serialization_credit_child.py

cd "$SERIAL_CREDIT_REPO_DIR"

"$SERIAL_CREDIT_PYTHON" "$RUNNER" --manifest "$SERIAL_CREDIT_MANIFEST" \
  prepare-tetrad --captures "$SERIAL_CREDIT_SMSB_CAPTURES" \
  --smsb-report "$SERIAL_CREDIT_SMSB_REPORT" \
  --authoring-output "$SERIAL_CREDIT_TETRAD_AUTHORING" \
  --manifest-output "$SERIAL_CREDIT_TETRAD_MANIFEST" >>"$TETRAD_LOG" 2>&1
serial_credit_record --record-type tetrad_construct --artifact "$SERIAL_CREDIT_TETRAD_MANIFEST"

mapfile -t request_rows < <(
  "$SERIAL_CREDIT_PYTHON" "$RUNNER" --manifest "$SERIAL_CREDIT_MANIFEST" \
    list-tetrad-requests --tetrad-manifest "$SERIAL_CREDIT_TETRAD_MANIFEST"
)
[[ ${#request_rows[@]} -eq 20 ]] || {
  echo 'SERIAL_CREDIT_NO_GO:TETRAD manifest does not contain 20 requests' >&2; exit 82;
}

completed=0
for request_row in "${request_rows[@]}"; do
  IFS=$'\t' read -r request_id example_id state_role <<<"$request_row"
  artifact=$(printf '%s/%02d.json' "$SERIAL_CREDIT_TETRAD_RESULTS" "$completed")
  credential=$(printf '%s/%02d.json' "$SERIAL_CREDIT_TETRAD_CREDENTIALS" "$completed")
  receipt=$(printf '%s/%02d.json' "$SERIAL_CREDIT_TETRAD_RECEIPTS" "$completed")
  child_log=$(printf '%s/%02d.log' "$SERIAL_CREDIT_TETRAD_CHILD_LOGS" "$completed")
  serial_credit_require_idle
  "$SERIAL_CREDIT_PYTHON" "$LAUNCHER" --manifest "$SERIAL_CREDIT_MANIFEST" \
    --artifact "$artifact" --credential "$credential" \
    --receipt "$receipt" --stdout-artifact "$child_log" \
    run-tetrad-request --tetrad-manifest "$SERIAL_CREDIT_TETRAD_MANIFEST" \
    --request-id "$request_id" --example-id "$example_id" \
    --state-role "$state_role" >>"$TETRAD_LOG" 2>&1
  serial_credit_wait_idle
  completed=$((completed + 1))
done
[[ $completed -eq 20 ]] || {
  echo 'SERIAL_CREDIT_NO_GO:TETRAD did not complete exactly 20 fresh requests' >&2; exit 83;
}

"$SERIAL_CREDIT_PYTHON" "$RUNNER" --manifest "$SERIAL_CREDIT_MANIFEST" \
  adjudicate-tetrad --tetrad-manifest "$SERIAL_CREDIT_TETRAD_MANIFEST" \
  --authoring "$SERIAL_CREDIT_TETRAD_AUTHORING" \
  --results-dir "$SERIAL_CREDIT_TETRAD_RESULTS" \
  --receipts-dir "$SERIAL_CREDIT_TETRAD_RECEIPTS" \
  --output "$SERIAL_CREDIT_TETRAD_REPORT" \
  >>"$TETRAD_LOG" 2>&1
serial_credit_record --record-type tetrad_adjudication --artifact "$SERIAL_CREDIT_TETRAD_REPORT"

"$SERIAL_CREDIT_PYTHON" \
  "$SERIAL_CREDIT_REPO_DIR/tools/h20/audit_qwen25_7b_serialization_credit.py" \
  --manifest "$SERIAL_CREDIT_MANIFEST" --write-report >>"$TETRAD_LOG" 2>&1

[[ ! -e $SERIAL_CREDIT_READONLY_REAUDIT ]] || {
  echo 'SERIAL_CREDIT_NO_GO:AUDIT read-only re-audit artifact already exists' >&2
  exit 85
}
"$SERIAL_CREDIT_PYTHON" \
  "$SERIAL_CREDIT_REPO_DIR/tools/h20/audit_qwen25_7b_serialization_credit.py" \
  --manifest "$SERIAL_CREDIT_MANIFEST" \
  >"$SERIAL_CREDIT_READONLY_REAUDIT" 2>>"$TETRAD_LOG"

"$SERIAL_CREDIT_PYTHON" - "$SERIAL_CREDIT_READONLY_REAUDIT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["status"] == "PASS"
assert report["decision"] == "SERIALIZATION_CREDIT_PILOT4_PASS"
assert report["ledger_prefix_record_count"] == 37
assert report["failures"] == []
PY

"$SERIAL_CREDIT_PYTHON" - "$SERIAL_CREDIT_FINAL_REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["status"] == "PASS"
assert report["decision"] == "SERIALIZATION_CREDIT_PILOT4_PASS"
assert report["scope"]["training_authorized"] is False
assert report["scope"]["method_selection_status"] == "PENDING_EVIDENCE_NO_SELECTION"
assert report["tetrad"]["effects_reportable"] is False
PY

echo "SERIAL_CREDIT_TETRAD4_PASS=$SERIAL_CREDIT_FINAL_REPORT"
echo 'Tetrad4 is construction/execution evidence only; audit32 and method selection remain pending.'

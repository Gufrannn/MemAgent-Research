#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=scripts/h20/serialization_credit_pilots_common.sh
source "$SCRIPT_DIR/serialization_credit_pilots_common.sh"

serial_credit_sanitize_inherited_environment
serial_credit_require_checkout
serial_credit_require_p0
serial_credit_acquire_lock
serial_credit_require_idle

[[ ! -e $SERIAL_CREDIT_SMSB_ROOT ]] || {
  echo 'SERIAL_CREDIT_NO_GO:SMSB append-only SMSB root already exists' >&2; exit 73;
}
mkdir -p "$SERIAL_CREDIT_SMSB_REPLAYS" "$SERIAL_CREDIT_SMSB_CREDENTIALS" \
  "$SERIAL_CREDIT_SMSB_RECEIPTS" "$SERIAL_CREDIT_SMSB_CHILD_LOGS"
readonly SMSB_LOG=$SERIAL_CREDIT_LOG_ROOT/smsb4.log
readonly RUNNER=$SERIAL_CREDIT_REPO_DIR/tools/h20/run_qwen25_7b_serialization_credit.py
readonly LAUNCHER=$SERIAL_CREDIT_REPO_DIR/tools/h20/launch_qwen25_7b_serialization_credit_child.py

export CUDA_VISIBLE_DEVICES=$SERIAL_CREDIT_GPUS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=0
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SERIAL_CREDIT_REPO_DIR
export TOKENIZERS_PARALLELISM=false
cd "$SERIAL_CREDIT_REPO_DIR"

"$SERIAL_CREDIT_PYTHON" "$LAUNCHER" --manifest "$SERIAL_CREDIT_MANIFEST" \
  --artifact "$SERIAL_CREDIT_SMSB_CAPTURES" \
  --credential "$SERIAL_CREDIT_SMSB_CREDENTIALS/capture.json" \
  --receipt "$SERIAL_CREDIT_SMSB_RECEIPTS/capture.json" \
  --stdout-artifact "$SERIAL_CREDIT_SMSB_CHILD_LOGS/capture.log" \
  capture-smsb >>"$SMSB_LOG" 2>&1
serial_credit_wait_idle

mapfile -t example_ids < <("$SERIAL_CREDIT_PYTHON" -c \
  'import json,sys; x=json.load(open(sys.argv[1])); print("\n".join(str(r["example_id"]) for r in x["pilot_rows"]))' \
  "$SERIAL_CREDIT_RESOLVED")
[[ ${#example_ids[@]} -eq 4 ]] || {
  echo 'SERIAL_CREDIT_NO_GO:SMSB P0 does not contain four pilot IDs' >&2; exit 74;
}

completed=0
for example_id in "${example_ids[@]}"; do
  for regime in temperature_zero matched_seed independent_seed; do
    artifact=$SERIAL_CREDIT_SMSB_REPLAYS/${example_id}_${regime}.json
    credential=$(printf '%s/%02d.json' "$SERIAL_CREDIT_SMSB_CREDENTIALS" "$completed")
    receipt=$(printf '%s/%02d.json' "$SERIAL_CREDIT_SMSB_RECEIPTS" "$completed")
    child_log=$(printf '%s/%02d.log' "$SERIAL_CREDIT_SMSB_CHILD_LOGS" "$completed")
    serial_credit_require_idle
    "$SERIAL_CREDIT_PYTHON" "$LAUNCHER" --manifest "$SERIAL_CREDIT_MANIFEST" \
      --artifact "$artifact" --credential "$credential" \
      --receipt "$receipt" --stdout-artifact "$child_log" \
      replay-smsb --captures "$SERIAL_CREDIT_SMSB_CAPTURES" \
      --example-id "$example_id" --regime "$regime" \
      >>"$SMSB_LOG" 2>&1
    serial_credit_wait_idle
    completed=$((completed + 1))
  done
done
[[ $completed -eq 12 ]] || {
  echo 'SERIAL_CREDIT_NO_GO:SMSB did not complete exactly 12 fresh replays' >&2; exit 75;
}

"$SERIAL_CREDIT_PYTHON" "$RUNNER" --manifest "$SERIAL_CREDIT_MANIFEST" \
  adjudicate-smsb --captures "$SERIAL_CREDIT_SMSB_CAPTURES" \
  --replays-dir "$SERIAL_CREDIT_SMSB_REPLAYS" \
  --receipts-dir "$SERIAL_CREDIT_SMSB_RECEIPTS" \
  --output "$SERIAL_CREDIT_SMSB_REPORT" \
  >>"$SMSB_LOG" 2>&1
serial_credit_record --record-type smsb_adjudication --artifact "$SERIAL_CREDIT_SMSB_REPORT"

"$SERIAL_CREDIT_PYTHON" - "$SERIAL_CREDIT_SMSB_REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["status"] == "PASS"
assert report["decision"] == "PASS_E_DET_SINGLE_REQUEST"
assert report["E_det_pass"] is True
assert report["training_authorized"] is False
assert report["method_selection_status"] == "PENDING_EVIDENCE_NO_SELECTION"
PY

echo "SERIAL_CREDIT_SMSB4_PASS=$SERIAL_CREDIT_SMSB_REPORT"
echo 'L2 is report-only; this PASS unlocks deterministic Tetrad4, not method training.'

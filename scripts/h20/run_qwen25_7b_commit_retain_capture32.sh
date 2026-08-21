#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=scripts/h20/commit_retain_capture32_common.sh
source "$SCRIPT_DIR/commit_retain_capture32_common.sh"

capture32_sanitize_environment
capture32_require_checkout
capture32_acquire_locks

CAPTURE32_RUNNER_STARTED=0
capture32_cleanup_on_exit() {
  local status=$?
  trap - EXIT
  if [[ $CAPTURE32_RUNNER_STARTED == 1 ]]; then
    # Keep all inherited per-device/legacy locks held while we prove that no
    # capture child remains on either selected GPU.  We never kill an
    # unclassified process on this shared host.
    capture32_wait_idle || status=81
  fi
  exit "$status"
}
trap capture32_cleanup_on_exit EXIT

capture32_export_runtime
capture32_require_h20_binding
capture32_require_p0
capture32_require_idle

[[ ! -e $CAPTURE32_CREDENTIAL && ! -e $CAPTURE32_CREDENTIAL_CONSUMPTION && \
   ! -e $CAPTURE32_CAPTURE && \
   ! -e $CAPTURE32_RUN_RECEIPT && ! -e $CAPTURE32_FINAL && \
   ! -e $CAPTURE32_TERMINAL_ANCHOR ]] || \
  capture32_die 73 'CAPTURE append-only capture artifact exists; use a new run ID for all 32 pairs'

mkdir -p "$(dirname "$CAPTURE32_CAPTURE")"
readonly CAPTURE32_CAPTURE_LOG=$CAPTURE32_LOG_ROOT/commit_retain_capture32.log
[[ ! -e $CAPTURE32_CAPTURE_LOG ]] || \
  capture32_die 73 'CAPTURE append-only capture log already exists'

cd "$CAPTURE32_REPO_DIR"
capture32_issue_capture_credential
CAPTURE32_RUNNER_STARTED=1
"$CAPTURE32_PYTHON" tools/h20/run_qwen25_7b_commit_retain_capture32.py \
  --manifest "$CAPTURE32_MANIFEST" capture \
  --credential "$CAPTURE32_CREDENTIAL" >>"$CAPTURE32_CAPTURE_LOG" 2>&1
capture32_wait_idle
CAPTURE32_RUNNER_STARTED=0
capture32_record_complete

"$CAPTURE32_PYTHON" tools/h20/preflight_qwen25_7b_commit_retain_capture32.py \
  --manifest "$CAPTURE32_MANIFEST" --write-final >>"$CAPTURE32_CAPTURE_LOG" 2>&1
"$CAPTURE32_PYTHON" tools/h20/preflight_qwen25_7b_commit_retain_capture32.py \
  --manifest "$CAPTURE32_MANIFEST" --verify-existing >>"$CAPTURE32_CAPTURE_LOG" 2>&1

[[ -f $CAPTURE32_FINAL && -f $CAPTURE32_TERMINAL_ANCHOR ]] || \
  capture32_die 76 'AUDIT did not produce both final report and external terminal anchor'

echo "CAPTURE32_AUDIT_COMPLETE=$CAPTURE32_FINAL"
echo "CAPTURE32_EXTERNAL_TERMINAL_ANCHOR=$CAPTURE32_TERMINAL_ANCHOR"
echo 'Exact-32 capture/audit only: no capture4 fill, run stitching, trainer, actor update, or method claim.'

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=scripts/h20/commit_retain_capture32_common.sh
source "$SCRIPT_DIR/commit_retain_capture32_common.sh"

capture32_sanitize_environment
capture32_require_checkout
capture32_acquire_locks
capture32_export_runtime
capture32_require_h20_binding
capture32_require_idle

[[ ! -e $CAPTURE32_LOG_ROOT ]] || \
  capture32_die 73 'P0 append-only run root exists; choose a new run ID'
[[ ! -e $CAPTURE32_PREREG_ANCHOR && ! -e $CAPTURE32_TERMINAL_ANCHOR ]] || \
  capture32_die 74 'P0 append-only external anchor exists; choose a new run ID'

cd "$CAPTURE32_REPO_DIR"
"$CAPTURE32_PYTHON" tools/h20/preflight_qwen25_7b_commit_retain_capture32.py \
  --manifest "$CAPTURE32_MANIFEST" --check-runtime --write-certificate

[[ -f $CAPTURE32_P0 && -f $CAPTURE32_RESOLVED && -f $CAPTURE32_LEDGER && \
   -f $CAPTURE32_PREREG_ANCHOR ]] || \
  capture32_die 75 'P0 supervisor did not create the complete certificate and external anchor set'
capture32_require_p0

echo "CAPTURE32_P0_PASS=$CAPTURE32_P0"
echo "CAPTURE32_EXTERNAL_PREREGISTRATION_ANCHOR=$CAPTURE32_PREREG_ANCHOR"
echo 'P0 only: no generation, trainer, actor update, or method selection occurred.'

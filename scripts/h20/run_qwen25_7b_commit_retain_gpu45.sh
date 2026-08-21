#!/usr/bin/env bash
set -euo pipefail

export COMMIT_RETAIN_CAPTURE_PROFILE=gpu45
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=scripts/h20/commit_retain_capture_common.sh
source "$SCRIPT_DIR/commit_retain_capture_common.sh"

commit_retain_sanitize_environment
commit_retain_require_checkout
commit_retain_require_p0
commit_retain_acquire_lock
commit_retain_require_idle

[[ ! -e $COMMIT_RETAIN_CREDENTIAL && ! -e $COMMIT_RETAIN_CAPTURE && ! -e $COMMIT_RETAIN_RUN_RECEIPT && ! -e $COMMIT_RETAIN_FINAL ]] || {
  echo 'COMMIT_RETAIN_NO_GO:CAPTURE append-only GPU45 capture artifact already exists' >&2; exit 73;
}
mkdir -p "$(dirname "$COMMIT_RETAIN_CAPTURE")"
readonly CAPTURE_LOG=$COMMIT_RETAIN_LOG_ROOT/commit_retain_capture.log

export CUDA_VISIBLE_DEVICES=$COMMIT_RETAIN_GPUS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=0
export PYTHONNOUSERSITE=1
export PYTHONPATH=$COMMIT_RETAIN_REPO_DIR
export TOKENIZERS_PARALLELISM=false

cd "$COMMIT_RETAIN_REPO_DIR"
commit_retain_issue_capture_credential
"$COMMIT_RETAIN_PYTHON" tools/h20/run_qwen25_7b_commit_retain.py \
  --manifest "$COMMIT_RETAIN_MANIFEST" capture \
  --credential "$COMMIT_RETAIN_CREDENTIAL" >>"$CAPTURE_LOG" 2>&1
commit_retain_wait_idle
commit_retain_record --record-type capture_complete --artifact "$COMMIT_RETAIN_CAPTURE"

"$COMMIT_RETAIN_PYTHON" tools/h20/audit_qwen25_7b_commit_retain.py \
  --manifest "$COMMIT_RETAIN_MANIFEST" --write-final >>"$CAPTURE_LOG" 2>&1
"$COMMIT_RETAIN_PYTHON" tools/h20/audit_qwen25_7b_commit_retain.py \
  --manifest "$COMMIT_RETAIN_MANIFEST" --verify-existing >>"$CAPTURE_LOG" 2>&1

echo "COMMIT_RETAIN_GPU45_CAPTURE_AUDIT_COMPLETE=$COMMIT_RETAIN_FINAL"
echo 'GPU45 capture/audit only: no trainer, no actor update, no method claim.'

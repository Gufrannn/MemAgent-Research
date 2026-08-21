#!/usr/bin/env bash
set -euo pipefail

export COMMIT_RETAIN_CAPTURE_PROFILE=gpu45
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=scripts/h20/commit_retain_capture_common.sh
source "$SCRIPT_DIR/commit_retain_capture_common.sh"

commit_retain_sanitize_environment
commit_retain_require_checkout
commit_retain_acquire_lock
commit_retain_require_idle

[[ ! -e $COMMIT_RETAIN_LOG_ROOT ]] || {
  echo 'COMMIT_RETAIN_NO_GO:P0 append-only GPU45 run root exists; choose a new run ID' >&2; exit 73;
}

export CUDA_VISIBLE_DEVICES=$COMMIT_RETAIN_GPUS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=0
export PYTHONNOUSERSITE=1
export PYTHONPATH=$COMMIT_RETAIN_REPO_DIR
export TOKENIZERS_PARALLELISM=false

cd "$COMMIT_RETAIN_REPO_DIR"
"$COMMIT_RETAIN_PYTHON" tools/h20/preflight_qwen25_7b_commit_retain.py \
  --manifest "$COMMIT_RETAIN_MANIFEST" --check-runtime --write-certificate

echo "COMMIT_RETAIN_GPU45_P0_PASS=$COMMIT_RETAIN_P0"

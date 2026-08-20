#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=scripts/h20/serialization_credit_pilots_common.sh
source "$SCRIPT_DIR/serialization_credit_pilots_common.sh"

serial_credit_sanitize_inherited_environment
serial_credit_require_checkout
serial_credit_acquire_lock
serial_credit_require_idle

[[ ! -e $SERIAL_CREDIT_LOG_ROOT ]] || {
  echo 'SERIAL_CREDIT_NO_GO:P0 append-only run root already exists; choose a new run ID' >&2
  exit 73
}

export CUDA_VISIBLE_DEVICES=$SERIAL_CREDIT_GPUS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=0
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SERIAL_CREDIT_REPO_DIR
export TOKENIZERS_PARALLELISM=false

cd "$SERIAL_CREDIT_REPO_DIR"
"$SERIAL_CREDIT_PYTHON" \
  "$SERIAL_CREDIT_REPO_DIR/tools/h20/preflight_qwen25_7b_serialization_credit.py" \
  --manifest "$SERIAL_CREDIT_MANIFEST" --check-runtime --write-certificate

echo "SERIAL_CREDIT_P0_PASS=$SERIAL_CREDIT_P0"

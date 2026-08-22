#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RWWPO_MANIFEST=$RWWPO_REPO_DIR/manifests/h20/qwen25_7b_tokenwise_tf_controller_seed2026.json
export RWWPO_OBJECTIVE_VARIANT=original_tokenwise
exec bash "$SCRIPT_DIR/run_qwen25_7b_tf_rwwpo.sh"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RWWPO_EXPECTED_BRANCH=h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822
export RWWPO_MANIFEST=${RWWPO_MANIFEST:-$RWWPO_REPO_DIR/manifests/h20/qwen25_7b_tf_rwwpo_seed2026.json}
export RWWPO_OBJECTIVE_VARIANT=${RWWPO_OBJECTIVE_VARIANT:-whole_prefix}
export RWWPO_CONTROLLER_VARIANT=feasible_backtracking
exec bash "$SCRIPT_DIR/run_qwen25_7b_rwwpo.sh"

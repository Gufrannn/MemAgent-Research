#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/gatea_frozen_common.sh"
gatea_require_clean_frozen_checkout
gatea_acquire_run_lock
gatea_require_p0_commit

[[ ! -e $GATEA_FRESH_OUTPUT ]] || { echo "GATE_A_NO_GO:P0 fresh output exists: $GATEA_FRESH_OUTPUT" >&2; exit 72; }
readonly GATEA_LOG=$GATEA_LOG_ROOT/${GATEA_FRESH_EXP}.log
[[ ! -e $GATEA_LOG ]] || { echo "GATE_A_NO_GO:P0 fresh log exists: $GATEA_LOG" >&2; exit 73; }
mkdir -p "$GATEA_LOG_ROOT" "$GATEA_CERTIFICATE_ROOT"

"$GATEA_PYTHON" "$GATEA_CODE/tools/h20/preflight_qwen25_7b_gatea.py" \
  --manifest "$GATEA_MANIFEST" --phase fresh --check-runtime
gatea_require_p0_commit
gatea_require_declared_gpus_idle
gatea_export_audit_environment
export GATE_A_EXPERIMENT_NAME=$GATEA_FRESH_EXP

env WORK_ROOT="$GATEA_WORK_ROOT" CODE="$GATEA_CODE" PYTHON="$GATEA_PYTHON" \
PHASE=fresh EXP="$GATEA_FRESH_EXP" RUN_SEED=2026 \
TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 \
N_GPUS=2 FSDP_SIZE=2 REWARD_MANAGER=naive \
GPU_MEMORY_UTILIZATION=0.55 \
bash "$GATEA_CODE/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$GATEA_LOG"

"$GATEA_PYTHON" "$GATEA_CODE/tools/h20/audit_qwen25_7b_gatea.py" \
  --manifest "$GATEA_MANIFEST" --phase p1 --write-report

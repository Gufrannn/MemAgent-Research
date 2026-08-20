#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/gatea_frozen_common.sh"
gatea_require_clean_frozen_checkout
gatea_export_audit_environment
export GATE_A_EXPERIMENT_NAME=$FRESH_EXP

[[ ! -e $FRESH_OUTPUT ]] || { echo "GATE_A_NO_GO:P0 fresh output exists: $FRESH_OUTPUT" >&2; exit 72; }
readonly LOG=$LOG_ROOT/${FRESH_EXP}.log
[[ ! -e $LOG ]] || { echo "GATE_A_NO_GO:P0 fresh log exists: $LOG" >&2; exit 73; }
mkdir -p "$LOG_ROOT" "$CERTIFICATE_ROOT"

"$PYTHON" "$CODE/tools/h20/preflight_qwen25_7b_gatea.py" \
  --manifest "$MANIFEST" --check-runtime --write-certificate
gatea_require_p0_commit
gatea_require_declared_gpus_idle

WORK_ROOT=$WORK_ROOT CODE=$CODE PYTHON=$PYTHON \
PHASE=fresh EXP=$FRESH_EXP RUN_SEED=2026 \
TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 \
N_GPUS=2 FSDP_SIZE=2 REWARD_MANAGER=naive \
GPU_MEMORY_UTILIZATION=0.55 \
bash "$CODE/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$LOG"

"$PYTHON" "$CODE/tools/h20/audit_qwen25_7b_gatea.py" \
  --manifest "$MANIFEST" --phase p1 --write-report

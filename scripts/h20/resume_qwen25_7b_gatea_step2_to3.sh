#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/gatea_frozen_common.sh"
gatea_require_clean_frozen_checkout
gatea_export_audit_environment
export GATE_A_EXPERIMENT_NAME=$RESUME_EXP

[[ -d $RESUME_SOURCE/actor && -f $RESUME_SOURCE/data.pt ]] || {
  echo "GATE_A_NO_GO:P2 missing exact P1 step2 source: $RESUME_SOURCE" >&2; exit 74;
}
[[ ! -e $RESUME_OUTPUT ]] || { echo "GATE_A_NO_GO:P2 resume output exists: $RESUME_OUTPUT" >&2; exit 75; }
readonly LOG=$LOG_ROOT/${RESUME_EXP}.log
[[ ! -e $LOG ]] || { echo "GATE_A_NO_GO:P2 resume log exists: $LOG" >&2; exit 76; }
[[ -f $CERTIFICATE_ROOT/p1_audit_report.json ]] || {
  echo 'GATE_A_NO_GO:P2 missing P1 audit certificate' >&2; exit 77;
}
gatea_require_p0_commit
"$PYTHON" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "PASS")' \
  "$CERTIFICATE_ROOT/p1_audit_report.json"
gatea_require_declared_gpus_idle

WORK_ROOT=$WORK_ROOT CODE=$CODE PYTHON=$PYTHON \
PHASE=resume EXP=$RESUME_EXP RESUME_FROM=$RESUME_SOURCE RUN_SEED=2026 \
TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 \
N_GPUS=2 FSDP_SIZE=2 REWARD_MANAGER=naive \
GPU_MEMORY_UTILIZATION=0.55 \
bash "$CODE/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$LOG"

"$PYTHON" "$CODE/tools/h20/audit_qwen25_7b_gatea.py" \
  --manifest "$MANIFEST" --phase final --write-report

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/gatea_frozen_common.sh"
gatea_require_clean_frozen_checkout
gatea_export_audit_environment
export GATE_A_EXPERIMENT_NAME=$GATEA_RESUME_EXP

[[ -d $GATEA_RESUME_SOURCE/actor && -f $GATEA_RESUME_SOURCE/data.pt ]] || {
  echo "GATE_A_NO_GO:P2 missing exact P1 step2 source: $GATEA_RESUME_SOURCE" >&2; exit 74;
}
[[ ! -e $GATEA_RESUME_OUTPUT ]] || { echo "GATE_A_NO_GO:P2 resume output exists: $GATEA_RESUME_OUTPUT" >&2; exit 75; }
readonly GATEA_LOG=$GATEA_LOG_ROOT/${GATEA_RESUME_EXP}.log
[[ ! -e $GATEA_LOG ]] || { echo "GATE_A_NO_GO:P2 resume log exists: $GATEA_LOG" >&2; exit 76; }
[[ -f $GATEA_CERTIFICATE_ROOT/p1_audit_report.json ]] || {
  echo 'GATE_A_NO_GO:P2 missing P1 audit certificate' >&2; exit 77;
}
gatea_require_p0_commit
"$GATEA_PYTHON" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "PASS")' \
  "$GATEA_CERTIFICATE_ROOT/p1_audit_report.json"
gatea_require_declared_gpus_idle

env WORK_ROOT="$GATEA_WORK_ROOT" CODE="$GATEA_CODE" PYTHON="$GATEA_PYTHON" \
PHASE=resume EXP="$GATEA_RESUME_EXP" RESUME_FROM="$GATEA_RESUME_SOURCE" RUN_SEED=2026 \
TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 \
N_GPUS=2 FSDP_SIZE=2 REWARD_MANAGER=naive \
GPU_MEMORY_UTILIZATION=0.55 \
bash "$GATEA_CODE/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$GATEA_LOG"

"$GATEA_PYTHON" "$GATEA_CODE/tools/h20/audit_qwen25_7b_gatea.py" \
  --manifest "$GATEA_MANIFEST" --phase final --write-report

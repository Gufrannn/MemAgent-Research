#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/rwwpo_common.sh"
[[ ${RWWPO_PHASE:-} == t5 || ${RWWPO_PHASE:-} == continue ]] || { echo 'RWWPO_NO_GO:RWWPO_PHASE=t5|continue' >&2; exit 74; }
rwwpo_require_checkout
PREFLIGHT=("$RWWPO_PYTHON" "$RWWPO_REPO_DIR/tools/h20/preflight_rwwpo.py" --manifest "$RWWPO_MANIFEST" --expected-commit "$RWWPO_EXPECTED_COMMIT" --gpu-pair "$GPU_PAIR" --e0 "$RWWPO_E0" --e1 "$RWWPO_E1" --baseline-import "$RWWPO_BASELINE" --original-resolved-manifest "$RWWPO_ORIGINAL_RESOLVED_MANIFEST" --original-resolved-sha256 "$RWWPO_ORIGINAL_RESOLVED_SHA256" --phase "$RWWPO_PHASE")
if [[ $RWWPO_PHASE == continue ]]; then
  [[ ${RWWPO_RESUME_STEP:-} =~ ^(5|10|15|20)$ ]] || { echo 'RWWPO_NO_GO:RWWPO_RESUME_STEP must be prior anchor' >&2; exit 75; }
  [[ ${RWWPO_TARGET_STEP:-} =~ ^(10|15|20|25)$ && $RWWPO_TARGET_STEP -gt $RWWPO_RESUME_STEP ]] || { echo 'RWWPO_NO_GO:invalid target anchor' >&2; exit 76; }
  PREFLIGHT+=(--resume-step "$RWWPO_RESUME_STEP")
fi
"${PREFLIGHT[@]}"
rwwpo_acquire_gpu_locks
rwwpo_require_idle
rwwpo_export_runtime
mkdir -p "$(dirname "$RWWPO_LOG")"
if [[ $RWWPO_PHASE == t5 ]]; then
  [[ ! -e $RWWPO_OUTPUT ]] || { echo 'RWWPO_NO_GO:T5 output exists; choose a new semantic run id' >&2; exit 77; }
  PHASE=fresh FRESH_TOTAL_STEPS=5
  EXTRA=()
  RWWPO_ATTEMPT_ID=${RWWPO_ATTEMPT_ID:-t5_primary}
else
  [[ -d $RWWPO_OUTPUT/global_step_$RWWPO_RESUME_STEP/actor && -f $RWWPO_OUTPUT/global_step_$RWWPO_RESUME_STEP/data.pt ]] || { echo 'RWWPO_NO_GO:resume checkpoint incomplete' >&2; exit 78; }
  PHASE=resume RESUME_TOTAL_STEPS=$RWWPO_TARGET_STEP RESUME_SOURCE_STEP=$RWWPO_RESUME_STEP
  EXTRA=(RESUME_FROM="$RWWPO_OUTPUT/global_step_$RWWPO_RESUME_STEP")
  RWWPO_ATTEMPT_ID=${RWWPO_ATTEMPT_ID:-resume_${RWWPO_RESUME_STEP}_to_${RWWPO_TARGET_STEP}}
fi
export RWWPO_ATTEMPT_ID
env WORK_ROOT="$RWWPO_WORK_ROOT" CODE="$RWWPO_REPO_DIR" PYTHON="$RWWPO_PYTHON" \
  MODEL="$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct" TRAIN="$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  VAL="$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" PHASE="$PHASE" EXP="$RWWPO_EXPERIMENT" RUN_SEED=2026 \
  TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 N_GPUS=2 FSDP_SIZE=2 REWARD_MANAGER=naive GPU_MEMORY_UTILIZATION=0.55 \
  FRESH_TOTAL_STEPS="${FRESH_TOTAL_STEPS:-5}" RESUME_TOTAL_STEPS="${RESUME_TOTAL_STEPS:-5}" RESUME_SOURCE_STEP="${RESUME_SOURCE_STEP:-0}" \
  SAVE_FREQ=5 MAX_ACTOR_CKPT_TO_KEEP=5 "${EXTRA[@]}" \
  bash "$RWWPO_REPO_DIR/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$RWWPO_LOG"

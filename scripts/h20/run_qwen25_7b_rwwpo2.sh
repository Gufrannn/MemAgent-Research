#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/rwwpo2_common.sh"

rwwpo2_require_checkout
rwwpo2_consume_attempt_id

PREFLIGHT=(
  "$RWWPO_PYTHON" "$RWWPO_REPO_DIR/tools/h20/preflight_rwwpo2.py"
  --resolved-contract "$RWWPO_RESOLVED_CONTRACT"
  --resolved-contract-sha256 "$RWWPO_RESOLVED_CONTRACT_SHA256"
  --expected-commit "$RWWPO_EXPECTED_COMMIT"
  --gpu-pair "$GPU_PAIR"
  --cell "$RWWPO_CELL"
  --experiment-seed "$RWWPO_EXPERIMENT_SEED"
  --target-round "$RWWPO_TARGET_ROUND"
  --phase "$RWWPO_PHASE"
  --e0 "$RWWPO_E0"
  --data-boundary-audit "$RWWPO_DATA_BOUNDARY_AUDIT"
  --base-protocol-audit "$RWWPO_BASE_PROTOCOL_AUDIT"
  --release-test-receipt "$RWWPO_RELEASE_TEST_RECEIPT"
  --release-test-receipt-sha256 "$RWWPO_RELEASE_TEST_RECEIPT_SHA256"
  --original-resolved-manifest "$RWWPO_ORIGINAL_RESOLVED_MANIFEST"
  --original-resolved-sha256 "$RWWPO_ORIGINAL_RESOLVED_SHA256"
  --output "$RWWPO_PREFLIGHT"
)

if [[ $RWWPO_PHASE == resume ]]; then
  [[ ${RWWPO_RESUME_ROUND:-} =~ ^[0-9]+$ && $RWWPO_RESUME_ROUND -gt 0 \
      && $((RWWPO_RESUME_ROUND % 10)) -eq 0 ]] || {
    echo 'RWWPO2_NO_GO:resume round must be a positive multiple of 10' >&2; exit 85;
  }
  [[ -f ${RWWPO_LINEAGE_PARENT_RECEIPT:-} ]] || {
    echo 'RWWPO2_NO_GO:missing authenticated lineage-parent receipt' >&2; exit 86;
  }
  [[ -d ${RWWPO_PARENT_OUTPUT_ROOT:-}/global_step_${RWWPO_RESUME_ROUND}/actor \
      && -f ${RWWPO_PARENT_OUTPUT_ROOT:-}/global_step_${RWWPO_RESUME_ROUND}/data.pt ]] || {
    echo 'RWWPO2_NO_GO:parent recovery checkpoint incomplete' >&2; exit 87;
  }
  PREFLIGHT+=(
    --lineage-parent "$RWWPO_LINEAGE_PARENT_RECEIPT"
    --resume-round "$RWWPO_RESUME_ROUND"
  )
  PHASE=resume
  RESUME_FROM=$RWWPO_PARENT_OUTPUT_ROOT/global_step_$RWWPO_RESUME_ROUND
  RWWPO_LINEAGE_START_ROUND=$((RWWPO_RESUME_ROUND + 1))
else
  [[ ! -e $RWWPO_OUTPUT ]] || { echo 'RWWPO2_NO_GO:fresh output exists' >&2; exit 88; }
  PHASE=fresh
  RESUME_FROM=
  RWWPO_RESUME_ROUND=0
  RWWPO_LINEAGE_START_ROUND=1
fi
if [[ $RWWPO_TARGET_ROUND == 400 ]]; then
  [[ -f ${RWWPO_R50_PROGRAM_GATE:-} \
     && ${RWWPO_R50_PROGRAM_GATE_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || {
    echo 'RWWPO2_NO_GO:R400 requires the frozen R50 program gate' >&2; exit 89;
  }
  [[ -f ${RWWPO_CONFIRMATION_SEAL:-} \
     && ${RWWPO_CONFIRMATION_SEAL_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || {
    echo 'RWWPO2_NO_GO:R400 requires a sealed disjoint confirmation receipt' >&2; exit 90;
  }
  PREFLIGHT+=(--r50-program-gate "$RWWPO_R50_PROGRAM_GATE")
  PREFLIGHT+=(--r50-program-gate-sha256 "$RWWPO_R50_PROGRAM_GATE_SHA256")
  PREFLIGHT+=(--confirmation-seal "$RWWPO_CONFIRMATION_SEAL")
  PREFLIGHT+=(--confirmation-seal-sha256 "$RWWPO_CONFIRMATION_SEAL_SHA256")
fi
"${PREFLIGHT[@]}"

read -r RWWPO_TAU_THETA RWWPO_TAU_LOGPROB RWWPO_TAU_GRADIENT \
  RWWPO_BEHAVIOR_COEFFICIENT_TOLERANCE \
  RWWPO_BEHAVIOR_GRADIENT_TOLERANCE \
  RWWPO_GRADIENT_SKETCH_CHUNK_ELEMENTS \
  RWWPO_FSDP_PARAMETER_COMMIT_PRIMITIVE \
  RWWPO_FSDP_WRITEBACK_MAX_WALL_SECONDS \
  RWWPO_MAX_TRIAL_FORWARD_SECONDS \
  RWWPO_RESOLVED_CONTRACT_REPORT_SHA256 \
  RWWPO_SOURCE_MANIFEST_SHA256 < <(
  "$RWWPO_PYTHON" -c '
import json,sys
r=json.load(open(sys.argv[1]))
t=r["numeric_thresholds"]
print(t["tau_theta"],t["tau_logprob"],t["tau_gradient"],
      r["behavior_coefficient_tolerance"],r["behavior_gradient_tolerance"],
      r["gradient_sketch_chunk_elements"],
      r["fsdp_parameter_commit_primitive"],
      r["fsdp_parameter_writeback_max_wall_seconds"],
      r["max_trial_forward_wall_seconds_per_transaction"],
      r["report_sha256"],r["source_manifest_sha256"])
' "$RWWPO_RESOLVED_CONTRACT"
)
export RWWPO_TAU_THETA RWWPO_TAU_LOGPROB RWWPO_TAU_GRADIENT
export RWWPO_BEHAVIOR_COEFFICIENT_TOLERANCE
export RWWPO_BEHAVIOR_GRADIENT_TOLERANCE RWWPO_LINEAGE_START_ROUND
export RWWPO_GRADIENT_SKETCH_CHUNK_ELEMENTS
export RWWPO_FSDP_PARAMETER_COMMIT_PRIMITIVE
export RWWPO_FSDP_WRITEBACK_MAX_WALL_SECONDS
export RWWPO_MAX_TRIAL_FORWARD_SECONDS
export RWWPO_RESOLVED_CONTRACT_SHA256 RWWPO_RESOLVED_CONTRACT_REPORT_SHA256
export RWWPO_SOURCE_MANIFEST_SHA256

rwwpo2_acquire_gpu_locks
rwwpo2_require_idle_twice
rwwpo2_export_runtime

EXTRA=()
if [[ $PHASE == resume ]]; then
  EXTRA+=(RESUME_FROM="$RESUME_FROM")
fi

# No S128 path is passed to the training process.  The framework still requires
# a syntactically valid val_files value even with validation disabled, so it is
# bound to the already-authenticated training parquet and never evaluated.
env WORK_ROOT="$RWWPO_WORK_ROOT" CODE="$RWWPO_REPO_DIR" PYTHON="$RWWPO_PYTHON" \
  MODEL="$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  TRAIN="$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  VAL="$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  PHASE="$PHASE" EXP="$RWWPO_EXPERIMENT" RUN_SEED="$RWWPO_EXPERIMENT_SEED" \
  TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 PPO_EPOCHS=2 \
  N_GPUS=2 FSDP_SIZE=2 REWARD_MANAGER=naive GPU_MEMORY_UTILIZATION=0.55 \
  FRESH_TOTAL_STEPS="$RWWPO_TARGET_ROUND" RESUME_TOTAL_STEPS="$RWWPO_TARGET_ROUND" \
  RESUME_SOURCE_STEP="$RWWPO_RESUME_ROUND" SAVE_FREQ=10 MAX_ACTOR_CKPT_TO_KEEP=2 \
  "${EXTRA[@]}" \
  bash "$RWWPO_REPO_DIR/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$RWWPO_LOG"

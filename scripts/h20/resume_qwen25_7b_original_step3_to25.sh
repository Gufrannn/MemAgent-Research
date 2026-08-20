#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/original_t25_common.sh"
t25_require_clean_checkout
t25_acquire_run_lock
t25_require_p0

[[ ! -e $T25_OUTPUT ]] || { echo "ORIGINAL_T25_NO_GO: output exists: $T25_OUTPUT" >&2; exit 72; }
[[ ! -e $T25_LOG ]] || { echo "ORIGINAL_T25_NO_GO: log exists: $T25_LOG" >&2; exit 73; }
[[ -d $T25_SOURCE/actor && -f $T25_SOURCE/data.pt ]] || {
  echo "ORIGINAL_T25_NO_GO: exact Gate A r5 global_step_3 is incomplete: $T25_SOURCE" >&2; exit 74;
}

"$T25_PYTHON" "$T25_REPO_DIR/tools/h20/preflight_qwen25_7b_original_t25.py" \
  --manifest "$T25_MANIFEST" --phase run --check-runtime
t25_require_gpus_idle
t25_export_execution_evidence

env WORK_ROOT="$T25_WORK_ROOT" CODE="$T25_REPO_DIR" PYTHON="$T25_PYTHON" \
MODEL="$T25_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
TRAIN="$T25_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
VAL="$T25_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
PHASE=resume EXP="$T25_EXPERIMENT" RESUME_FROM="$T25_SOURCE" RUN_SEED=2026 \
TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 \
N_GPUS=2 FSDP_SIZE=2 REWARD_MANAGER=naive GPU_MEMORY_UTILIZATION=0.55 \
RESUME_TOTAL_STEPS=25 RESUME_SOURCE_STEP=3 SAVE_FREQ=5 MAX_ACTOR_CKPT_TO_KEEP=5 \
EMIT_TRAINER_OVERRIDES=0 \
bash "$T25_REPO_DIR/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$T25_LOG"

"$T25_PYTHON" "$T25_REPO_DIR/tools/h20/audit_qwen25_7b_original_t25.py" \
  --manifest "$T25_MANIFEST" --write-report

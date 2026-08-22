#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'usage: continue_qwen25_7b_mic.sh SOURCE_STEP TARGET_STEP' >&2; exit 64; }
SOURCE_STEP=$1; TARGET_STEP=$2
case "$SOURCE_STEP:$TARGET_STEP" in 5:10|10:15|15:20|20:25) ;; *) echo 'MIC_NO_GO: continuation must follow 5->10->15->20->25' >&2; exit 65;; esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/mic_common.sh"
mic_require_checkout; mic_require_training_gates
mic_require_gate "$MIC_CERT/t${SOURCE_STEP}_audit.json" MIC_T${SOURCE_STEP}_AUDIT_PASS
if [[ $SOURCE_STEP -eq 5 ]]; then
  mic_require_gate "$MIC_CERT/t5_health.json" MIC_T5_TRAINING_HEALTH_PASS
fi
mic_acquire_gpu_locks; mic_require_idle
[[ -d $MIC_OUTPUT/global_step_${SOURCE_STEP}/actor && -f $MIC_OUTPUT/global_step_${SOURCE_STEP}/data.pt ]] || {
  echo 'MIC_NO_GO: resume checkpoint incomplete' >&2; exit 80;
}
[[ ! -e $MIC_OUTPUT/global_step_${TARGET_STEP} ]] || { echo 'MIC_NO_GO: target checkpoint exists' >&2; exit 81; }
mic_export_training
env WORK_ROOT="$MEMAGENT_MIC_WORK_ROOT" CODE="$MEMAGENT_MIC_REPO_DIR" PYTHON="$MIC_PYTHON" \
  MODEL="$MEMAGENT_MIC_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  TRAIN="$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  VAL="$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  PHASE=resume EXP="$MIC_EXPERIMENT" RESUME_FROM="$MIC_OUTPUT/global_step_${SOURCE_STEP}" \
  RESUME_SOURCE_STEP="$SOURCE_STEP" RESUME_TOTAL_STEPS="$TARGET_STEP" RUN_SEED=2026 \
  TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 N_GPUS=2 FSDP_SIZE=2 \
  REWARD_MANAGER=naive GPU_MEMORY_UTILIZATION=0.55 SAVE_FREQ=5 MAX_ACTOR_CKPT_TO_KEEP=5 \
  bash "$MEMAGENT_MIC_REPO_DIR/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$MIC_ROOT/train_${SOURCE_STEP}_to_${TARGET_STEP}.log"

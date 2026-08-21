#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/mic_common.sh"
mic_require_checkout; mic_require_training_gates; mic_acquire_gpu_locks; mic_require_idle
[[ ! -e $MIC_OUTPUT && ! -e $MIC_LEDGER && ! -e $MIC_CRITIC_ROOT ]] || {
  echo 'MIC_NO_GO: fresh T5 output already exists; choose a unique run ID' >&2; exit 79;
}
mic_export_training
env WORK_ROOT="$MEMAGENT_MIC_WORK_ROOT" CODE="$MEMAGENT_MIC_REPO_DIR" PYTHON="$MIC_PYTHON" \
  MODEL="$MEMAGENT_MIC_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  TRAIN="$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  VAL="$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  PHASE=fresh EXP="$MIC_EXPERIMENT" RUN_SEED=2026 TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 \
  PPO_MINI_BATCH_SIZE=4 N_GPUS=2 FSDP_SIZE=2 REWARD_MANAGER=naive \
  GPU_MEMORY_UTILIZATION=0.55 FRESH_TOTAL_STEPS=5 SAVE_FREQ=5 MAX_ACTOR_CKPT_TO_KEEP=5 \
  bash "$MEMAGENT_MIC_REPO_DIR/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$MIC_ROOT/train_t5.log"

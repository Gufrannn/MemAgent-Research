#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/data/cw/memagent_work}
CODE=${CODE:-$WORK_ROOT/code/MemAgent}
MODEL=${MODEL:-$WORK_ROOT/models/Qwen2.5-7B-Instruct}
TRAIN=${TRAIN:-$WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet}
VAL=${VAL:-$WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet}
PYTHON=${PYTHON:-$WORK_ROOT/.venv/bin/python}

PHASE=${PHASE:-fresh}
EXP=${EXP:-gate_a_qwen25_7b_seed2026}
RUN_SEED=${RUN_SEED:-2026}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
ROLLOUT_N=${ROLLOUT_N:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
N_GPUS=${N_GPUS:-2}
FSDP_SIZE=${FSDP_SIZE:-$N_GPUS}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}
OUT=$WORK_ROOT/logs/memory_agent/$EXP

case "$PHASE" in
  fresh)
    TOTAL_STEPS=2
    RESUME_MODE=disable
    RESUME_ARGS=()
    if [[ -e "$OUT" ]] && [[ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite non-empty run: $OUT" >&2
      exit 44
    fi
    ;;
  resume)
    TOTAL_STEPS=3
    RESUME_MODE=resume_path
    RESUME_FROM=${RESUME_FROM:-$OUT/global_step_2}
    [[ -d "$RESUME_FROM/actor" && -f "$RESUME_FROM/data.pt" ]] || {
      echo "Missing complete step-2 checkpoint: $RESUME_FROM" >&2
      exit 45
    }
    RESUME_ARGS=(trainer.resume_from_path="$RESUME_FROM")
    ;;
  *) echo "PHASE must be fresh or resume" >&2; exit 46 ;;
esac

[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || {
  echo "CUDA_VISIBLE_DEVICES must explicitly name the GPUs already confirmed free." >&2
  exit 47
}
IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#VISIBLE_GPUS[@]} -eq $N_GPUS && $FSDP_SIZE -eq $N_GPUS ]] || {
  echo "Visible GPU count, N_GPUS and FSDP_SIZE must match." >&2
  exit 48
}
[[ $N_GPUS -eq 2 || $N_GPUS -eq 4 || $N_GPUS -eq 6 || $N_GPUS -eq 8 ]] || {
  echo "Gate A supports only an explicit 2, 4, 6 or 8 GPU allocation." >&2
  exit 48
}

for path in "$PYTHON" "$MODEL/config.json" "$TRAIN" "$VAL"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 49; }
done

mkdir -p "$OUT" "$WORK_ROOT/logs/gate_a" "$WORK_ROOT/cache/ray"
RAY_TMP="$WORK_ROOT/cache/ray/${EXP}_${PHASE}_$$"
mkdir -p "$RAY_TMP"
cd "$CODE"

unset RAY_ADDRESS
export RAY_TMPDIR="$RAY_TMP"
export TMPDIR="$RAY_TMP"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export VLLM_USE_V1=0
export NCCL_DEBUG=WARN

"$PYTHON" -m verl.trainer.main_ppo \
  recurrent.enable=memory \
  recurrent.memory.config.chunk_size=5000 \
  algorithm.adv_estimator=grpo \
  algorithm.grpo_use_adv=False \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  +actor_rollout_ref.rollout.seed="$RUN_SEED" \
  +actor_rollout_ref.rollout.trajectory_seed_mode=independent \
  actor_rollout_ref.rollout.val_kwargs.n=2 \
  trainer.logger=['console'] \
  actor_rollout_ref.actor.optim.lr_warmup_steps=2 \
  actor_rollout_ref.actor.clip_ratio_high=0.20 \
  actor_rollout_ref.actor.entropy_coeff=0.000 \
  data.train_files="$TRAIN" \
  data.val_files="$VAL" \
  data.shuffle=False \
  data.filter_overlong_prompts=True \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  +data.dataloader_num_workers=0 \
  data.truncation=center \
  +data.context_key=context \
  data.max_prompt_length=8192 \
  data.max_response_length=1024 \
  reward_model.reward_manager=thread \
  custom_reward_function.path="$CODE/recurrent/research/hotpotqa_dense_reward.py" \
  custom_reward_function.name=compute_score \
  +custom_reward_function.reward_kwargs.f1_weight=0.95 \
  +custom_reward_function.reward_kwargs.grounded_box_bonus=0.05 \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$FSDP_SIZE" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.project_name=memagent_7b_serialization_credit \
  trainer.experiment_name="$EXP" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=30 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$OUT" \
  ray_init.num_cpus=64 \
  "${RESUME_ARGS[@]}"

echo "Gate A phase=$PHASE finished; Ray temp was $RAY_TMP"

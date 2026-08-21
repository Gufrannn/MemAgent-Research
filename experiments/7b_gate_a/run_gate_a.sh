#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/data/cw/memagent_work}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE=${CODE:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
MODEL=${MODEL:-$WORK_ROOT/models/Qwen2.5-7B-Instruct}
TRAIN=${TRAIN:-$WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet}
VAL=${VAL:-$WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet}
PYTHON=${PYTHON:-$WORK_ROOT/.venv/bin/python}

PHASE=${PHASE:-fresh}
EXP=${EXP:-gate_a_qwen25_7b_seed2026}
RUN_SEED=${RUN_SEED:-2026}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
ROLLOUT_N=${ROLLOUT_N:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
N_GPUS=${N_GPUS:-2}
FSDP_SIZE=${FSDP_SIZE:-$N_GPUS}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}
REWARD_MANAGER=${REWARD_MANAGER:-naive}
FRESH_TOTAL_STEPS=${FRESH_TOTAL_STEPS:-2}
RESUME_TOTAL_STEPS=${RESUME_TOTAL_STEPS:-3}
RESUME_SOURCE_STEP=${RESUME_SOURCE_STEP:-2}
SAVE_FREQ=${SAVE_FREQ:-1}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}
OUT=$WORK_ROOT/logs/memory_agent/$EXP

case "$PHASE" in
  fresh)
    TOTAL_STEPS=$FRESH_TOTAL_STEPS
    RESUME_MODE=disable
    RESUME_ARGS=()
    if [[ -e "$OUT" ]] && [[ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Refusing to overwrite non-empty run: $OUT" >&2
      exit 44
    fi
    ;;
  resume)
    TOTAL_STEPS=$RESUME_TOTAL_STEPS
    RESUME_MODE=resume_path
    RESUME_FROM=${RESUME_FROM:-$OUT/global_step_$RESUME_SOURCE_STEP}
    [[ -d "$RESUME_FROM/actor" && -f "$RESUME_FROM/data.pt" ]] || {
      echo "Missing complete step-$RESUME_SOURCE_STEP checkpoint: $RESUME_FROM" >&2
      exit 45
    }
    [[ $(basename -- "$RESUME_FROM") == "global_step_$RESUME_SOURCE_STEP" ]] || {
      echo "Resume source basename must be global_step_$RESUME_SOURCE_STEP: $RESUME_FROM" >&2
      exit 45
    }
    RESUME_ARGS=(trainer.resume_from_path="$RESUME_FROM")
    ;;
  *) echo "PHASE must be fresh or resume" >&2; exit 46 ;;
esac

[[ $TOTAL_STEPS =~ ^[0-9]+$ && $RESUME_SOURCE_STEP =~ ^[0-9]+$ && $SAVE_FREQ =~ ^-?[0-9]+$ && $MAX_ACTOR_CKPT_TO_KEEP =~ ^[0-9]+$ ]] || {
  echo "Step/checkpoint controls must be integers: TOTAL_STEPS=$TOTAL_STEPS RESUME_SOURCE_STEP=$RESUME_SOURCE_STEP SAVE_FREQ=$SAVE_FREQ MAX_ACTOR_CKPT_TO_KEEP=$MAX_ACTOR_CKPT_TO_KEEP" >&2
  exit 53
}
[[ $TOTAL_STEPS -gt 0 && $SAVE_FREQ -ne 0 && $MAX_ACTOR_CKPT_TO_KEEP -gt 0 ]] || {
  echo "Invalid step/checkpoint controls: TOTAL_STEPS=$TOTAL_STEPS SAVE_FREQ=$SAVE_FREQ MAX_ACTOR_CKPT_TO_KEEP=$MAX_ACTOR_CKPT_TO_KEEP" >&2
  exit 54
}
if [[ $PHASE == resume && $TOTAL_STEPS -le $RESUME_SOURCE_STEP ]]; then
  echo "Resume total steps must exceed the source step: TOTAL_STEPS=$TOTAL_STEPS RESUME_SOURCE_STEP=$RESUME_SOURCE_STEP" >&2
  exit 55
fi

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

GLOBAL_ROLLOUT_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
GLOBAL_ROLLOUT_MINI_BATCH_SIZE=$((PPO_MINI_BATCH_SIZE * ROLLOUT_N))
[[ $TRAIN_BATCH_SIZE -gt 0 && $ROLLOUT_N -gt 0 && $PPO_MINI_BATCH_SIZE -gt 0 ]] || {
  echo "[GATE_A_BATCH_DIAG] batch sizes must be positive: TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE ROLLOUT_N=$ROLLOUT_N PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE" >&2
  exit 50
}
[[ $((TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE)) -eq 0 ]] || {
  echo "[GATE_A_BATCH_DIAG] invalid prompt mini-batch: TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE (PPO_MINI_BATCH_SIZE counts prompts)" >&2
  exit 51
}
[[ $((GLOBAL_ROLLOUT_MINI_BATCH_SIZE % N_GPUS)) -eq 0 ]] || {
  echo "[GATE_A_BATCH_DIAG] rollout-expanded mini-batch is not divisible by data-parallel world size: PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE ROLLOUT_N=$ROLLOUT_N global_rollout_mini_batch_size=$GLOBAL_ROLLOUT_MINI_BATCH_SIZE N_GPUS=$N_GPUS FSDP_SIZE=$FSDP_SIZE" >&2
  exit 52
}
echo "[GATE_A_BATCH_DIAG] configured global_batch_size=$GLOBAL_ROLLOUT_BATCH_SIZE prompt_mini_batch_size=$PPO_MINI_BATCH_SIZE global_rollout_mini_batch_size=$GLOBAL_ROLLOUT_MINI_BATCH_SIZE N_GPUS=$N_GPUS FSDP_SIZE=$FSDP_SIZE"

for path in "$PYTHON" "$MODEL/config.json" "$TRAIN" "$VAL"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 49; }
done

TRAINER_OVERRIDES=(
  recurrent.enable=memory
  recurrent.memory.config.chunk_size=5000
  recurrent.memory.config.max_chunks=8
  algorithm.adv_estimator=grpo
  algorithm.grpo_use_adv=False
  "actor_rollout_ref.rollout.n=$ROLLOUT_N"
  "+actor_rollout_ref.rollout.seed=$RUN_SEED"
  +actor_rollout_ref.rollout.trajectory_seed_mode=independent
  actor_rollout_ref.rollout.val_kwargs.n=2
  "trainer.logger=['console']"
  actor_rollout_ref.actor.optim.lr_warmup_steps=2
  actor_rollout_ref.actor.clip_ratio_high=0.20
  actor_rollout_ref.actor.entropy_coeff=0.000
  "data.train_files=$TRAIN"
  "data.val_files=$VAL"
  data.shuffle=False
  data.filter_overlong_prompts=True
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  +data.dataloader_num_workers=0
  data.truncation=center
  +data.context_key=context
  data.max_prompt_length=8192
  data.max_response_length=1024
  "reward_model.reward_manager=$REWARD_MANAGER"
  "custom_reward_function.path=$CODE/recurrent/research/hotpotqa_dense_reward.py"
  custom_reward_function.name=compute_score
  +custom_reward_function.reward_kwargs.f1_weight=0.95
  +custom_reward_function.reward_kwargs.grounded_box_bonus=0.05
  "actor_rollout_ref.model.path=$MODEL"
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.model.use_remove_padding=True
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  "actor_rollout_ref.actor.fsdp_config.fsdp_size=$FSDP_SIZE"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.load_format=dummy_dtensor
  actor_rollout_ref.rollout.temperature=1
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.rollout.max_num_batched_tokens=16384
  actor_rollout_ref.rollout.max_num_seqs=16
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0
  actor_rollout_ref.rollout.val_kwargs.top_p=0.7
  algorithm.kl_ctrl.kl_coef=0.001
  trainer.critic_warmup=0
  trainer.project_name=memagent_7b_serialization_credit
  "trainer.experiment_name=$EXP"
  trainer.val_before_train=False
  "trainer.n_gpus_per_node=$N_GPUS"
  trainer.nnodes=1
  "trainer.save_freq=$SAVE_FREQ"
  trainer.test_freq=-1
  trainer.total_epochs=30
  "trainer.total_training_steps=$TOTAL_STEPS"
  "trainer.resume_mode=$RESUME_MODE"
  "trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP"
  trainer.default_hdfs_dir=null
  "trainer.default_local_dir=$OUT"
  ray_init.num_cpus=64
)

# Bash 3.2 treats expansion of an empty array as an unbound variable under
# `set -u`.  Append the resume-only override explicitly so the fresh entry is
# portable across the local audit host and the H20 Linux runtime.
if [[ $PHASE == resume ]]; then
  TRAINER_OVERRIDES+=("${RESUME_ARGS[@]}")
fi

if [[ ${EMIT_TRAINER_OVERRIDES:-0} == 1 ]]; then
  "$PYTHON" -c 'import json,sys; print(json.dumps(sys.argv[1:], separators=(",", ":")))' \
    "${TRAINER_OVERRIDES[@]}" "$@"
  exit 0
fi

mkdir -p "$OUT" "$WORK_ROOT/logs/gate_a"
# Ray appends a long session/socket suffix. Keep the base below /tmp so the
# resulting AF_UNIX socket path stays under Linux's 107-byte limit.
RAY_TMP="/tmp/mga_${UID}_${PHASE}_$$"
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

"$PYTHON" -m verl.trainer.main_ppo "${TRAINER_OVERRIDES[@]}" "$@"

echo "Gate A phase=$PHASE finished; Ray temp was $RAY_TMP"

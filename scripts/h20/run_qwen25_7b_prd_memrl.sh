#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$SCRIPT_DIR/prd_memrl_common.sh"
prd_require_env
prd_paths

readonly ACTION=${1:-}
readonly E0_CERT=$PRD_CERT_ROOT/e0.json
readonly E1_CERT=$PRD_CERT_ROOT/e1.json
readonly PAPER_CERT=$PRD_CERT_ROOT/paper_review.json
readonly P0_CERT=$PRD_CERT_ROOT/p0.json

case "$ACTION" in
  e0)
    [[ ! -e $E0_CERT ]] || prd_die 'E0 certificate already exists; use a new RUN_ID'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_gate.py" e0 --output "$E0_CERT"
    ;;
  e1)
    [[ ${E1_ROWS:-} == /* && -f ${E1_ROWS:-} ]] || prd_die 'E1_ROWS must name frozen Original JSONL'
    [[ ! -e $E1_CERT ]] || prd_die 'E1 certificate already exists; use a new RUN_ID'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_gate.py" e1 --rows "$E1_ROWS" --output "$E1_CERT"
    ;;
  preflight)
    [[ ${PRD_PRIOR_MODEL:-} == /* ]] || prd_die 'PRD_PRIOR_MODEL must be explicit for P0'
    [[ ${PRD_BASE_MODEL:-} == /data/cw/memagent_work/models/Qwen2.5-7B-Instruct ]] || prd_die 'PRD_BASE_MODEL must be the canonical 7B path'
    [[ ${ORIGINAL_TRAINING_RESOLVED:-} == /data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json ]] || prd_die 'Original training resolved path mismatch'
    [[ ! -e $P0_CERT ]] || prd_die 'P0 certificate already exists; use a new RUN_ID'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_gate.py" preflight \
      --expected-commit "$EXPECTED_COMMIT" --gpu-pair "$GPU_PAIR" --e0 "$E0_CERT" \
      --paper-review "$PAPER_CERT" --prior-model "$PRD_PRIOR_MODEL" --base-model "$PRD_BASE_MODEL" \
      --original-training-resolved "$ORIGINAL_TRAINING_RESOLVED" --output "$P0_CERT"
    ;;
  bind)
    [[ ${BASELINE_CERT:-} == /* && -f ${BASELINE_CERT:-} ]] || prd_die 'BASELINE_CERT must name the read-only certified import'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_orchestrator.py" bind \
      --run-root "$PRD_RUN_ROOT" --run-id "$RUN_ID" --commit "$EXPECTED_COMMIT" \
      --gpu-pair "$GPU_PAIR" --baseline "$BASELINE_CERT" --p0 "$P0_CERT"
    ;;
  prepare-run)
    [[ ${CAPACITY_NATS:-} =~ ^[0-9]+([.][0-9]+)?$ ]] || prd_die 'CAPACITY_NATS must be explicit'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_orchestrator.py" stage full \
      --run-root "$PRD_RUN_ROOT" --run-id "$RUN_ID" --commit "$EXPECTED_COMMIT" --capacity "$CAPACITY_NATS"
    ;;
  prepare-continuation)
    [[ ${CAPACITY_NATS:-} =~ ^[0-9]+([.][0-9]+)?$ ]] || prd_die 'CAPACITY_NATS must be explicit'
    [[ ${RESUME_CHECKPOINT:-} == /* ]] || prd_die 'RESUME_CHECKPOINT must be explicit'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_orchestrator.py" stage continue \
      --run-root "$PRD_RUN_ROOT" --run-id "$RUN_ID" --commit "$EXPECTED_COMMIT" \
      --capacity "$CAPACITY_NATS" --resume "$RESUME_CHECKPOINT"
    ;;
  evaluate)
    [[ ${CAPACITY_NATS:-} =~ ^[0-9]+([.][0-9]+)?$ ]] || prd_die 'CAPACITY_NATS must be explicit'
    [[ ${EVAL_INPUT_TEMPLATE:-} == /* && ${EVAL_INPUT_TEMPLATE:-} == *'{anchor}'* ]] || prd_die 'EVAL_INPUT_TEMPLATE must be absolute and contain {anchor}'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_orchestrator.py" evaluate \
      --run-root "$PRD_RUN_ROOT" --capacity "$CAPACITY_NATS" --input-template "$EVAL_INPUT_TEMPLATE" \
      --anchors "${EVAL_ANCHORS:-5,10,15,20,25}"
    ;;
  produce-s128)
    [[ ${CAPACITY_NATS:-} =~ ^[0-9]+([.][0-9]+)?$ && ${ANCHOR:-} =~ ^(5|10|15|20|25)$ ]] || prd_die 'CAPACITY_NATS and ANCHOR are required'
    [[ ${VALIDATION_PARQUET:-} == /data/cw/memagent_work/datasets/hotpotqa/hotpotqa_dev.parquet ]] || prd_die 'wrong frozen validation path'
    [[ $(sha256sum "$VALIDATION_PARQUET" | awk '{print $1}') == 54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6 ]] || prd_die 'validation SHA drift'
    cid=c${CAPACITY_NATS%.*}; checkpoint=$PRD_RUN_ROOT/frontier/$cid/checkpoints/global_step_$ANCHOR
    [[ -f $checkpoint/prd_checkpoint.json && -f $PRD_RUN_ROOT/frontier/$cid/checkpoints/global_step_25/prd_checkpoint.json ]] || prd_die 'T25-complete checkpoint set required before S128'
    prd_acquire_gpu_locks; export CUDA_VISIBLE_DEVICES=$GPU_PAIR
    tmp=$PRD_RUN_ROOT/frontier/$cid/raw_terminal/.producer_anchor_$ANCHOR
    [[ ! -e $tmp ]] || prd_die 'raw producer temp already exists'; mkdir -p "$tmp/generated"
    export PYTHONNOUSERSITE=1 PYTHONPATH=$PRD_REPO CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
    "$PRD_PYTHON" -m verl.trainer.main_ppo \
      recurrent.enable=memory recurrent.memory.config.chunk_size=5000 recurrent.memory.config.max_chunks=8 \
      recurrent.memory.config.max_prompt_length=1024 recurrent.memory.config.max_memorization_length=1024 recurrent.memory.config.max_final_response_length=1024 \
      data.train_files="$VALIDATION_PARQUET" data.val_files="$VALIDATION_PARQUET" data.train_batch_size=2 data.shuffle=False \
      data.filter_overlong_prompts=True data.filter_overlong_prompts_workers=1 +data.dataloader_num_workers=0 +data.include_source_order_index=True \
      data.truncation=center +data.context_key=context +data.val_max_samples=128 data.max_prompt_length=8192 data.max_response_length=1024 \
      algorithm.adv_estimator=grpo algorithm.grpo_use_adv=False algorithm.use_kl_in_reward=False \
      actor_rollout_ref.model.path=/data/cw/memagent_work/models/Qwen2.5-7B-Instruct actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=2 actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
      actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
      actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=sync actor_rollout_ref.rollout.load_format=dummy_dtensor actor_rollout_ref.rollout.n=1 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.55 actor_rollout_ref.rollout.val_kwargs.n=1 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=False actor_rollout_ref.rollout.val_kwargs.temperature=0.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
      reward_model.reward_manager=naive trainer.logger='["console"]' trainer.project_name=prd_memrl_s128 trainer.experiment_name="${RUN_ID}_${cid}_a${ANCHOR}" \
      trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.val_before_train=True +trainer.val_only=True trainer.validation_data_dir="$tmp/generated" \
      trainer.save_freq=-1 trainer.test_freq=-1 trainer.total_epochs=1 trainer.total_training_steps=1 trainer.default_hdfs_dir=null trainer.default_local_dir="$tmp/no_checkpoint" \
      trainer.resume_mode=actor_only_eval trainer.resume_from_path="$checkpoint" ray_init.num_cpus=64
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/materialize_prd_s128_rows.py" --input "$tmp/generated/$ANCHOR.jsonl" \
      --output "$PRD_RUN_ROOT/frontier/$cid/raw_terminal/anchor_$ANCHOR.jsonl" \
      --stable-resolved /data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json \
      --validation-parquet "$VALIDATION_PARQUET" --checkpoint-metadata "$checkpoint/prd_checkpoint.json" \
      --run-id "$RUN_ID" --git-commit "$EXPECTED_COMMIT" --frontier-id "$cid" --global-step "$ANCHOR"
    ;;
  final-audit)
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_orchestrator.py" audit \
      --run-root "$PRD_RUN_ROOT" --ledger "$PRD_LEDGER" --output "$PRD_CERT_ROOT/final_audit.json"
    ;;
  train-t25|recover-from-t5)
    [[ ${CAPACITY_NATS:-} =~ ^[0-9]+([.][0-9]+)?$ ]] || prd_die 'CAPACITY_NATS must be explicit'
    case ",$CAPACITY_NATS," in ,128.0,|,256.0,|,512.0,) ;; *) prd_die 'CAPACITY_NATS is not in the frozen frontier' ;; esac
    prd_verify_gate "$P0_CERT" PRD_P0_PASS
    if [[ $ACTION == recover-from-t5 ]]; then
      cid=c${CAPACITY_NATS%.*}
      [[ ${RESUME_CHECKPOINT:-} == "$PRD_RUN_ROOT/frontier/$cid/checkpoints/global_step_5" ]] || prd_die 'continuation must resume this run/capacity exact step 5'
    else
      [[ -z ${RESUME_CHECKPOINT:-} ]] || prd_die 'Method-T25 must start fresh; resume/warm-start forbidden'
    fi
    prd_acquire_gpu_locks
    export CUDA_VISIBLE_DEVICES=$GPU_PAIR
    [[ ${PRD_PRIOR_MODEL:-} == /* ]] || prd_die 'PRD_PRIOR_MODEL must be explicit'
    cid=c${CAPACITY_NATS%.*}
    stage=full
    [[ $ACTION == recover-from-t5 ]] && stage=continue
    "$PRD_PYTHON" - "$PRD_RUN_ROOT/resolved_run.json" "$PRD_RUN_ROOT/frontier/$cid/launch_${stage}.json" "$RUN_ID" "$EXPECTED_COMMIT" "$GPU_PAIR" "$CAPACITY_NATS" "$PRD_PRIOR_MODEL" "$PRD_BASE_MODEL" <<'PY'
import hashlib,json,pathlib,sys
run=json.load(open(sys.argv[1])); launch=json.load(open(sys.argv[2]))
assert run["run_id"]==sys.argv[3] and run["git_commit"]==sys.argv[4]
assert run["gpu_pair"]==sys.argv[5] and launch["gpu_pair"]==sys.argv[5]
assert float(launch["capacity_nats"])==float(sys.argv[6])
assert launch["frontier_id"]=="c"+str(int(float(sys.argv[6])))
prior=pathlib.Path(sys.argv[7]).resolve(); frozen=run["prior_model"]
assert str(prior)==frozen["path"] and launch["prior_model"]==frozen
base=pathlib.Path(sys.argv[8]).resolve(); assert str(base)==run["base_model"]["path"] and launch["base_model"]==run["base_model"]
for root,model in ((prior,frozen),(base,run["base_model"])):
 for item in model["files"]:
  p=root/item["path"]; assert p.is_file() and not p.is_symlink() and p.stat().st_size==item["size"]
  assert hashlib.sha256(p.read_bytes()).hexdigest()==item["sha256"]
PY
    phase=fresh
    [[ $ACTION == recover-from-t5 ]] && phase=resume
    exec env WORK_ROOT="$WORK_ROOT" CODE="$PRD_REPO" PYTHON="$PRD_PYTHON" \
      PHASE="$phase" EXP="${RUN_ID}_${cid}" RUN_SEED=2026 \
      TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 N_GPUS=2 FSDP_SIZE=2 \
      REWARD_MANAGER=naive GPU_MEMORY_UTILIZATION=0.55 \
      FRESH_TOTAL_STEPS=25 RESUME_TOTAL_STEPS=25 RESUME_SOURCE_STEP=5 \
      RESUME_FROM="${RESUME_CHECKPOINT:-}" SAVE_FREQ=5 MAX_ACTOR_CKPT_TO_KEEP=5 \
      OUTPUT_ROOT="$PRD_RUN_ROOT/frontier/$cid/checkpoints" \
      MODEL_PATH="$PRD_BASE_MODEL" \
      PRD_MEMRL_ENABLE=1 PRD_MEMRL_CAPACITY="$CAPACITY_NATS" \
      PRD_RUN_ID="$RUN_ID" PRD_FRONTIER_ID="$cid" PRD_GIT_COMMIT="$EXPECTED_COMMIT" \
      PRD_EXECUTION_LEDGER="$PRD_LEDGER" \
      PRD_PRIOR_MODEL="$PRD_PRIOR_MODEL" \
      bash "$PRD_REPO/experiments/7b_gate_a/run_gate_a.sh"
    ;;
  *) prd_die 'unknown action (expected e0/e1/preflight/bind/prepare-run/prepare-continuation/produce-s128/evaluate/final-audit/train-t25/recover-from-t5)' ;;
esac

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/cosi_common.sh"
unset CORAL_E1_CAPTURE_DIR
cosi_checkout_guard
cosi_acquire_gpu_locks
readonly PYTHON=$MEMAGENT_COSI_WORK_ROOT/.venv/bin/python
readonly RUN_ID=${MEMAGENT_COSI_RUN_ID:-coral_seed2026_primary_v1}
readonly EXP=qwen25_7b_coral_fresh_t25_seed2026_${RUN_ID}
readonly RUN_ROOT=$MEMAGENT_COSI_WORK_ROOT/logs/coral/$RUN_ID
readonly OUTPUT=$MEMAGENT_COSI_WORK_ROOT/logs/memory_agent/$EXP
readonly HEALTH=$RUN_ROOT/certificates/t5_health.json
[[ -d $OUTPUT/global_step_5/actor && -f $OUTPUT/global_step_5/data.pt ]] || { echo CORAL_NO_GO:missing_t5_checkpoint >&2; exit 80; }
mkdir -p "$RUN_ROOT/certificates"
"$PYTHON" "$MEMAGENT_COSI_REPO_DIR/tools/h20/audit_coral_t5_health.py" \
  --run-root "$RUN_ROOT" --checkpoint "$OUTPUT/global_step_5" \
  --exact-boundary --output "$HEALTH"
"$PYTHON" "$MEMAGENT_COSI_REPO_DIR/tools/h20/preflight_qwen25_7b_cosi.py" --manifest "$MEMAGENT_COSI_REPO_DIR/manifests/h20/qwen25_7b_cosi_seed2026.json" --stage continue --write-certificate
export GATE_A_FROZEN_AUDIT=1 GATE_A_EXECUTION_LEDGER=$RUN_ROOT/gate_a_execution_ledger.jsonl
export GATE_A_EXPERIMENT_NAME=$EXP GATE_A_GIT_COMMIT=$MEMAGENT_COSI_EXPECTED_COMMIT
export GATE_A_RUN_ID; GATE_A_RUN_ID=$(printf '%s' "$MEMAGENT_COSI_EXPECTED_COMMIT:$RUN_ID" | shasum -a 256 | cut -c1-32)
export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
export GATE_A_WEIGHT_DIGEST_SAMPLES=256 CORAL_EXECUTION_LEDGER=$RUN_ROOT/coral_execution_ledger.jsonl
export WORK_ROOT=$MEMAGENT_COSI_WORK_ROOT CODE=$MEMAGENT_COSI_REPO_DIR PYTHON
export MODEL=$MEMAGENT_COSI_WORK_ROOT/models/Qwen2.5-7B-Instruct TRAIN=$MEMAGENT_COSI_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet VAL=$MEMAGENT_COSI_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet
export PHASE=resume EXP RESUME_FROM=$OUTPUT/global_step_5 RESUME_SOURCE_STEP=5 RESUME_TOTAL_STEPS=25
export RUN_SEED=2026 TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 N_GPUS=2 FSDP_SIZE=2 SAVE_FREQ=1 MAX_ACTOR_CKPT_TO_KEEP=30 REWARD_MANAGER=naive GPU_MEMORY_UTILIZATION=0.55
cd "$MEMAGENT_COSI_REPO_DIR"
bash experiments/7b_gate_a/run_gate_a.sh +algorithm.coral.enabled=true +algorithm.coral.active_from_update=1 +algorithm.coral.schedule=odd_writer_even_terminal_answer_v2 +algorithm.coral.role_partition=nonfinal_memory_writer_vs_final_answer +algorithm.coral.require_recurrent=true +algorithm.coral.require_grpo=true +algorithm.coral.require_gate_a_sync=true trainer.project_name=memagent_coral
echo "CORAL_T25_TRAINING_COMPLETE:$OUTPUT/global_step_25"

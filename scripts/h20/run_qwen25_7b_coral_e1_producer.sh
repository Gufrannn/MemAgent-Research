#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/cosi_common.sh"
cosi_checkout_guard
cosi_acquire_gpu_locks
readonly PYTHON=$MEMAGENT_COSI_WORK_ROOT/.venv/bin/python
readonly RUN_ID=${MEMAGENT_COSI_E1_RUN_ID:-coral_e1_seed2026_v11}
[[ $RUN_ID =~ ^[a-z0-9][a-z0-9_-]{7,79}$ ]] || {
  echo CORAL_E1_NO_GO:run_id >&2; exit 78;
}
case "$RUN_ID" in
  coral_e1_seed2026_v3|coral_e1_seed2026_v4|coral_e1_seed2026_v5|coral_e1_seed2026_v6|coral_e1_seed2026_v7|coral_e1_seed2026_v8|coral_e1_seed2026_v9|coral_e1_seed2026_v10)
    echo CORAL_E1_NO_GO:retired_evidence_run_id >&2; exit 78 ;;
esac
readonly EXP=qwen25_7b_coral_e1_actual_loss_${RUN_ID}
readonly RUN_ROOT=$MEMAGENT_COSI_WORK_ROOT/logs/coral_e1/$RUN_ID
readonly OUTPUT=$MEMAGENT_COSI_WORK_ROOT/logs/memory_agent/$EXP
readonly CAPTURE=$RUN_ROOT/actual_loss_receipts
readonly CERT=$MEMAGENT_COSI_WORK_ROOT/logs/cosi_preflight/certificates
[[ ! -e $RUN_ROOT && ! -e $OUTPUT ]] || {
  echo CORAL_E1_NO_GO:append_only_output_exists >&2; exit 79;
}
[[ ! -e $CERT/coral_e1_evidence.json && ! -e $CERT/coral_e1_final_report.json ]] || {
  echo CORAL_E1_NO_GO:certificate_exists >&2; exit 80;
}
"$PYTHON" "$MEMAGENT_COSI_REPO_DIR/tools/h20/preflight_qwen25_7b_cosi.py" \
  --manifest "$MEMAGENT_COSI_REPO_DIR/manifests/h20/qwen25_7b_cosi_seed2026.json" \
  --stage research
mkdir -p "$CAPTURE" "$CERT"
"$PYTHON" "$MEMAGENT_COSI_REPO_DIR/tools/h20/coral_dataproto_clone_oracle.py" \
  --output "$RUN_ROOT/coral_dataproto_clone_oracle.json"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$MEMAGENT_COSI_REPO_DIR/tools/h20/coral_e1_fsdp_sketch_oracle.py" \
  --output "$RUN_ROOT/coral_e1_fsdp_sketch_oracle.json"
export GATE_A_FROZEN_AUDIT=1
export GATE_A_EXECUTION_LEDGER=$RUN_ROOT/gate_a_execution_ledger.jsonl
export GATE_A_EXPERIMENT_NAME=$EXP GATE_A_GIT_COMMIT=$MEMAGENT_COSI_EXPECTED_COMMIT
export GATE_A_RUN_ID
GATE_A_RUN_ID=$(printf '%s' "$MEMAGENT_COSI_EXPECTED_COMMIT:$RUN_ID" | shasum -a 256 | cut -c1-32)
export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
export GATE_A_WEIGHT_DIGEST_SAMPLES=256
export CORAL_EXECUTION_LEDGER=$RUN_ROOT/coral_execution_ledger.jsonl
export CORAL_E1_CAPTURE_DIR=$CAPTURE
export WORK_ROOT=$MEMAGENT_COSI_WORK_ROOT CODE=$MEMAGENT_COSI_REPO_DIR PYTHON
export MODEL=$MEMAGENT_COSI_WORK_ROOT/models/Qwen2.5-7B-Instruct
export TRAIN=$MEMAGENT_COSI_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet
export VAL=$MEMAGENT_COSI_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet
export PHASE=fresh EXP RUN_SEED=2026 TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4
export N_GPUS=2 FSDP_SIZE=2 FRESH_TOTAL_STEPS=15 SAVE_FREQ=1 MAX_ACTOR_CKPT_TO_KEEP=30
export REWARD_MANAGER=naive GPU_MEMORY_UTILIZATION=0.55
cd "$MEMAGENT_COSI_REPO_DIR"
bash experiments/7b_gate_a/run_gate_a.sh \
  +algorithm.coral.enabled=true \
  +algorithm.coral.active_from_update=1 \
  +algorithm.coral.schedule=odd_writer_even_terminal_answer_v2 \
  +algorithm.coral.role_partition=nonfinal_memory_writer_vs_final_answer \
  +algorithm.coral.require_recurrent=true \
  +algorithm.coral.require_grpo=true \
  +algorithm.coral.require_gate_a_sync=true \
  trainer.project_name=memagent_coral_e1
unset CORAL_E1_CAPTURE_DIR
"$PYTHON" tools/h20/seal_coral_e1.py \
  --capture-root "$CAPTURE" \
  --training-output "$OUTPUT" \
  --base-model "$MODEL" \
  --dataproto-clone-oracle "$RUN_ROOT/coral_dataproto_clone_oracle.json" \
  --sketch-oracle "$RUN_ROOT/coral_e1_fsdp_sketch_oracle.json" \
  --gate-a-ledger "$RUN_ROOT/gate_a_execution_ledger.jsonl" \
  --expected-commit "$MEMAGENT_COSI_EXPECTED_COMMIT" \
  --output "$CERT/coral_e1_evidence.json"
"$PYTHON" tools/h20/audit_coral_e1.py \
  --evidence "$CERT/coral_e1_evidence.json" \
  --output "$CERT/coral_e1_final_report.json"
echo "CORAL_E1_ACTUAL_PRODUCER_COMPLETE:$CERT/coral_e1_final_report.json"

#!/usr/bin/env bash
set -euo pipefail

for name in WORK_ROOT REPO_DIR EXPECTED_COMMIT GPU_PAIR RUN_ID; do
  value_name=MEMAGENT_MIC_$name
  [[ -n ${!value_name:-} ]] || { echo "MIC_NO_GO: set $value_name" >&2; exit 66; }
done
[[ $MEMAGENT_MIC_WORK_ROOT == /* && $MEMAGENT_MIC_REPO_DIR == /* ]] || {
  echo 'MIC_NO_GO: work root and repo must be absolute' >&2; exit 67;
}
[[ $MEMAGENT_MIC_EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ ]] || {
  echo 'MIC_NO_GO: expected commit must be full SHA' >&2; exit 68;
}
[[ $MEMAGENT_MIC_GPU_PAIR =~ ^[0-9]+,[0-9]+$ ]] || {
  echo 'MIC_NO_GO: GPU pair must be explicit N,M' >&2; exit 69;
}
IFS=, read -r MIC_GPU_A MIC_GPU_B <<<"$MEMAGENT_MIC_GPU_PAIR"
[[ $MIC_GPU_A -lt $MIC_GPU_B ]] || { echo 'MIC_NO_GO: GPU pair must be canonical ascending' >&2; exit 70; }

readonly MIC_PYTHON=$MEMAGENT_MIC_WORK_ROOT/.venv/bin/python
readonly MIC_MANIFEST=$MEMAGENT_MIC_REPO_DIR/manifests/h20/qwen25_7b_mic_seed2026.json
readonly MIC_ROOT=$MEMAGENT_MIC_WORK_ROOT/logs/mic_frozen_20260822/$MEMAGENT_MIC_RUN_ID
readonly MIC_CERT=$MIC_ROOT/certificates
readonly MIC_P0=$MIC_CERT/p0.json
readonly MIC_E0=$MIC_CERT/e0.json
readonly MIC_E1=$MIC_CERT/e1.json
readonly MIC_BASELINE=$MIC_CERT/baseline_import.json
readonly MIC_BASELINE_ATTEMPTS=$MIC_ROOT/baseline_materialization_attempts
readonly MIC_CHECKPOINT_AUTHORITY=$MEMAGENT_MIC_REPO_DIR/manifests/h20/qwen25_7b_mic_checkpoint_authority.json
readonly MIC_CURVE_AUTHORITY=$MEMAGENT_MIC_REPO_DIR/manifests/h20/qwen25_7b_mic_original_curve_authority.json
readonly MIC_CHECKPOINT_AUTHORITY_CERT=$MIC_CERT/checkpoint_authority.json
readonly MIC_CURVE_RESOLVED=$MEMAGENT_MIC_WORK_ROOT/logs/s128_original_all_anchor_frozen_20260821/certificates/p0_resolved_manifest.json
readonly MIC_PAPER_REVIEW=$MEMAGENT_MIC_REPO_DIR/docs/papers/mic_release_review.json
readonly MIC_LEDGER=$MIC_ROOT/mic_execution_ledger.jsonl
readonly MIC_WEIGHT_LEDGER=$MIC_ROOT/mic_weight_sync_ledger.jsonl
readonly MIC_CRITIC_ROOT=$MIC_ROOT/critic_checkpoints
readonly MIC_EXPERIMENT=qwen25_7b_h20_mic_${MEMAGENT_MIC_RUN_ID}_seed2026
readonly MIC_OUTPUT=$MEMAGENT_MIC_WORK_ROOT/logs/memory_agent/$MIC_EXPERIMENT

mic_require_checkout() {
  [[ $(cd "$MEMAGENT_MIC_REPO_DIR" && git branch --show-current) == h20/qwen25-7b-mic-t25-frozen-20260822 ]] || {
    echo 'MIC_NO_GO: wrong branch' >&2; exit 71;
  }
  [[ -z $(cd "$MEMAGENT_MIC_REPO_DIR" && git status --porcelain) ]] || {
    echo 'MIC_NO_GO: dirty worktree' >&2; exit 72;
  }
  [[ $(cd "$MEMAGENT_MIC_REPO_DIR" && git rev-parse HEAD) == "$MEMAGENT_MIC_EXPECTED_COMMIT" ]] || {
    echo 'MIC_NO_GO: commit mismatch' >&2; exit 73;
  }
}

mic_acquire_gpu_locks() {
  command -v flock >/dev/null || { echo 'MIC_NO_GO: flock missing' >&2; exit 74; }
  mkdir -p "$MEMAGENT_MIC_WORK_ROOT/locks"
  local lock_a=$MEMAGENT_MIC_WORK_ROOT/locks/memagent_h20_gpu_${MIC_GPU_A}.lock
  local lock_b=$MEMAGENT_MIC_WORK_ROOT/locks/memagent_h20_gpu_${MIC_GPU_B}.lock
  exec 8>"$lock_a"; flock -n 8 || { echo "MIC_NO_GO: lock conflict $lock_a" >&2; exit 75; }
  exec 9>"$lock_b"; flock -n 9 || { echo "MIC_NO_GO: lock conflict $lock_b" >&2; exit 75; }
}

mic_require_idle() {
  command -v nvidia-smi >/dev/null || { echo 'MIC_NO_GO: nvidia-smi missing' >&2; exit 76; }
  local pids
  pids=$(nvidia-smi -i "$MEMAGENT_MIC_GPU_PAIR" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${pids//[[:space:]]/} ]] || {
    echo "MIC_NO_GO: selected GPUs occupied; no process changed: $pids" >&2; exit 77;
  }
}

mic_require_gate() {
  local path=$1 decision=$2
  "$MIC_PYTHON" -c 'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("status")=="PASS" and x.get("decision")==sys.argv[2] else 1)' \
    "$path" "$decision" || { echo "MIC_NO_GO: missing gate $decision" >&2; exit 78; }
}

mic_require_training_gates() {
  mic_require_gate "$MIC_P0" MIC_P0_PASS
  mic_require_gate "$MIC_E0" MIC_E0_PASS
  mic_require_gate "$MIC_PAPER_REVIEW" MIC_PAPER_REVIEW_GO
}

mic_next_eval_attempt() {
  local step=$1 index=1 candidate
  local container=$MIC_ROOT/eval_t${step}_attempts
  mkdir -p "$container"
  while :; do
    candidate=$(printf '%s/attempt_%04d' "$container" "$index")
    if mkdir "$candidate" 2>/dev/null; then
      printf '%s\n' "$candidate"
      return 0
    fi
    index=$((index + 1))
  done
}

mic_next_baseline_attempt() {
  local index=1 candidate
  mkdir -p "$MIC_BASELINE_ATTEMPTS"
  while :; do
    candidate=$(printf '%s/attempt_%04d' "$MIC_BASELINE_ATTEMPTS" "$index")
    if mkdir "$candidate" 2>/dev/null; then
      printf '%s\n' "$candidate"
      return 0
    fi
    index=$((index + 1))
  done
}

mic_export_training() {
  unset MEMAGENT_MIC_EVAL_STEP MEMAGENT_MIC_EVAL_DIR \
    MEMAGENT_MIC_EVAL_IDENTITY_PATH MEMAGENT_MIC_EVAL_IDENTITY_SHA256 \
    MEMAGENT_MIC_EVAL_SUMMARY_PATH MEMAGENT_MIC_EVAL_GENERATION_PATH \
    MEMAGENT_MIC_EVAL_TRAINING_AUDIT_SHA256 \
    MEMAGENT_MIC_EVAL_ORIGINAL_PROTOCOL_SHA256 \
    MEMAGENT_MIC_EVAL_ORIGINAL_REWARD_CODE_SHA256
  export CUDA_VISIBLE_DEVICES=$MEMAGENT_MIC_GPU_PAIR
  export MEMAGENT_MIC_REQUIRED=1
  export MEMAGENT_MIC_ENABLE=1
  export MEMAGENT_MIC_LEDGER_PATH=$MIC_LEDGER
  export MEMAGENT_MIC_CRITIC_ROOT=$MIC_CRITIC_ROOT
  export MEMAGENT_MIC_EXPECTED_COMMIT
  export GATE_A_FROZEN_AUDIT=1
  export GATE_A_EXECUTION_LEDGER=$MIC_WEIGHT_LEDGER
  export GATE_A_EXPERIMENT_NAME=$MIC_EXPERIMENT
  export GATE_A_GIT_COMMIT=$MEMAGENT_MIC_EXPECTED_COMMIT
  export GATE_A_RUN_ID=$MEMAGENT_MIC_RUN_ID
  export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
  export GATE_A_WEIGHT_DIGEST_SAMPLES=256
}

mic_export_evaluation() {
  export CUDA_VISIBLE_DEVICES=$MEMAGENT_MIC_GPU_PAIR
  export MEMAGENT_MIC_REQUIRED=1
  export MEMAGENT_MIC_ENABLE=1
  export MEMAGENT_MIC_EXPECTED_COMMIT
  export MEMAGENT_MIC_RUN_ID
  export GATE_A_FROZEN_AUDIT=0
  unset MEMAGENT_MIC_LEDGER_PATH MEMAGENT_MIC_CRITIC_ROOT \
    GATE_A_EXECUTION_LEDGER GATE_A_EXPERIMENT_NAME GATE_A_GIT_COMMIT GATE_A_RUN_ID
  export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
  export GATE_A_WEIGHT_DIGEST_SAMPLES=256
}

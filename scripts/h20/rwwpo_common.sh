#!/usr/bin/env bash
set -euo pipefail
[[ -n ${RWWPO_WORK_ROOT:-} && $RWWPO_WORK_ROOT == /* ]] || { echo 'RWWPO_NO_GO:set absolute RWWPO_WORK_ROOT' >&2; exit 60; }
[[ -n ${RWWPO_REPO_DIR:-} && $RWWPO_REPO_DIR == /* ]] || { echo 'RWWPO_NO_GO:set absolute RWWPO_REPO_DIR' >&2; exit 61; }
[[ ${RWWPO_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || { echo 'RWWPO_NO_GO:set exact RWWPO_EXPECTED_COMMIT' >&2; exit 62; }
[[ ${GPU_PAIR:-} =~ ^[0-9]+,[0-9]+$ ]] || { echo 'RWWPO_NO_GO:set explicit GPU_PAIR=N,M' >&2; exit 63; }
IFS=, read -r RWWPO_GPU_A RWWPO_GPU_B <<< "$GPU_PAIR"
(( RWWPO_GPU_A < RWWPO_GPU_B )) || { echo 'RWWPO_NO_GO:GPU_PAIR must be distinct canonical ascending' >&2; exit 64; }
[[ ${RWWPO_RUN_ID:-} =~ ^[a-z0-9][a-z0-9_-]{5,63}$ ]] || { echo 'RWWPO_NO_GO:set semantic RWWPO_RUN_ID (not random UUID)' >&2; exit 65; }
[[ -f ${RWWPO_ORIGINAL_RESOLVED_MANIFEST:-} && ${RWWPO_ORIGINAL_RESOLVED_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || { echo 'RWWPO_NO_GO:bind accepted Original resolved manifest path and SHA256' >&2; exit 59; }
readonly RWWPO_EXPECTED_BRANCH=${RWWPO_EXPECTED_BRANCH:-h20/qwen25-7b-rwwpo-t25-frozen-20260822}
readonly RWWPO_MANIFEST=${RWWPO_MANIFEST:-$RWWPO_REPO_DIR/manifests/h20/qwen25_7b_rwwpo_seed2026.json}
readonly RWWPO_OBJECTIVE_VARIANT=${RWWPO_OBJECTIVE_VARIANT:-whole_prefix}
readonly RWWPO_CONTROLLER_VARIANT=${RWWPO_CONTROLLER_VARIANT:-hard_rollback}
readonly RWWPO_CERT_ROOT=$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates
readonly RWWPO_E0=$RWWPO_CERT_ROOT/e0.json
readonly RWWPO_E1=$RWWPO_CERT_ROOT/e1.json
readonly RWWPO_BASELINE=$RWWPO_CERT_ROOT/baseline_import.json
readonly RWWPO_EXPERIMENT=qwen25_7b_rwwpo_${RWWPO_OBJECTIVE_VARIANT}_${RWWPO_CONTROLLER_VARIANT}_seed2026_${RWWPO_RUN_ID}
readonly RWWPO_OUTPUT=$RWWPO_WORK_ROOT/logs/memory_agent/$RWWPO_EXPERIMENT
readonly RWWPO_LEDGER_DIR=$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/actual_loss
readonly RWWPO_EXECUTION_LEDGER=$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/execution.jsonl
readonly RWWPO_LOG=$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/train.log
readonly RWWPO_PYTHON=$RWWPO_WORK_ROOT/.venv/bin/python

rwwpo_require_checkout() {
  [[ $(cd "$RWWPO_REPO_DIR" && git rev-parse HEAD) == "$RWWPO_EXPECTED_COMMIT" ]] || { echo 'RWWPO_NO_GO:commit drift' >&2; exit 66; }
  [[ $(cd "$RWWPO_REPO_DIR" && git branch --show-current) == "$RWWPO_EXPECTED_BRANCH" ]] || { echo 'RWWPO_NO_GO:branch drift' >&2; exit 67; }
  [[ -z $(cd "$RWWPO_REPO_DIR" && git status --porcelain) ]] || { echo 'RWWPO_NO_GO:dirty tree' >&2; exit 68; }
}
rwwpo_acquire_gpu_locks() {
  command -v flock >/dev/null || { echo 'RWWPO_NO_GO:flock unavailable' >&2; exit 69; }
  mkdir -p "$RWWPO_WORK_ROOT/locks"
  exec 8>"$RWWPO_WORK_ROOT/locks/memagent_h20_gpu_${RWWPO_GPU_A}.lock"
  flock -n 8 || { echo 'RWWPO_NO_GO:first GPU lock conflict; no process changed' >&2; exit 70; }
  exec 9>"$RWWPO_WORK_ROOT/locks/memagent_h20_gpu_${RWWPO_GPU_B}.lock"
  flock -n 9 || { echo 'RWWPO_NO_GO:second GPU lock conflict; no process changed' >&2; exit 71; }
}
rwwpo_require_idle() {
  command -v nvidia-smi >/dev/null || { echo 'RWWPO_NO_GO:nvidia-smi unavailable' >&2; exit 72; }
  local pids
  pids=$(nvidia-smi -i "$GPU_PAIR" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${pids//[[:space:]]/} ]] || { echo "RWWPO_NO_GO:GPU occupied; no process changed: $pids" >&2; exit 73; }
}
rwwpo_export_runtime() {
  export CUDA_VISIBLE_DEVICES=$GPU_PAIR RWWPO_ENABLE=1 RWWPO_LEDGER_DIR RWWPO_Q_MIN=0.5
  export RWWPO_OBJECTIVE_VARIANT RWWPO_CONTROLLER_VARIANT
  export GATE_A_FROZEN_AUDIT=1 GATE_A_EXECUTION_LEDGER=$RWWPO_EXECUTION_LEDGER
  export GATE_A_EXPERIMENT_NAME=$RWWPO_EXPERIMENT GATE_A_GIT_COMMIT=$RWWPO_EXPECTED_COMMIT
  export GATE_A_RUN_ID
  GATE_A_RUN_ID=$(printf '%s' "$RWWPO_EXPECTED_COMMIT:$RWWPO_RUN_ID" | sha256sum | cut -c1-32)
  export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
  export GATE_A_WEIGHT_DIGEST_SAMPLES=256
}

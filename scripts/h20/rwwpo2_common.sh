#!/usr/bin/env bash
set -euo pipefail

[[ -n ${RWWPO_WORK_ROOT:-} && $RWWPO_WORK_ROOT == /* ]] || {
  echo 'RWWPO2_NO_GO:set absolute RWWPO_WORK_ROOT' >&2; exit 60;
}
[[ -n ${RWWPO_REPO_DIR:-} && $RWWPO_REPO_DIR == /* ]] || {
  echo 'RWWPO2_NO_GO:set absolute RWWPO_REPO_DIR' >&2; exit 61;
}
[[ ${RWWPO_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'RWWPO2_NO_GO:set exact RWWPO_EXPECTED_COMMIT' >&2; exit 62;
}
[[ ${GPU_PAIR:-} =~ ^[0-9]+,[0-9]+$ ]] || {
  echo 'RWWPO2_NO_GO:set explicit GPU_PAIR=N,M' >&2; exit 63;
}
IFS=, read -r RWWPO_GPU_A RWWPO_GPU_B <<< "$GPU_PAIR"
(( RWWPO_GPU_A < RWWPO_GPU_B )) || {
  echo 'RWWPO2_NO_GO:GPU_PAIR must be distinct canonical ascending' >&2; exit 64;
}
[[ ${RWWPO_RUN_ID:-} =~ ^[a-z0-9][a-z0-9_-]{5,63}$ ]] || {
  echo 'RWWPO2_NO_GO:set semantic one-use RWWPO_RUN_ID' >&2; exit 65;
}
[[ ${RWWPO_CELL:-} =~ ^[ABCDE]$ ]] || {
  echo 'RWWPO2_NO_GO:RWWPO_CELL must be A|B|C|D|E' >&2; exit 66;
}
[[ ${RWWPO_EXPERIMENT_SEED:-} =~ ^20(2[6-9]|3[0-3])$ ]] || {
  echo 'RWWPO2_NO_GO:seed must be preregistered 2026..2033' >&2; exit 67;
}
[[ ${RWWPO_PHASE:-} == fresh || ${RWWPO_PHASE:-} == resume ]] || {
  echo 'RWWPO2_NO_GO:RWWPO_PHASE=fresh|resume' >&2; exit 68;
}
[[ ${RWWPO_TARGET_ROUND:-} =~ ^(50|400)$ ]] || {
  echo 'RWWPO2_NO_GO:RWWPO_TARGET_ROUND=50|400' >&2; exit 69;
}
[[ -f ${RWWPO_RESOLVED_CONTRACT:-} && ${RWWPO_RESOLVED_CONTRACT_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || {
  echo 'RWWPO2_NO_GO:bind resolved numeric contract path/SHA' >&2; exit 70;
}
[[ -f ${RWWPO_E0:-} && -f ${RWWPO_DATA_BOUNDARY_AUDIT:-} \
   && -f ${RWWPO_BASE_PROTOCOL_AUDIT:-} ]] || {
  echo 'RWWPO2_NO_GO:bind E0, data-boundary, and base-protocol receipts' >&2; exit 71;
}
[[ -f ${RWWPO_ORIGINAL_RESOLVED_MANIFEST:-} && ${RWWPO_ORIGINAL_RESOLVED_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || {
  echo 'RWWPO2_NO_GO:bind accepted Original training resolved path/SHA' >&2; exit 72;
}

readonly RWWPO_EXPECTED_BRANCH=h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822
readonly RWWPO_MANIFEST=$RWWPO_REPO_DIR/manifests/h20/qwen25_7b_rwwpo2_r400_k2_seed2026.json
case "$RWWPO_CELL" in
  D) readonly RWWPO_OBJECTIVE_VARIANT=original_tokenwise RWWPO_CONTROLLER_VARIANT=none ;;
  C) readonly RWWPO_OBJECTIVE_VARIANT=original_tokenwise RWWPO_CONTROLLER_VARIANT=feasible_backtracking ;;
  E) readonly RWWPO_OBJECTIVE_VARIANT=per_write_joint RWWPO_CONTROLLER_VARIANT=feasible_backtracking ;;
  B) readonly RWWPO_OBJECTIVE_VARIANT=whole_prefix RWWPO_CONTROLLER_VARIANT=feasible_backtracking ;;
  A) readonly RWWPO_OBJECTIVE_VARIANT=whole_prefix RWWPO_CONTROLLER_VARIANT=hard_rollback ;;
esac
if [[ $RWWPO_CELL == A && $RWWPO_TARGET_ROUND != 50 ]]; then
  echo 'RWWPO2_NO_GO:cell A is R50-only' >&2; exit 73
fi

readonly RWWPO_ATTEMPT_ROOT=$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID
readonly RWWPO_CERT_ROOT=$RWWPO_ATTEMPT_ROOT/certificates
readonly RWWPO_LEDGER_DIR=$RWWPO_ATTEMPT_ROOT/actual_loss
readonly RWWPO_EXECUTION_LEDGER=$RWWPO_ATTEMPT_ROOT/execution.jsonl
readonly RWWPO_LOG=$RWWPO_ATTEMPT_ROOT/train.log
readonly RWWPO_PREFLIGHT=$RWWPO_CERT_ROOT/p0_preflight.json
readonly RWWPO_EXPERIMENT=rwwpo2_${RWWPO_CELL}_seed${RWWPO_EXPERIMENT_SEED}_${RWWPO_RUN_ID}
readonly RWWPO_OUTPUT=$RWWPO_WORK_ROOT/logs/memory_agent/$RWWPO_EXPERIMENT
readonly RWWPO_PYTHON=$RWWPO_WORK_ROOT/.venv/bin/python

rwwpo2_require_checkout() {
  [[ $(cd "$RWWPO_REPO_DIR" && git rev-parse HEAD) == "$RWWPO_EXPECTED_COMMIT" ]] || {
    echo 'RWWPO2_NO_GO:commit drift' >&2; exit 74;
  }
  [[ $(cd "$RWWPO_REPO_DIR" && git branch --show-current) == "$RWWPO_EXPECTED_BRANCH" ]] || {
    echo 'RWWPO2_NO_GO:branch drift' >&2; exit 75;
  }
  [[ -z $(cd "$RWWPO_REPO_DIR" && git status --porcelain) ]] || {
    echo 'RWWPO2_NO_GO:dirty tree' >&2; exit 76;
  }
}

rwwpo2_consume_attempt_id() {
  [[ ! -e $RWWPO_ATTEMPT_ROOT ]] || {
    echo 'RWWPO2_NO_GO:run ID already consumed' >&2; exit 77;
  }
  mkdir "$RWWPO_ATTEMPT_ROOT" || {
    echo 'RWWPO2_NO_GO:run ID allocation race' >&2; exit 78;
  }
  printf '%s\n' "$RWWPO_EXPECTED_COMMIT:$RWWPO_CELL:$RWWPO_EXPERIMENT_SEED" \
    > "$RWWPO_ATTEMPT_ROOT/RUN_ID_CONSUMED"
  mkdir "$RWWPO_CERT_ROOT"
}

rwwpo2_acquire_gpu_locks() {
  command -v flock >/dev/null || { echo 'RWWPO2_NO_GO:flock unavailable' >&2; exit 79; }
  mkdir -p "$RWWPO_WORK_ROOT/locks"
  exec 8>"$RWWPO_WORK_ROOT/locks/memagent_h20_gpu_${RWWPO_GPU_A}.lock"
  flock -n 8 || { echo 'RWWPO2_NO_GO:first GPU lock conflict; no process changed' >&2; exit 80; }
  exec 9>"$RWWPO_WORK_ROOT/locks/memagent_h20_gpu_${RWWPO_GPU_B}.lock"
  flock -n 9 || { echo 'RWWPO2_NO_GO:second GPU lock conflict; no process changed' >&2; exit 81; }
}

rwwpo2_require_idle_twice() {
  command -v nvidia-smi >/dev/null || { echo 'RWWPO2_NO_GO:nvidia-smi unavailable' >&2; exit 82; }
  local first second
  first=$(nvidia-smi -i "$GPU_PAIR" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${first//[[:space:]]/} ]] || {
    echo "RWWPO2_NO_GO:GPU occupied; no process changed: $first" >&2; exit 83;
  }
  second=$(nvidia-smi -i "$GPU_PAIR" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${second//[[:space:]]/} ]] || {
    echo "RWWPO2_NO_GO:GPU became occupied; no process changed: $second" >&2; exit 84;
  }
}

rwwpo2_export_runtime() {
  export CUDA_VISIBLE_DEVICES=$GPU_PAIR
  export RWWPO_ENABLE=1 RWWPO_LEDGER_DIR RWWPO_OBJECTIVE_VARIANT RWWPO_CONTROLLER_VARIANT
  export RWWPO_PROGRAM_VERSION=rwwpo2-k2 RWWPO_CELL
  export RWWPO_INNER_TRANSACTIONS=2 RWWPO_Q_MIN=0.5 RWWPO_ROOT_Q_MIN=0.5
  export RWWPO_PROPOSAL_SCHEDULE_KIND=constant_with_linear_warmup
  export RWWPO_PROPOSAL_BASE_LR=1e-6 RWWPO_PROPOSAL_WARMUP=2 RWWPO_PROPOSAL_TOTAL=800
  export RWWPO_ATTEMPT_ID=$RWWPO_RUN_ID
  export GATE_A_FROZEN_AUDIT=1 GATE_A_EXECUTION_LEDGER=$RWWPO_EXECUTION_LEDGER
  export GATE_A_EXPERIMENT_NAME=$RWWPO_EXPERIMENT GATE_A_GIT_COMMIT=$RWWPO_EXPECTED_COMMIT
  export GATE_A_RUN_ID
  GATE_A_RUN_ID=$(printf '%s' "$RWWPO_EXPECTED_COMMIT:$RWWPO_RUN_ID" | sha256sum | cut -c1-32)
  export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
  export GATE_A_WEIGHT_DIGEST_SAMPLES=256
}

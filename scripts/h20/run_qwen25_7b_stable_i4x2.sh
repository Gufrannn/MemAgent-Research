#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/stable_i4x2_common.sh"
stable_i_require_clean_frozen_checkout
stable_i_acquire_run_lock
stable_i_require_p0
stable_i_require_declared_gpus_idle

readonly STABLE_I_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight

# Stable-I reuses the read-only sampled-weight RPC but must never write into a
# Gate A ledger inherited from another user's shell on this shared account.
unset GATE_A_FROZEN_AUDIT GATE_A_EXECUTION_LEDGER GATE_A_EXPERIMENT_NAME
unset GATE_A_GIT_COMMIT GATE_A_RUN_ID
export GATE_A_WEIGHT_DIGEST_PARAMETERS=$STABLE_I_DIGEST_PARAMETERS
export GATE_A_WEIGHT_DIGEST_SAMPLES=256

run_attempt() {
  local attempt_id=$1
  local attempt_root=$STABLE_I_LOG_ROOT/$attempt_id
  local run_log=$attempt_root/run.log
  local ray_tmp=/tmp/msi_${UID}_${attempt_id}_$$
  local overrides_file=$ray_tmp/trainer_overrides.txt
  local -a trainer_overrides

  # Recheck immediately before each independent attempt.  The project lock
  # prevents our own wrappers racing, while this catches unrelated jobs on a
  # shared account that do not honor that lock.
  stable_i_require_declared_gpus_idle

  [[ ! -e $attempt_root ]] || {
    echo "STABLE_I_NO_GO:APPEND_ONLY attempt path already exists: $attempt_root" >&2
    exit 72
  }
  mkdir -p "$attempt_root" "$ray_tmp"

  "$STABLE_I_PYTHON" "$STABLE_I_REPO_DIR/tools/h20/preflight_qwen25_7b_stable_i4x2.py" \
    --manifest "$STABLE_I_MANIFEST" --emit-trainer-overrides --attempt-id "$attempt_id" \
    >"$overrides_file"
  mapfile -t trainer_overrides <"$overrides_file"
  [[ ${#trainer_overrides[@]} -gt 0 ]] || {
    echo "STABLE_I_NO_GO:CONFIG empty frozen trainer argv for $attempt_id" >&2
    exit 73
  }

  "$STABLE_I_PYTHON" "$STABLE_I_REPO_DIR/tools/h20/preflight_qwen25_7b_stable_i4x2.py" \
    --manifest "$STABLE_I_MANIFEST" --record-attempt-event start --attempt-id "$attempt_id"

  unset RAY_ADDRESS
  export CUDA_VISIBLE_DEVICES=$STABLE_I_GPU_DECLARATION
  export RAY_TMPDIR=$ray_tmp
  export TMPDIR=$ray_tmp
  export HYDRA_FULL_ERROR=1
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export WANDB_MODE=offline
  export VLLM_USE_V1=0
  export NCCL_DEBUG=WARN

  cd "$STABLE_I_REPO_DIR"
  "$STABLE_I_PYTHON" -m verl.trainer.main_ppo \
    "${trainer_overrides[@]}" \
    2>&1 | tee -a "$run_log"

  "$STABLE_I_PYTHON" "$STABLE_I_REPO_DIR/tools/h20/preflight_qwen25_7b_stable_i4x2.py" \
    --manifest "$STABLE_I_MANIFEST" --record-attempt-event finish --attempt-id "$attempt_id"
}

run_attempt repeat_a
stable_i_wait_for_attempt_cleanup
run_attempt repeat_b

"$STABLE_I_PYTHON" "$STABLE_I_REPO_DIR/tools/h20/audit_qwen25_7b_stable_i4x2.py" \
  --manifest "$STABLE_I_MANIFEST" --write-report

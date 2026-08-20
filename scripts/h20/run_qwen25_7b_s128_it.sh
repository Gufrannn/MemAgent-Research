#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/s128_it_common.sh"
s128_it_require_checkout
s128_it_acquire_lock
s128_it_require_p0
s128_it_require_idle

readonly S128_IT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
while IFS= read -r inherited_gate_a_variable; do
  unset "$inherited_gate_a_variable"
done < <(compgen -v GATE_A_ || true)
unset ORIGINAL_T25_EXPECTED_RUNTIME_CONFIG_SHA256
unset ORIGINAL_T25_TRAINER_OVERRIDE_ARGV_SHA256
export GATE_A_WEIGHT_DIGEST_PARAMETERS=$S128_IT_DIGEST_PARAMETERS
export GATE_A_WEIGHT_DIGEST_SAMPLES=256

run_interface() {
  local interface_id=$1
  local interface_root
  local run_log ray_tmp overrides_file
  local -a trainer_overrides
  case "$interface_id" in
    I) interface_root=$S128_IT_LOG_ROOT/interface_i_base ;;
    T25) interface_root=$S128_IT_LOG_ROOT/interface_t25_original ;;
    *) echo "S128_IT_NO_GO:CONFIG unknown interface $interface_id" >&2; exit 73 ;;
  esac
  run_log=$interface_root/run.log
  ray_tmp=/tmp/ms128_${UID}_${interface_id}_$$
  overrides_file=$ray_tmp/trainer_overrides.txt
  s128_it_require_idle
  [[ ! -e $interface_root ]] || {
    echo "S128_IT_NO_GO:APPEND_ONLY interface path exists: $interface_root" >&2; exit 72;
  }
  mkdir -p "$interface_root" "$ray_tmp"
  "$S128_IT_PYTHON" "$S128_IT_REPO_DIR/tools/h20/preflight_qwen25_7b_s128_it.py" \
    --manifest "$S128_IT_MANIFEST" --emit-trainer-overrides --interface "$interface_id" \
    >"$overrides_file"
  mapfile -t trainer_overrides <"$overrides_file"
  [[ ${#trainer_overrides[@]} -gt 0 ]] || {
    echo "S128_IT_NO_GO:CONFIG empty frozen argv for $interface_id" >&2; exit 73;
  }
  "$S128_IT_PYTHON" "$S128_IT_REPO_DIR/tools/h20/preflight_qwen25_7b_s128_it.py" \
    --manifest "$S128_IT_MANIFEST" --record-interface-event start --interface "$interface_id"

  unset RAY_ADDRESS
  export CUDA_VISIBLE_DEVICES=$S128_IT_GPUS
  export RAY_TMPDIR=$ray_tmp
  export TMPDIR=$ray_tmp
  export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  export WANDB_MODE=offline VLLM_USE_V1=0 NCCL_DEBUG=WARN
  cd "$S128_IT_REPO_DIR"
  "$S128_IT_PYTHON" -m verl.trainer.main_ppo "${trainer_overrides[@]}" 2>&1 | tee -a "$run_log"

  "$S128_IT_PYTHON" "$S128_IT_REPO_DIR/tools/h20/preflight_qwen25_7b_s128_it.py" \
    --manifest "$S128_IT_MANIFEST" --record-interface-event finish --interface "$interface_id"
}

run_interface I
s128_it_wait_cleanup
run_interface T25

"$S128_IT_PYTHON" "$S128_IT_REPO_DIR/tools/h20/audit_qwen25_7b_s128_it.py" \
  --manifest "$S128_IT_MANIFEST" --write-report

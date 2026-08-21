#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/original_s128_curve_common.sh"
original_curve_require_checkout
original_curve_acquire_lock
original_curve_require_p0

readonly ORIGINAL_CURVE_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight

# Shared accounts frequently retain experiment-writer variables in the shell.
# Clear them before setting only the read-only digest configuration used here.
while IFS= read -r inherited_variable; do
  unset "$inherited_variable"
done < <(compgen -v GATE_A_ || true)
while IFS= read -r inherited_variable; do
  unset "$inherited_variable"
done < <(compgen -v ORIGINAL_T25_ || true)
while IFS= read -r inherited_variable; do
  unset "$inherited_variable"
done < <(compgen -v MEMAGENT_T25_ || true)
export GATE_A_WEIGHT_DIGEST_PARAMETERS=$ORIGINAL_CURVE_DIGEST_PARAMETERS
export GATE_A_WEIGHT_DIGEST_SAMPLES=256

ORIGINAL_CURVE_ACTIVE_RAY_TMP=
cleanup_original_curve_ray_tmp() {
  local target=${ORIGINAL_CURVE_ACTIVE_RAY_TMP:-}
  [[ -n $target ]] || return 0
  if [[ ! $target =~ ^/tmp/oc(i|5|10|15|20|25)\.[[:alnum:]]{6}$ ]]; then
    echo "ORIGINAL_S128_CURVE_NO_GO:CLEANUP refusing unexpected Ray temp path: $target" >&2
    return 82
  fi
  rm -rf -- "$target"
  [[ ${RAY_TMPDIR:-} != "$target" ]] || unset RAY_TMPDIR
  [[ ${TMPDIR:-} != "$target" ]] || unset TMPDIR
  ORIGINAL_CURVE_ACTIVE_RAY_TMP=
}
trap cleanup_original_curve_ray_tmp EXIT

interface_root() {
  case "$1" in
    I) echo "$ORIGINAL_CURVE_LOG_ROOT/interface_i_base" ;;
    Original5) echo "$ORIGINAL_CURVE_LOG_ROOT/interface_original_step5" ;;
    Original10) echo "$ORIGINAL_CURVE_LOG_ROOT/interface_original_step10" ;;
    Original15) echo "$ORIGINAL_CURVE_LOG_ROOT/interface_original_step15" ;;
    Original20) echo "$ORIGINAL_CURVE_LOG_ROOT/interface_original_step20" ;;
    Original25) echo "$ORIGINAL_CURVE_LOG_ROOT/interface_original_step25" ;;
    *) echo "ORIGINAL_S128_CURVE_NO_GO:CONFIG unknown interface $1" >&2; exit 73 ;;
  esac
}

record_event() {
  "$ORIGINAL_CURVE_PYTHON" \
    "$ORIGINAL_CURVE_REPO_DIR/tools/h20/preflight_qwen25_7b_original_s128_curve.py" \
    --manifest "$ORIGINAL_CURVE_MANIFEST" --record-interface-event "$2" --interface "$1"
}

process_interface() {
  local interface_id=$1 mode root run_log ray_tag ray_tmp overrides_file
  local -a trainer_overrides
  mode=$("$ORIGINAL_CURVE_PYTHON" \
    "$ORIGINAL_CURVE_REPO_DIR/tools/h20/preflight_qwen25_7b_original_s128_curve.py" \
    --manifest "$ORIGINAL_CURVE_MANIFEST" --interface-mode --interface "$interface_id")
  if [[ $mode == import ]]; then
    echo "ORIGINAL_S128_CURVE_IMPORT: $interface_id uses hash-authenticated prior evidence read-only"
    record_event "$interface_id" import
    return 0
  fi
  [[ $mode == run ]] || {
    echo "ORIGINAL_S128_CURVE_NO_GO:PLAN invalid mode $mode for $interface_id" >&2; exit 74;
  }

  original_curve_require_idle
  root=$(interface_root "$interface_id")
  [[ ! -e $root ]] || {
    echo "ORIGINAL_S128_CURVE_NO_GO:APPEND_ONLY interface path exists: $root" >&2; exit 72;
  }
  case "$interface_id" in
    I) ray_tag=i ;;
    Original5) ray_tag=5 ;;
    Original10) ray_tag=10 ;;
    Original15) ray_tag=15 ;;
    Original20) ray_tag=20 ;;
    Original25) ray_tag=25 ;;
    *) echo "ORIGINAL_S128_CURVE_NO_GO:CONFIG unknown Ray tag for $interface_id" >&2; exit 73 ;;
  esac
  # Ray appends session_<timestamp>_<pid>/sockets/plasma_store.  Keep this
  # mktemp base extremely short so the complete Linux AF_UNIX path is <107.
  ray_tmp=$(mktemp -d "/tmp/oc${ray_tag}.XXXXXX")
  [[ $ray_tmp =~ ^/tmp/oc(i|5|10|15|20|25)\.[[:alnum:]]{6}$ ]] || {
    echo "ORIGINAL_S128_CURVE_NO_GO:CONFIG unsafe Ray temp path: $ray_tmp" >&2; exit 73;
  }
  ORIGINAL_CURVE_ACTIVE_RAY_TMP=$ray_tmp
  overrides_file=$ray_tmp/trainer_overrides.txt
  "$ORIGINAL_CURVE_PYTHON" \
    "$ORIGINAL_CURVE_REPO_DIR/tools/h20/preflight_qwen25_7b_original_s128_curve.py" \
    --manifest "$ORIGINAL_CURVE_MANIFEST" --emit-trainer-overrides --interface "$interface_id" \
    >"$overrides_file"
  mapfile -t trainer_overrides <"$overrides_file"
  [[ ${#trainer_overrides[@]} -gt 0 ]] || {
    echo "ORIGINAL_S128_CURVE_NO_GO:CONFIG empty frozen argv for $interface_id" >&2; exit 73;
  }
  record_event "$interface_id" start
  mkdir -p "$root"
  run_log=$root/run.log

  unset RAY_ADDRESS
  export CUDA_VISIBLE_DEVICES=$ORIGINAL_CURVE_GPUS
  export RAY_TMPDIR=$ray_tmp
  export TMPDIR=$ray_tmp
  export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  export WANDB_MODE=offline VLLM_USE_V1=0 NCCL_DEBUG=WARN
  cd "$ORIGINAL_CURVE_REPO_DIR"
  "$ORIGINAL_CURVE_PYTHON" -m verl.trainer.main_ppo "${trainer_overrides[@]}" \
    2>&1 | tee -a "$run_log"
  record_event "$interface_id" finish
  original_curve_wait_cleanup
  cleanup_original_curve_ray_tmp
}

for interface_id in I Original5 Original10 Original15 Original20 Original25; do
  process_interface "$interface_id"
done

"$ORIGINAL_CURVE_PYTHON" \
  "$ORIGINAL_CURVE_REPO_DIR/tools/h20/audit_qwen25_7b_original_s128_curve.py" \
  --manifest "$ORIGINAL_CURVE_MANIFEST" --write-report

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/hdr_memrl_common.sh"
: "${HDR_ANCHOR:?set HDR_ANCHOR=5,10,15,20,25}"
[[ $HDR_ANCHOR =~ ^(5|10|15|20|25)$ ]] || { echo HDR_NO_GO:invalid_anchor >&2; exit 81; }
hdr_require_checkout; hdr_require_gates
readonly EVAL_ROOT=$HDR_ROOT/eval
readonly MERGED=$EVAL_ROOT/t${HDR_ANCHOR}_merged
readonly STEP_DIR=$HDR_OUTPUT/global_step_$HDR_ANCHOR
[[ -d $STEP_DIR/actor && -f $STEP_DIR/hdr_checkpoint_binding.json ]] || { echo HDR_NO_GO:anchor_checkpoint_missing >&2; exit 82; }
[[ ! -e $MERGED ]] || { echo HDR_NO_GO:merged_output_exists >&2; exit 83; }
mkdir -p "$EVAL_ROOT"
"$HDR_PYTHON" "$HDR_REPO/scripts/model_merger.py" --backend fsdp \
  --hf_model_path "$MEMAGENT_HDR_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --local_dir "$STEP_DIR/actor" --target_dir "$MERGED"
export CUDA_VISIBLE_DEVICES=$GPU_PAIR
"$HDR_PYTHON" "$HDR_REPO/tools/h20/run_hdr_strict_vllm_eval.py" \
  --suite "$EVAL_ROOT/fixed_s128_all_horizons.parquet" --model "$MERGED" \
  --output "$EVAL_ROOT/t${HDR_ANCHOR}_horizons.json" --seed 2026 --tensor-parallel-size 2
"$HDR_PYTHON" "$HDR_REPO/tools/h20/run_hdr_strict_vllm_eval.py" \
  --suite "$EVAL_ROOT/fixed_s128_nominal_h8.parquet" --model "$MERGED" \
  --output "$EVAL_ROOT/t${HDR_ANCHOR}_s128_nominal.json" --seed 2026 --tensor-parallel-size 2
UNIFORM_ARGS=()
if [[ $HDR_VARIANT == dro && $HDR_ANCHOR -eq 25 ]]; then
  [[ -f $EVAL_ROOT/uniform_erm_t25_horizons_authority.json ]] || { echo HDR_NO_GO:uniform_t25_authority_missing >&2; exit 84; }
  UNIFORM_ARGS=(--uniform-horizons "$EVAL_ROOT/uniform_erm_t25_horizons_authority.json")
fi
"$HDR_PYTHON" "$HDR_REPO/tools/h20/hdr_memrl_control.py" health-gate \
  --variant "$HDR_VARIANT" --anchor "$HDR_ANCHOR" --checkpoint-binding "$STEP_DIR/hdr_checkpoint_binding.json" \
  --baseline-import "$HDR_CERT/baseline_import.json" \
  --method-s128 "$EVAL_ROOT/t${HDR_ANCHOR}_s128_nominal.json" \
  --method-horizons "$EVAL_ROOT/t${HDR_ANCHOR}_horizons.json" \
  --original-horizons "$EVAL_ROOT/original_t${HDR_ANCHOR}_horizons_authority.json" \
  --model-path "$MERGED" --seed 2026 --nominal 8 --unseen 10 24 \
  "${UNIFORM_ARGS[@]}" --output "$HDR_CERT/t${HDR_ANCHOR}_health.json" --ledger "$HDR_LEDGER"

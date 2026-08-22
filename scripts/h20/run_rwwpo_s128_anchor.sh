#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); source "$SCRIPT_DIR/rwwpo_common.sh"
[[ ${RWWPO_EVAL_STEP:-} =~ ^(5|10|15|20|25)$ ]] || { echo 'RWWPO_S128_NO_GO:set anchor step' >&2; exit 80; }
[[ -f ${RWWPO_EVAL_RESOLVED_MANIFEST:-} && ${RWWPO_EVAL_RESOLVED_SHA256:-} =~ ^[0-9a-f]{64}$ && ${RWWPO_EVAL_MANIFEST_HASH:-} =~ ^[0-9a-f]{64}$ ]] || { echo 'RWWPO_S128_NO_GO:bind frozen S128 resolved identity manifest' >&2; exit 81; }
rwwpo_require_checkout; rwwpo_acquire_gpu_locks; rwwpo_require_idle
CHECKPOINT=$RWWPO_OUTPUT/global_step_$RWWPO_EVAL_STEP; [[ -d $CHECKPOINT/actor && -f $CHECKPOINT/data.pt ]] || { echo 'RWWPO_S128_NO_GO:checkpoint incomplete' >&2; exit 82; }
DIAG_ARGS=(); EVAL_SUFFIX=""
if [[ ${RWWPO_DIAGNOSTIC_ONLY:-0} == 1 ]]; then
  [[ ${RWWPO_EVAL_ATTEMPT_SUFFIX:-} =~ ^diag_[a-z0-9][a-z0-9_-]{2,31}$ ]] || { echo 'RWWPO_S128_NO_GO:set diagnostic attempt suffix' >&2; exit 84; }
  EVAL_SUFFIX=_$RWWPO_EVAL_ATTEMPT_SUFFIX; DIAG_ARGS=(--diagnostic-only)
fi
EVAL_ROOT=$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/s128_t$RWWPO_EVAL_STEP$EVAL_SUFFIX; [[ ! -e $EVAL_ROOT ]] || { echo 'RWWPO_S128_NO_GO:append-only eval output exists' >&2; exit 83; }
mkdir -p "$EVAL_ROOT"; mapfile -t OVERRIDES < <("$RWWPO_PYTHON" "$RWWPO_REPO_DIR/tools/h20/preflight_rwwpo_s128.py" --checkpoint "$CHECKPOINT" --validation "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" --model "$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct" --resolved-manifest "$RWWPO_EVAL_RESOLVED_MANIFEST" --resolved-manifest-sha256 "$RWWPO_EVAL_RESOLVED_SHA256" --eval-hash "$RWWPO_EVAL_MANIFEST_HASH" --run-id "$RWWPO_RUN_ID$EVAL_SUFFIX" --output "$EVAL_ROOT" --step "$RWWPO_EVAL_STEP" "${DIAG_ARGS[@]}")
unset GATE_A_FROZEN_AUDIT GATE_A_EXECUTION_LEDGER; export CUDA_VISIBLE_DEVICES=$GPU_PAIR
export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight GATE_A_WEIGHT_DIGEST_SAMPLES=256
RAY_TMP=/tmp/mrwe_${UID}_${RWWPO_EVAL_STEP}_$$; mkdir -p "$RAY_TMP"; export RAY_TMPDIR=$RAY_TMP TMPDIR=$RAY_TMP HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline VLLM_USE_V1=0 NCCL_DEBUG=WARN
cd "$RWWPO_REPO_DIR"; "$RWWPO_PYTHON" -m verl.trainer.main_ppo "${OVERRIDES[@]}" 2>&1 | tee -a "$EVAL_ROOT/run.log"

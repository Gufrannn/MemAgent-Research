#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/cosi_common.sh"
cosi_checkout_guard
cosi_acquire_gpu_locks
[[ $# == 1 && $1 =~ ^(5|10|15|20|25)$ ]] || { echo CORAL_S128_NO_GO:usage_STEP >&2; exit 81; }
readonly STEP=$1
readonly PYTHON=$MEMAGENT_COSI_WORK_ROOT/.venv/bin/python
readonly RUN_ID=${MEMAGENT_COSI_RUN_ID:-coral_seed2026_primary_v1}
readonly EXP=qwen25_7b_coral_fresh_t25_seed2026_${RUN_ID}
readonly CHECKPOINT=$MEMAGENT_COSI_WORK_ROOT/logs/memory_agent/$EXP/global_step_$STEP
readonly EVAL_ROOT=$MEMAGENT_COSI_WORK_ROOT/logs/coral/$RUN_ID/fixed_s128/T$STEP
readonly VALIDATION=$MEMAGENT_COSI_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet
readonly OVERRIDES=$EVAL_ROOT/certificates/overrides.txt
readonly OVERRIDES_TMP=$(mktemp /tmp/coral_s128_overrides.XXXXXX)
[[ -d $CHECKPOINT && ! -e $EVAL_ROOT ]] || { echo CORAL_S128_NO_GO:checkpoint_or_append_only >&2; exit 82; }
"$PYTHON" "$MEMAGENT_COSI_REPO_DIR/tools/h20/preflight_coral_s128.py" \
  --checkpoint "$CHECKPOINT" --step "$STEP" --output-root "$EVAL_ROOT" --emit-overrides >"$OVERRIDES_TMP"
mv "$OVERRIDES_TMP" "$OVERRIDES"
mapfile -t trainer_overrides <"$OVERRIDES"
[[ ${#trainer_overrides[@]} -gt 0 ]] || { echo CORAL_S128_NO_GO:empty_overrides >&2; exit 83; }
while IFS= read -r inherited; do unset "$inherited"; done < <(compgen -v GATE_A_ || true)
export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
export GATE_A_WEIGHT_DIGEST_SAMPLES=256
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline VLLM_USE_V1=0 NCCL_DEBUG=WARN
export RAY_TMPDIR=/tmp/mcoral_s128_${UID}_${STEP}_$$ TMPDIR=$RAY_TMPDIR
mkdir -p "$RAY_TMPDIR"
cd "$MEMAGENT_COSI_REPO_DIR"
"$PYTHON" -m verl.trainer.main_ppo "${trainer_overrides[@]}" 2>&1 | tee "$EVAL_ROOT/run.log"
"$PYTHON" tools/h20/audit_coral_s128.py --evaluation-root "$EVAL_ROOT" \
  --checkpoint "$CHECKPOINT" --validation "$VALIDATION" \
  --output "$EVAL_ROOT/certificates/final_report.json"

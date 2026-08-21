#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'usage: eval_audit_qwen25_7b_mic.sh STEP' >&2; exit 64; }
STEP=$1
case "$STEP" in 5|10|15|20|25) ;; *) echo 'MIC_NO_GO: invalid anchor' >&2; exit 65;; esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/mic_common.sh"
mic_require_checkout; mic_require_training_gates; mic_acquire_gpu_locks; mic_require_idle
[[ -d $MIC_OUTPUT/global_step_${STEP}/actor ]] || { echo 'MIC_NO_GO: actor checkpoint absent' >&2; exit 80; }
EVAL_ROOT=$MIC_ROOT/eval_t${STEP}
[[ ! -e $EVAL_ROOT ]] || { echo 'MIC_NO_GO: evaluation output exists' >&2; exit 81; }
mkdir -p "$EVAL_ROOT"
mic_export_training
export MEMAGENT_MIC_EVAL_STEP=$STEP MEMAGENT_MIC_EVAL_DIR=$EVAL_ROOT/raw
IDENTITY_SOURCE=$(
  "$MIC_PYTHON" -c 'import json,os,sys; x=json.load(open(sys.argv[1])); name="Original"+sys.argv[2]; rows=[r for r in x["prediction_files"] if r["interface"]==name]; assert len(rows)==1; print(os.path.expandvars(rows[0]["path"]))' \
    "$MEMAGENT_MIC_BASELINE_INVENTORY" "$STEP"
)
export MEMAGENT_MIC_EVAL_IDENTITY_PATH=$IDENTITY_SOURCE
export MEMAGENT_MIC_EVAL_IDENTITY_SHA256
MEMAGENT_MIC_EVAL_IDENTITY_SHA256=$("$MIC_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$IDENTITY_SOURCE")
env WORK_ROOT="$MEMAGENT_MIC_WORK_ROOT" CODE="$MEMAGENT_MIC_REPO_DIR" PYTHON="$MIC_PYTHON" \
  MODEL="$MEMAGENT_MIC_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  TRAIN="$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  VAL="$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  PHASE=resume EXP="$MIC_EXPERIMENT" RESUME_SOURCE_STEP="$STEP" RESUME_TOTAL_STEPS=$((STEP+1)) \
  RUN_SEED=2026 TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 N_GPUS=2 FSDP_SIZE=2 \
  REWARD_MANAGER=naive GPU_MEMORY_UTILIZATION=0.55 SAVE_FREQ=5 MAX_ACTOR_CKPT_TO_KEEP=5 \
  bash "$MEMAGENT_MIC_REPO_DIR/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$EVAL_ROOT/eval.log"

"$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" prepare-eval \
  --generations "$EVAL_ROOT/raw/${STEP}.jsonl" \
  --validation "$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  --identity-source "$IDENTITY_SOURCE" --output "$EVAL_ROOT/predictions.jsonl"
"$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" audit \
  --p0 "$MIC_P0" --e0 "$MIC_E0" --e1 "$MIC_E1" --paper-review "$MIC_PAPER_REVIEW" \
  --baseline "$MIC_BASELINE" --ledger "$MIC_LEDGER" --target-step "$STEP" \
  --weight-ledger "$MIC_WEIGHT_LEDGER" \
  --output "$MIC_CERT/t${STEP}_audit.json"
"$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" evaluate \
  --predictions "$EVAL_ROOT/predictions.jsonl" --baseline "$MIC_BASELINE" --step "$STEP" \
  --output "$MIC_CERT/$( [[ $STEP -eq 5 ]] && echo t5_health || echo t${STEP}_eval ).json"

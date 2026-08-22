#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'usage: eval_audit_qwen25_7b_mic.sh STEP' >&2; exit 64; }
STEP=$1
case "$STEP" in 5|10|15|20|25) ;; *) echo 'MIC_NO_GO: invalid anchor' >&2; exit 65;; esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/mic_common.sh"
mic_require_checkout; mic_require_training_gates
if [[ ! -e $MIC_BASELINE_INVENTORY ]]; then
  "$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" materialize-baseline \
    --p0 "$MIC_P0" --curve-report "$MEMAGENT_MIC_ORIGINAL_CURVE_REPORT" \
    --curve-resolved "$MIC_CURVE_RESOLVED" --search-root "$MEMAGENT_MIC_WORK_ROOT/logs" \
    --validation "$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
    --curve-authority "$MIC_CURVE_AUTHORITY" \
    --output-root "$MIC_BASELINE_ROOT/rows" --output "$MIC_BASELINE_INVENTORY"
fi
export MEMAGENT_MIC_BASELINE_INVENTORY=$MIC_BASELINE_INVENTORY
export MEMAGENT_MIC_BASELINE_AUTHORITY_SHA256
MEMAGENT_MIC_BASELINE_AUTHORITY_SHA256=$(
  "$MIC_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["original_curve_report_sha256"])' "$MIC_P0"
)
if [[ ! -e $MIC_BASELINE ]]; then
  "$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" import-baseline \
    --manifest "$MIC_MANIFEST" --p0 "$MIC_P0" --output "$MIC_BASELINE"
else
  "$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" import-baseline \
    --manifest "$MIC_MANIFEST" --p0 "$MIC_P0" --output "$MIC_BASELINE" --verify-existing
fi
mic_require_gate "$MIC_BASELINE" MIC_BASELINE_IMPORT_PASS
mic_require_gate "$MIC_CERT/t${STEP}_audit.json" MIC_T${STEP}_AUDIT_PASS
[[ -d $MIC_OUTPUT/global_step_${STEP}/actor ]] || { echo 'MIC_NO_GO: actor checkpoint absent' >&2; exit 80; }
if [[ ! -e $MIC_CHECKPOINT_AUTHORITY_CERT ]]; then
  "$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" \
    materialize-checkpoint-authority --authority "$MIC_CHECKPOINT_AUTHORITY" \
    --p0 "$MIC_P0" --output-root "$MIC_OUTPUT" --ledger "$MIC_LEDGER" \
    --output "$MIC_CHECKPOINT_AUTHORITY_CERT"
fi
mic_require_gate "$MIC_CHECKPOINT_AUTHORITY_CERT" MIC_CHECKPOINT_AUTHORITY_PASS
IDENTITY_SOURCE=$(
  "$MIC_PYTHON" -c 'import json,os,sys; x=json.load(open(sys.argv[1])); name="Original"+sys.argv[2]; rows=[r for r in x["prediction_files"] if r["interface"]==name]; assert len(rows)==1; print(os.path.expandvars(rows[0]["path"]))' \
    "$MEMAGENT_MIC_BASELINE_INVENTORY" "$STEP"
)

mic_prepare_anchor() {
  "$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" prepare-eval \
    --generations "$EVAL_ROOT/raw/${STEP}.jsonl" \
    --validation "$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
    --identity-source "$IDENTITY_SOURCE" \
    --execution-summary "$EVAL_ROOT/execution_summary.json" \
    --training-audit "$MIC_CERT/t${STEP}_audit.json" \
    --baseline "$MIC_BASELINE" --p0 "$MIC_P0" \
    --checkpoint "$MIC_OUTPUT/global_step_${STEP}" --step "$STEP" \
    --checkpoint-authority "$MIC_CHECKPOINT_AUTHORITY" \
    --checkpoint-authority-certificate "$MIC_CHECKPOINT_AUTHORITY_CERT" \
    --output-root "$MIC_OUTPUT" --weight-ledger "$MIC_WEIGHT_LEDGER" \
    --mic-ledger "$MIC_LEDGER" \
    --output "$EVAL_ROOT/predictions.jsonl" --report "$EVAL_ROOT/prepare.json" "$@"
}

if [[ -e $MIC_CERT/t${STEP}_eval.json ]]; then
  EVAL_ROOT=$(
    "$MIC_PYTHON" -c 'import json,pathlib,re,sys; root=pathlib.Path(json.load(open(sys.argv[1]))["evaluation_root"]).resolve(); expected=pathlib.Path(sys.argv[2]).resolve(); assert root.parent==expected and re.fullmatch(r"attempt_[0-9]{4}",root.name); print(root)' \
      "$MIC_CERT/t${STEP}_eval.json" "$MIC_ROOT/eval_t${STEP}_attempts"
  )
  mic_prepare_anchor --verify-existing
  "$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" evaluate \
    --predictions "$EVAL_ROOT/predictions.jsonl" --baseline "$MIC_BASELINE" \
    --p0 "$MIC_P0" --step "$STEP" --output "$MIC_CERT/t${STEP}_eval.json" \
    --verify-existing
  echo "MIC_T${STEP}_EVAL_ALREADY_COMPLETE_AND_REAUTHENTICATED"
  exit 0
fi

mic_acquire_gpu_locks; mic_require_idle
EVAL_ROOT=$(mic_next_eval_attempt "$STEP")
mic_export_evaluation
export MEMAGENT_MIC_EVAL_STEP=$STEP MEMAGENT_MIC_EVAL_DIR=$EVAL_ROOT/raw
export MEMAGENT_MIC_EVAL_IDENTITY_PATH=$IDENTITY_SOURCE
export MEMAGENT_MIC_EVAL_IDENTITY_SHA256
MEMAGENT_MIC_EVAL_IDENTITY_SHA256=$("$MIC_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$IDENTITY_SOURCE")
export MEMAGENT_MIC_EVAL_SUMMARY_PATH=$EVAL_ROOT/execution_summary.json
export MEMAGENT_MIC_EVAL_GENERATION_PATH=$EVAL_ROOT/raw/${STEP}.jsonl
export MEMAGENT_MIC_EVAL_ORIGINAL_PROTOCOL_SHA256 MEMAGENT_MIC_EVAL_ORIGINAL_REWARD_CODE_SHA256
read -r MEMAGENT_MIC_EVAL_ORIGINAL_PROTOCOL_SHA256 MEMAGENT_MIC_EVAL_ORIGINAL_REWARD_CODE_SHA256 < <(
  "$MIC_PYTHON" -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["shared_generation_protocol_sha256"], x["original_reward_code_sha256"])' \
    "$MIC_BASELINE"
)
export MEMAGENT_MIC_EVAL_TRAINING_AUDIT_SHA256
MEMAGENT_MIC_EVAL_TRAINING_AUDIT_SHA256=$(
  "$MIC_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
    "$MIC_CERT/t${STEP}_audit.json"
)
env WORK_ROOT="$MEMAGENT_MIC_WORK_ROOT" CODE="$MEMAGENT_MIC_REPO_DIR" PYTHON="$MIC_PYTHON" \
  MODEL="$MEMAGENT_MIC_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  TRAIN="$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  VAL="$MEMAGENT_MIC_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  PHASE=resume EXP="$MIC_EXPERIMENT" RESUME_SOURCE_STEP="$STEP" RESUME_TOTAL_STEPS=$((STEP+1)) \
  RUN_SEED=2026 TRAIN_BATCH_SIZE=2 ROLLOUT_N=1 PPO_MINI_BATCH_SIZE=2 N_GPUS=2 FSDP_SIZE=2 \
  REWARD_MANAGER=naive GPU_MEMORY_UTILIZATION=0.55 SAVE_FREQ=5 MAX_ACTOR_CKPT_TO_KEEP=5 \
  bash "$MEMAGENT_MIC_REPO_DIR/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$EVAL_ROOT/eval.log"

mic_prepare_anchor
"$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" evaluate \
  --predictions "$EVAL_ROOT/predictions.jsonl" --baseline "$MIC_BASELINE" --p0 "$MIC_P0" \
  --step "$STEP" \
  --output "$MIC_CERT/t${STEP}_eval.json"

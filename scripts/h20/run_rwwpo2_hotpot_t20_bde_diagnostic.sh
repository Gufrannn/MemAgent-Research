#!/usr/bin/env bash
set -euo pipefail

[[ -n ${RWWPO_REPO_DIR:-} && $RWWPO_REPO_DIR == /* ]] || {
  echo 'RWWPO2_T20_DIAG_NO_GO:set absolute RWWPO_REPO_DIR' >&2; exit 60;
}
[[ -n ${RWWPO_WORK_ROOT:-} && $RWWPO_WORK_ROOT == /* ]] || {
  echo 'RWWPO2_T20_DIAG_NO_GO:set absolute RWWPO_WORK_ROOT' >&2; exit 61;
}
[[ ${RWWPO_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'RWWPO2_T20_DIAG_NO_GO:set exact RWWPO_EXPECTED_COMMIT' >&2; exit 62;
}
[[ ${RWWPO2_TRAINING_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'RWWPO2_T20_DIAG_NO_GO:set exact RWWPO2_TRAINING_COMMIT' >&2; exit 63;
}
[[ ${GPU_PAIR:-} =~ ^[0-9]+,[0-9]+$ ]] || {
  echo 'RWWPO2_T20_DIAG_NO_GO:set GPU_PAIR=N,M' >&2; exit 64;
}
[[ -n ${RWWPO2_DIAG_ROOT:-} && $RWWPO2_DIAG_ROOT == /* \
    && ! -e $RWWPO2_DIAG_ROOT ]] || {
  echo 'RWWPO2_T20_DIAG_NO_GO:set unused absolute RWWPO2_DIAG_ROOT' >&2; exit 65;
}
[[ -f ${RWWPO2_S128_RESOLVED:-} \
    && ${RWWPO2_S128_RESOLVED_SHA256:-} =~ ^[0-9a-f]{64}$ \
    && ${RWWPO2_S128_MANIFEST_HASH:-} =~ ^[0-9a-f]{64}$ ]] || {
  echo 'RWWPO2_T20_DIAG_NO_GO:bind fixed-S128 identity' >&2; exit 66;
}
for cell in B D E; do
  variable=RWWPO2_${cell}_T20_CHECKPOINT
  checkpoint=${!variable:-}
  [[ $checkpoint == /*/global_step_20 && -f $checkpoint/data.pt \
      && -f $checkpoint/actor/model_world_size_2_rank_0.pt \
      && -f $checkpoint/actor/model_world_size_2_rank_1.pt ]] || {
    echo "RWWPO2_T20_DIAG_NO_GO:incomplete $cell T20 checkpoint" >&2; exit 67;
  }
done

cd "$RWWPO_REPO_DIR"
[[ $(git rev-parse HEAD) == "$RWWPO_EXPECTED_COMMIT" \
    && $(git branch --show-current) == \
      h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822 \
    && -z $(git status --porcelain) ]] || {
  echo 'RWWPO2_T20_DIAG_NO_GO:checkout' >&2; exit 68;
}
[[ $(sha256sum "$RWWPO2_S128_RESOLVED" | awk '{print $1}') == \
    "$RWWPO2_S128_RESOLVED_SHA256" ]] || {
  echo 'RWWPO2_T20_DIAG_NO_GO:S128 resolved SHA' >&2; exit 69;
}

RWWPO_PYTHON=$RWWPO_WORK_ROOT/.venv/bin/python
[[ -x $RWWPO_PYTHON ]] || { echo 'RWWPO2_T20_DIAG_NO_GO:venv' >&2; exit 70; }
mkdir -p "$RWWPO_WORK_ROOT/locks"
IFS=, read -r GPU_A GPU_B <<< "$GPU_PAIR"
(( GPU_A < GPU_B )) || { echo 'RWWPO2_T20_DIAG_NO_GO:GPU order' >&2; exit 71; }
exec 8>"$RWWPO_WORK_ROOT/locks/memagent_h20_gpu_${GPU_A}.lock"
flock -n 8 || { echo 'RWWPO2_T20_DIAG_NO_GO:first GPU lock' >&2; exit 72; }
exec 9>"$RWWPO_WORK_ROOT/locks/memagent_h20_gpu_${GPU_B}.lock"
flock -n 9 || { echo 'RWWPO2_T20_DIAG_NO_GO:second GPU lock' >&2; exit 73; }
wait_for_idle() {
  local label=$1 processes attempt
  for attempt in $(seq 1 24); do
    processes=$(nvidia-smi -i "$GPU_PAIR" --query-compute-apps=pid \
      --format=csv,noheader,nounits)
    if [[ -z ${processes//[[:space:]]/} ]]; then
      return 0
    fi
    sleep 5
  done
  echo "RWWPO2_T20_DIAG_NO_GO:GPU process after $label:$processes" >&2
  return 1
}
wait_for_idle startup || exit 74

mkdir -p "$(dirname "$RWWPO2_DIAG_ROOT")"
mkdir "$RWWPO2_DIAG_ROOT"
mkdir "$RWWPO2_DIAG_ROOT/certificates"
printf '%s\n' "$RWWPO_EXPECTED_COMMIT:$RWWPO2_TRAINING_COMMIT:$GPU_PAIR" \
  > "$RWWPO2_DIAG_ROOT/RUN_ID_CONSUMED"

export CUDA_VISIBLE_DEVICES=$GPU_PAIR HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline VLLM_USE_V1=0
export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
export GATE_A_WEIGHT_DIGEST_SAMPLES=256
unset GATE_A_FROZEN_AUDIT GATE_A_EXECUTION_LEDGER

for cell in B D E; do
  checkpoint_variable=RWWPO2_${cell}_T20_CHECKPOINT
  checkpoint=${!checkpoint_variable}
  manifest=$RWWPO2_DIAG_ROOT/${cell}_diagnostic_eval_manifest.json
  eval_root=$RWWPO2_DIAG_ROOT/${cell}_s128_t20
  certificate=$RWWPO2_DIAG_ROOT/certificates/${cell}_t20_s128_diagnostic.json
  "$RWWPO_PYTHON" tools/h20/materialize_rwwpo_diagnostic_eval_manifest.py \
    --source "$RWWPO2_S128_RESOLVED" \
    --source-sha256 "$RWWPO2_S128_RESOLVED_SHA256" \
    --checkpoint-root "$(dirname "$checkpoint")" \
    --training-commit "$RWWPO2_TRAINING_COMMIT" \
    --expected-commit "$RWWPO_EXPECTED_COMMIT" \
    --step 20 --output "$manifest"
  manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
  mkdir "$eval_root"
  mapfile -t overrides < <(
    "$RWWPO_PYTHON" tools/h20/preflight_rwwpo_s128.py \
      --checkpoint "$checkpoint" \
      --validation "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
      --model "$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
      --resolved-manifest "$manifest" \
      --resolved-manifest-sha256 "$manifest_sha" \
      --eval-hash "$RWWPO2_S128_MANIFEST_HASH" \
      --run-id "rwwpo2_t20_${cell,,}_diagnostic" \
      --output "$eval_root" --step 20 --diagnostic-only
  )
  ray_tmp=/tmp/rwwpo2_t20_${cell,,}_${UID}_$$
  mkdir -p "$ray_tmp"
  export RAY_TMPDIR=$ray_tmp TMPDIR=$ray_tmp
  echo "===== START $cell-T20 FIXED-S128 DIAGNOSTIC ====="
  "$RWWPO_PYTHON" -m verl.trainer.main_ppo "${overrides[@]}" \
    2>&1 | tee -a "$eval_root/run.log"
  "$RWWPO_PYTHON" tools/h20/audit_rwwpo_s128.py \
    --diagnostic-only --expected-commit "$RWWPO_EXPECTED_COMMIT" \
    --eval-root "$eval_root" --step 20 \
    --validation "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
    --resolved-manifest "$manifest" \
    --expected-manifest-sha256 "$manifest_sha" --output "$certificate"
  "$RWWPO_PYTHON" -c '
import json,sys
r=json.load(open(sys.argv[1])); print(sys.argv[2],json.dumps(r["metrics"],sort_keys=True))
' "$certificate" "$cell"
  wait_for_idle "$cell" || exit 75
done

"$RWWPO_PYTHON" tools/h20/compare_rwwpo2_hotpot_t20_bde.py \
  --b "$RWWPO2_DIAG_ROOT/certificates/B_t20_s128_diagnostic.json" \
  --d "$RWWPO2_DIAG_ROOT/certificates/D_t20_s128_diagnostic.json" \
  --e "$RWWPO2_DIAG_ROOT/certificates/E_t20_s128_diagnostic.json" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO2_DIAG_ROOT/certificates/BDE_t20_comparison.json"
touch "$RWWPO2_DIAG_ROOT/PIPELINE_PASS"
echo "RWWPO2 B/D/E T20 DIAGNOSTIC PIPELINE PASS"

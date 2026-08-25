#!/usr/bin/env bash
set -euo pipefail

for name in RWWPO_WORK_ROOT RWWPO_REPO_DIR RWWPO_EXPECTED_COMMIT GPU_PAIR \
  RWWPO_MANIFEST RWWPO_MANIFEST_SHA256 RWWPO_RELEASE_TEST_RECEIPT \
  RWWPO_RELEASE_TEST_RECEIPT_SHA256 \
  RWWPO_BABILONG_CHECKPOINT RWWPO_BABILONG_DATA RWWPO_BABILONG_MODEL \
  RWWPO_BABILONG_RESOLVED RWWPO_BABILONG_RESOLVED_SHA256 \
  RWWPO_BABILONG_EVAL_ROOT RWWPO_BABILONG_INTERFACE_ID \
  RWWPO_BABILONG_ATTEMPT_ID; do
  [[ -n ${!name:-} ]] || { echo "RWWPO2_BABILONG_EVAL_NO_GO:missing $name" >&2; exit 60; }
done
[[ $RWWPO_WORK_ROOT == /* && $RWWPO_REPO_DIR == /* \
   && $RWWPO_BABILONG_CHECKPOINT == /* && $RWWPO_BABILONG_DATA == /* \
   && $RWWPO_BABILONG_MODEL == /* && $RWWPO_BABILONG_RESOLVED == /* \
   && $RWWPO_BABILONG_EVAL_ROOT == /* ]] || {
  echo 'RWWPO2_BABILONG_EVAL_NO_GO:all runtime paths must be absolute' >&2; exit 61;
}
[[ $RWWPO_EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ \
   && $RWWPO_MANIFEST_SHA256 =~ ^[0-9a-f]{64}$ \
   && $RWWPO_RELEASE_TEST_RECEIPT_SHA256 =~ ^[0-9a-f]{64}$ \
   && $RWWPO_BABILONG_RESOLVED_SHA256 =~ ^[0-9a-f]{64}$ \
   && $GPU_PAIR =~ ^[0-9]+,[0-9]+$ ]] || {
  echo 'RWWPO2_BABILONG_EVAL_NO_GO:commit/SHA/GPU syntax' >&2; exit 62;
}
IFS=, read -r GPU_A GPU_B <<< "$GPU_PAIR"
(( GPU_A < GPU_B )) || { echo 'RWWPO2_BABILONG_EVAL_NO_GO:canonical GPU pair' >&2; exit 63; }
[[ $(cd "$RWWPO_REPO_DIR" && git rev-parse HEAD) == "$RWWPO_EXPECTED_COMMIT" \
   && $(cd "$RWWPO_REPO_DIR" && git branch --show-current) == \
      h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822 \
   && -z $(cd "$RWWPO_REPO_DIR" && git status --porcelain) ]] || {
  echo 'RWWPO2_BABILONG_EVAL_NO_GO:checkout' >&2; exit 64;
}
[[ ! -e $RWWPO_BABILONG_EVAL_ROOT ]] || {
  echo 'RWWPO2_BABILONG_EVAL_NO_GO:append-only eval root exists' >&2; exit 65;
}
"$RWWPO_WORK_ROOT/.venv/bin/python" \
  "$RWWPO_REPO_DIR/tools/h20/verify_rwwpo2_release_tests.py" \
  --receipt "$RWWPO_RELEASE_TEST_RECEIPT" \
  --receipt-sha256 "$RWWPO_RELEASE_TEST_RECEIPT_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --manifest "$RWWPO_MANIFEST" \
  --manifest-sha256 "$RWWPO_MANIFEST_SHA256" \
  --work-root "$RWWPO_WORK_ROOT"
command -v flock >/dev/null
mkdir -p "$RWWPO_WORK_ROOT/locks"
exec 8>"$RWWPO_WORK_ROOT/locks/memagent_h20_gpu_${GPU_A}.lock"
flock -n 8 || { echo 'RWWPO2_BABILONG_EVAL_NO_GO:first GPU lock conflict' >&2; exit 79; }
exec 9>"$RWWPO_WORK_ROOT/locks/memagent_h20_gpu_${GPU_B}.lock"
flock -n 9 || { echo 'RWWPO2_BABILONG_EVAL_NO_GO:second GPU lock conflict' >&2; exit 79; }
for pass in 1 2; do
  ACTIVE=$(nvidia-smi -i "$GPU_PAIR" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${ACTIVE//[[:space:]]/} ]] || {
    echo "RWWPO2_BABILONG_EVAL_NO_GO:GPU occupied pass${pass}; no process changed" >&2
    exit 79
  }
  [[ $pass == 1 ]] && sleep 5
done

mkdir -p "$RWWPO_BABILONG_EVAL_ROOT"
mapfile -t OVERRIDES < <(
  "$RWWPO_WORK_ROOT/.venv/bin/python" \
    "$RWWPO_REPO_DIR/tools/h20/preflight_rwwpo2_babilong.py" \
    --checkpoint "$RWWPO_BABILONG_CHECKPOINT" \
    --validation "$RWWPO_BABILONG_DATA" --model "$RWWPO_BABILONG_MODEL" \
    --resolved-manifest "$RWWPO_BABILONG_RESOLVED" \
    --resolved-manifest-sha256 "$RWWPO_BABILONG_RESOLVED_SHA256" \
    --eval-root "$RWWPO_BABILONG_EVAL_ROOT" \
    --interface-id "$RWWPO_BABILONG_INTERFACE_ID" \
    --attempt-id "$RWWPO_BABILONG_ATTEMPT_ID" \
    --expected-commit "$RWWPO_EXPECTED_COMMIT"
)
[[ ${#OVERRIDES[@]} -gt 0 ]] || { echo 'RWWPO2_BABILONG_EVAL_NO_GO:empty argv' >&2; exit 66; }
unset GATE_A_FROZEN_AUDIT GATE_A_EXECUTION_LEDGER RAY_ADDRESS
export CUDA_VISIBLE_DEVICES=$GPU_PAIR
export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
export GATE_A_WEIGHT_DIGEST_SAMPLES=256
RAY_TMP=/tmp/mr2b_${UID}_$$
mkdir -p "$RAY_TMP"
export RAY_TMPDIR=$RAY_TMP TMPDIR=$RAY_TMP HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline VLLM_USE_V1=0 NCCL_DEBUG=WARN
cd "$RWWPO_REPO_DIR"
"$RWWPO_WORK_ROOT/.venv/bin/python" -m verl.trainer.main_ppo \
  "${OVERRIDES[@]}" 2>&1 | tee -a "$RWWPO_BABILONG_EVAL_ROOT/run.log"

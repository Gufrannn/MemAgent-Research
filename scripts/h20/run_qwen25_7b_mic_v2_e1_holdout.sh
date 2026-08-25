#!/usr/bin/env bash
set -euo pipefail

for name in MEMAGENT_MIC_V2_WORK_ROOT MEMAGENT_MIC_V2_REPO_DIR \
  MEMAGENT_MIC_V2_EXPECTED_COMMIT MEMAGENT_MIC_V2_E1_DEV_RUN_ID \
  MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID MEMAGENT_MIC_V2_GPU_PAIR
do
  [[ -n "${!name:-}" ]] || { echo "MIC_V2_E1_NO_GO: missing $name" >&2; exit 60; }
done
[[ "$MEMAGENT_MIC_V2_WORK_ROOT" = /* && "$MEMAGENT_MIC_V2_REPO_DIR" = /* ]] || exit 73
[[ "$MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || exit 61
[[ "$MEMAGENT_MIC_V2_GPU_PAIR" =~ ^([0-9]+),([0-9]+)$ ]] || exit 62
GPU_LEFT="${BASH_REMATCH[1]}"; GPU_RIGHT="${BASH_REMATCH[2]}"
(( GPU_LEFT < GPU_RIGHT )) || exit 63
[[ -z "${PYTHONOPTIMIZE:-}" ]] || exit 64
readonly PY="$MEMAGENT_MIC_V2_WORK_ROOT/.venv/bin/python"
readonly DEV_ROOT="$MEMAGENT_MIC_V2_WORK_ROOT/logs/mic_v2_e1/$MEMAGENT_MIC_V2_E1_DEV_RUN_ID"
readonly ROOT="$MEMAGENT_MIC_V2_WORK_ROOT/logs/mic_v2_e1/$MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID"
readonly LOCKS="$MEMAGENT_MIC_V2_WORK_ROOT/locks"
test -x "$PY" || exit 74
test "$(git -C "$MEMAGENT_MIC_V2_REPO_DIR" rev-parse HEAD)" = \
  "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" || exit 65
test -z "$(git -C "$MEMAGENT_MIC_V2_REPO_DIR" status --porcelain)" || exit 66
test -f "$DEV_ROOT/certificates/e1_dev_selection.json" || exit 67
[[ ! -e "$ROOT" ]] || { echo "MIC_V2_E1_NO_GO: holdout attempt already exists" >&2; exit 68; }
mkdir -p "$LOCKS"
exec 7>"$LOCKS/memagent_mic_v2_e1_${MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID}.lock"; flock -n 7 || exit 69
exec 8>"$LOCKS/memagent_h20_gpu_${GPU_LEFT}.lock"; flock -n 8 || exit 70
exec 9>"$LOCKS/memagent_h20_gpu_${GPU_RIGHT}.lock"; flock -n 9 || exit 71
[[ -z "$(nvidia-smi -i "$MEMAGENT_MIC_V2_GPU_PAIR" --query-compute-apps=pid \
  --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]] || exit 72
cd "$MEMAGENT_MIC_V2_REPO_DIR"
export MEMAGENT_MIC_V2_E1_OFFICIAL_ENTRY=locked-shell-v1
export MEMAGENT_MIC_V2_E1_LOCK_RUN_ID="$MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID"
export MEMAGENT_MIC_V2_E1_LOCK_WORK_ROOT="$MEMAGENT_MIC_V2_WORK_ROOT"
export MEMAGENT_MIC_V2_E1_LOCK_GPU_PAIR="$MEMAGENT_MIC_V2_GPU_PAIR"
export MEMAGENT_MIC_V2_E1_LOCK_FDS=7,8,9
export MEMAGENT_MIC_V2_E1_LOCK_PATH_7="$LOCKS/memagent_mic_v2_e1_${MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID}.lock"
export MEMAGENT_MIC_V2_E1_LOCK_PATH_8="$LOCKS/memagent_h20_gpu_${GPU_LEFT}.lock"
export MEMAGENT_MIC_V2_E1_LOCK_PATH_9="$LOCKS/memagent_h20_gpu_${GPU_RIGHT}.lock"
export VLLM_USE_MODELSCOPE=False CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$PY" \
  tools/h20/mic_v2_e1_pipeline.py preflight-holdout \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --work-root "$MEMAGENT_MIC_V2_WORK_ROOT" --output-root "$ROOT" \
  --run-id "$MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID" --gpu-pair "$MEMAGENT_MIC_V2_GPU_PAIR" \
  --dev-root "$DEV_ROOT"
export CUDA_VISIBLE_DEVICES="$MEMAGENT_MIC_V2_GPU_PAIR"
for mode in produce replay; do
  PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$PY" \
    tools/h20/run_qwen25_7b_mic_v2_e1_collect.py \
    --repo "$MEMAGENT_MIC_V2_REPO_DIR" --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
    --output-root "$ROOT" --run-id "$MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID" --mode "$mode" \
    2>&1 | tee -a "$ROOT/e1_holdout.log"
done
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$PY" \
  tools/h20/mic_v2_e1_pipeline.py seal-holdout-states \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --work-root "$MEMAGENT_MIC_V2_WORK_ROOT" --output-root "$ROOT" \
  --run-id "$MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID" --gpu-pair "$MEMAGENT_MIC_V2_GPU_PAIR"
export CUDA_VISIBLE_DEVICES="$MEMAGENT_MIC_V2_GPU_PAIR"
for mode in produce replay; do
  PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$PY" \
    tools/h20/run_qwen25_7b_mic_v2_e1_features.py \
    --repo "$MEMAGENT_MIC_V2_REPO_DIR" --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
    --output-root "$ROOT" --run-id "$MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID" --split e1_holdout \
    --mode "$mode" 2>&1 | tee -a "$ROOT/e1_holdout.log"
done
for command in score-holdout evaluate-holdout verify-holdout; do
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$PY" \
    tools/h20/mic_v2_e1_pipeline.py "$command" \
    --repo "$MEMAGENT_MIC_V2_REPO_DIR" --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
    --work-root "$MEMAGENT_MIC_V2_WORK_ROOT" --output-root "$ROOT" \
    --run-id "$MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID" --gpu-pair "$MEMAGENT_MIC_V2_GPU_PAIR"
done
echo "MIC_V2_E1_HOLDOUT_PASS"

#!/usr/bin/env bash
set -euo pipefail

for name in MEMAGENT_MIC_V2_WORK_ROOT MEMAGENT_MIC_V2_REPO_DIR \
  MEMAGENT_MIC_V2_EXPECTED_COMMIT MEMAGENT_MIC_V2_E1_DEV_RUN_ID \
  MEMAGENT_MIC_V2_GPU_PAIR
do
  [[ -n "${!name:-}" ]] || { echo "MIC_V2_E1_NO_GO: missing $name" >&2; exit 40; }
done
[[ "$MEMAGENT_MIC_V2_WORK_ROOT" = /* && "$MEMAGENT_MIC_V2_REPO_DIR" = /* ]] || exit 41
[[ "$MEMAGENT_MIC_V2_E1_DEV_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || exit 42
[[ "$MEMAGENT_MIC_V2_GPU_PAIR" =~ ^([0-9]+),([0-9]+)$ ]] || exit 43
GPU_LEFT="${BASH_REMATCH[1]}"; GPU_RIGHT="${BASH_REMATCH[2]}"
(( GPU_LEFT < GPU_RIGHT )) || exit 44
[[ -z "${PYTHONOPTIMIZE:-}" ]] || exit 45

readonly PY="$MEMAGENT_MIC_V2_WORK_ROOT/.venv/bin/python"
readonly ROOT="$MEMAGENT_MIC_V2_WORK_ROOT/logs/mic_v2_e1/$MEMAGENT_MIC_V2_E1_DEV_RUN_ID"
readonly LOCKS="$MEMAGENT_MIC_V2_WORK_ROOT/locks"
test -x "$PY"
test "$(git -C "$MEMAGENT_MIC_V2_REPO_DIR" rev-parse HEAD)" = \
  "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" || exit 46
test -z "$(git -C "$MEMAGENT_MIC_V2_REPO_DIR" status --porcelain)" || exit 47
[[ ! -e "$ROOT" ]] || { echo "MIC_V2_E1_NO_GO: dev attempt already exists" >&2; exit 48; }
mkdir -p "$LOCKS"
exec 7>"$LOCKS/memagent_mic_v2_e1_${MEMAGENT_MIC_V2_E1_DEV_RUN_ID}.lock"
flock -n 7 || exit 49
exec 8>"$LOCKS/memagent_h20_gpu_${GPU_LEFT}.lock"
flock -n 8 || exit 50
exec 9>"$LOCKS/memagent_h20_gpu_${GPU_RIGHT}.lock"
flock -n 9 || exit 51
[[ -z "$(nvidia-smi -i "$MEMAGENT_MIC_V2_GPU_PAIR" --query-compute-apps=pid \
  --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]] || exit 52

cd "$MEMAGENT_MIC_V2_REPO_DIR"
export MEMAGENT_MIC_V2_E1_OFFICIAL_ENTRY=locked-shell-v1
export MEMAGENT_MIC_V2_E1_LOCK_RUN_ID="$MEMAGENT_MIC_V2_E1_DEV_RUN_ID"
export MEMAGENT_MIC_V2_E1_LOCK_WORK_ROOT="$MEMAGENT_MIC_V2_WORK_ROOT"
export MEMAGENT_MIC_V2_E1_LOCK_GPU_PAIR="$MEMAGENT_MIC_V2_GPU_PAIR"
export MEMAGENT_MIC_V2_E1_LOCK_FDS=7,8,9
export MEMAGENT_MIC_V2_E1_LOCK_PATH_7="$LOCKS/memagent_mic_v2_e1_${MEMAGENT_MIC_V2_E1_DEV_RUN_ID}.lock"
export MEMAGENT_MIC_V2_E1_LOCK_PATH_8="$LOCKS/memagent_h20_gpu_${GPU_LEFT}.lock"
export MEMAGENT_MIC_V2_E1_LOCK_PATH_9="$LOCKS/memagent_h20_gpu_${GPU_RIGHT}.lock"
export VLLM_USE_MODELSCOPE=False
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$PY" -m py_compile \
  recurrent/research/mic_v2_e1.py tools/h20/mic_v2_e1_pipeline.py \
  tools/h20/run_qwen25_7b_mic_v2_e1_collect.py \
  tools/h20/run_qwen25_7b_mic_v2_e1_features.py

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$PY" \
  tools/h20/mic_v2_e1_pipeline.py preflight-dev \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --work-root "$MEMAGENT_MIC_V2_WORK_ROOT" --output-root "$ROOT" \
  --run-id "$MEMAGENT_MIC_V2_E1_DEV_RUN_ID" --gpu-pair "$MEMAGENT_MIC_V2_GPU_PAIR"

export CUDA_VISIBLE_DEVICES="$MEMAGENT_MIC_V2_GPU_PAIR"
for mode in produce replay; do
  PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$PY" \
    tools/h20/run_qwen25_7b_mic_v2_e1_collect.py \
    --repo "$MEMAGENT_MIC_V2_REPO_DIR" --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
    --output-root "$ROOT" --run-id "$MEMAGENT_MIC_V2_E1_DEV_RUN_ID" --mode "$mode" \
    2>&1 | tee -a "$ROOT/e1_dev.log"
done

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$PY" \
  tools/h20/mic_v2_e1_pipeline.py seal-dev-states \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --work-root "$MEMAGENT_MIC_V2_WORK_ROOT" --output-root "$ROOT" \
  --run-id "$MEMAGENT_MIC_V2_E1_DEV_RUN_ID" --gpu-pair "$MEMAGENT_MIC_V2_GPU_PAIR"

export CUDA_VISIBLE_DEVICES="$MEMAGENT_MIC_V2_GPU_PAIR"
for mode in produce replay; do
  PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$PY" \
    tools/h20/run_qwen25_7b_mic_v2_e1_features.py \
    --repo "$MEMAGENT_MIC_V2_REPO_DIR" --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
    --output-root "$ROOT" --run-id "$MEMAGENT_MIC_V2_E1_DEV_RUN_ID" --split e1_dev \
    --mode "$mode" 2>&1 | tee -a "$ROOT/e1_dev.log"
done

for command in score-dev select-dev verify-dev; do
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$PY" \
    tools/h20/mic_v2_e1_pipeline.py "$command" \
    --repo "$MEMAGENT_MIC_V2_REPO_DIR" --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
    --work-root "$MEMAGENT_MIC_V2_WORK_ROOT" --output-root "$ROOT" \
    --run-id "$MEMAGENT_MIC_V2_E1_DEV_RUN_ID" --gpu-pair "$MEMAGENT_MIC_V2_GPU_PAIR"
done
echo "MIC_V2_E1_DEV_SELECTION_PASS"

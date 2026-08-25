#!/usr/bin/env bash
set -euo pipefail

for name in \
  MEMAGENT_MIC_V2_WORK_ROOT \
  MEMAGENT_MIC_V2_REPO_DIR \
  MEMAGENT_MIC_V2_EXPECTED_COMMIT \
  MEMAGENT_MIC_V2_CALIBRATION_RUN_ID \
  MEMAGENT_MIC_V2_GPU_PAIR
do
  if [[ -z "${!name:-}" ]]; then
    echo "MIC_V2_NO_GO: missing required environment $name" >&2
    exit 40
  fi
done

if [[ "$MEMAGENT_MIC_V2_WORK_ROOT" != /* || "$MEMAGENT_MIC_V2_REPO_DIR" != /* ]]; then
  echo "MIC_V2_NO_GO: work/repository roots must be absolute" >&2
  exit 41
fi
if [[ ! "$MEMAGENT_MIC_V2_CALIBRATION_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "MIC_V2_NO_GO: unsafe calibration run ID" >&2
  exit 42
fi
if [[ ! "$MEMAGENT_MIC_V2_GPU_PAIR" =~ ^([0-9]+),([0-9]+)$ ]]; then
  echo "MIC_V2_NO_GO: GPU pair must be explicit N,M" >&2
  exit 43
fi
GPU_LEFT="${BASH_REMATCH[1]}"
GPU_RIGHT="${BASH_REMATCH[2]}"
if (( GPU_LEFT >= GPU_RIGHT )); then
  echo "MIC_V2_NO_GO: GPU pair must be unique and canonical ascending" >&2
  exit 44
fi
if [[ -n "${PYTHONOPTIMIZE:-}" ]]; then
  echo "MIC_V2_NO_GO: PYTHONOPTIMIZE is forbidden" >&2
  exit 45
fi

readonly MIC_V2_PYTHON="$MEMAGENT_MIC_V2_WORK_ROOT/.venv/bin/python"
readonly OUTPUT_ROOT="$MEMAGENT_MIC_V2_WORK_ROOT/logs/mic_v2_reference_length/$MEMAGENT_MIC_V2_CALIBRATION_RUN_ID"
readonly LOG_ROOT="$MEMAGENT_MIC_V2_WORK_ROOT/logs/mic_v2_reference_length"
readonly LOCK_ROOT="$MEMAGENT_MIC_V2_WORK_ROOT/locks"
readonly RUN_LOCK="$LOCK_ROOT/memagent_mic_v2_${MEMAGENT_MIC_V2_CALIBRATION_RUN_ID}.lock"
readonly LEFT_LOCK="$LOCK_ROOT/memagent_h20_gpu_${GPU_LEFT}.lock"
readonly RIGHT_LOCK="$LOCK_ROOT/memagent_h20_gpu_${GPU_RIGHT}.lock"

test -x "$MIC_V2_PYTHON"
test "$(git -C "$MEMAGENT_MIC_V2_REPO_DIR" rev-parse HEAD)" = \
  "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" || {
    echo "MIC_V2_NO_GO: exact Git commit mismatch" >&2
    exit 46
  }
test -z "$(git -C "$MEMAGENT_MIC_V2_REPO_DIR" status --porcelain)" || {
  echo "MIC_V2_NO_GO: worktree is dirty" >&2
  exit 47
}

mkdir -p "$LOG_ROOT" "$LOCK_ROOT"
exec 7>"$RUN_LOCK"
flock -n 7 || {
  echo "MIC_V2_NO_GO: calibration run lock conflict $RUN_LOCK" >&2
  exit 48
}

if [[ -e "$OUTPUT_ROOT" ]]; then
  if [[ "${MEMAGENT_MIC_V2_CALIBRATION_RESUME:-0}" != 1 ]]; then
    echo "MIC_V2_NO_GO: attempt exists; set MEMAGENT_MIC_V2_CALIBRATION_RESUME=1 to resume" >&2
    exit 49
  fi
  if [[ -e "$OUTPUT_ROOT/certificates/reference_length.json" ]]; then
    echo "MIC_V2_NO_GO: completed calibration is immutable" >&2
    exit 50
  fi
else
  mkdir "$OUTPUT_ROOT"
  mkdir "$OUTPUT_ROOT/authorities" "$OUTPUT_ROOT/certificates" "$OUTPUT_ROOT/trajectories"
fi

cd "$MEMAGENT_MIC_V2_REPO_DIR"
export VLLM_USE_MODELSCOPE=False
PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" -m py_compile \
  tools/h20/mic_v2_reference_length_calibration.py \
  tools/h20/run_qwen25_7b_mic_v2_reference_length_calibration.py

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" \
  tools/h20/mic_v2_reference_length_calibration.py preflight \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" \
  --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$MEMAGENT_MIC_V2_CALIBRATION_RUN_ID" \
  --gpu-pair "$MEMAGENT_MIC_V2_GPU_PAIR"

exec 8>"$LEFT_LOCK"
flock -n 8 || {
  echo "MIC_V2_NO_GO: lock conflict $LEFT_LOCK" >&2
  exit 51
}
exec 9>"$RIGHT_LOCK"
flock -n 9 || {
  echo "MIC_V2_NO_GO: lock conflict $RIGHT_LOCK" >&2
  exit 52
}

GPU_PROCESSES="$(nvidia-smi -i "$MEMAGENT_MIC_V2_GPU_PAIR" \
  --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "$GPU_PROCESSES" ]]; then
  echo "MIC_V2_NO_GO: selected GPU pair is occupied; no process was killed" >&2
  printf '%s\n' "$GPU_PROCESSES" >&2
  exit 53
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$MEMAGENT_MIC_V2_GPU_PAIR"
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

if [[ ! -e "$OUTPUT_ROOT/certificates/execution.json" ]]; then
  PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" \
    tools/h20/run_qwen25_7b_mic_v2_reference_length_calibration.py \
    --repo "$MEMAGENT_MIC_V2_REPO_DIR" \
    --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
    --output-root "$OUTPUT_ROOT" \
    --run-id "$MEMAGENT_MIC_V2_CALIBRATION_RUN_ID" \
    --mode produce 2>&1 | \
    tee -a "$OUTPUT_ROOT/calibration.log"
fi

PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" \
  tools/h20/run_qwen25_7b_mic_v2_reference_length_calibration.py \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" \
  --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$MEMAGENT_MIC_V2_CALIBRATION_RUN_ID" \
  --mode replay 2>&1 | \
  tee -a "$OUTPUT_ROOT/calibration.log"

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" \
  tools/h20/mic_v2_reference_length_calibration.py finalize \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" \
  --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$MEMAGENT_MIC_V2_CALIBRATION_RUN_ID"

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" \
  tools/h20/mic_v2_reference_length_calibration.py verify \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" \
  --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$MEMAGENT_MIC_V2_CALIBRATION_RUN_ID"

echo "MIC_V2_REFERENCE_LENGTH_CALIBRATION_PASS"

#!/usr/bin/env bash
set -euo pipefail

[[ -n ${MEMAGENT_ORIGINAL_CURVE_WORK_ROOT:-} ]] || {
  echo 'ORIGINAL_S128_CURVE_NO_GO:P0 set MEMAGENT_ORIGINAL_CURVE_WORK_ROOT explicitly' >&2; exit 66;
}
[[ -n ${MEMAGENT_ORIGINAL_CURVE_REPO_DIR:-} ]] || {
  echo 'ORIGINAL_S128_CURVE_NO_GO:P0 set MEMAGENT_ORIGINAL_CURVE_REPO_DIR explicitly' >&2; exit 67;
}
[[ ${MEMAGENT_ORIGINAL_CURVE_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'ORIGINAL_S128_CURVE_NO_GO:P0 expected commit must be a full Git SHA' >&2; exit 65;
}
[[ -n ${MEMAGENT_ORIGINAL_CURVE_GPU_PAIR:-} ]] || {
  echo 'ORIGINAL_S128_CURVE_NO_GO:P0 set MEMAGENT_ORIGINAL_CURVE_GPU_PAIR explicitly' >&2; exit 83;
}
[[ $MEMAGENT_ORIGINAL_CURVE_GPU_PAIR =~ ^(0|[1-9]|[12][0-9]|3[01]),(0|[1-9]|[12][0-9]|3[01])$ ]] || {
  echo 'ORIGINAL_S128_CURVE_NO_GO:P0 GPU pair must be canonical decimal A,B' >&2; exit 84;
}
ORIGINAL_CURVE_GPU_A=${BASH_REMATCH[1]}
ORIGINAL_CURVE_GPU_B=${BASH_REMATCH[2]}
(( ORIGINAL_CURVE_GPU_A < ORIGINAL_CURVE_GPU_B )) || {
  echo 'ORIGINAL_S128_CURVE_NO_GO:P0 GPU pair must be distinct and ascending' >&2; exit 86;
}
[[ $MEMAGENT_ORIGINAL_CURVE_WORK_ROOT == /* && $MEMAGENT_ORIGINAL_CURVE_REPO_DIR == /* ]] || {
  echo 'ORIGINAL_S128_CURVE_NO_GO:P0 task-scoped paths must be absolute' >&2; exit 69;
}

readonly ORIGINAL_CURVE_WORK_ROOT=$MEMAGENT_ORIGINAL_CURVE_WORK_ROOT
readonly ORIGINAL_CURVE_REPO_DIR=$MEMAGENT_ORIGINAL_CURVE_REPO_DIR
readonly ORIGINAL_CURVE_EXPECTED_COMMIT=$MEMAGENT_ORIGINAL_CURVE_EXPECTED_COMMIT
readonly ORIGINAL_CURVE_PYTHON=$ORIGINAL_CURVE_WORK_ROOT/.venv/bin/python
readonly ORIGINAL_CURVE_MANIFEST=$ORIGINAL_CURVE_REPO_DIR/manifests/h20/qwen25_7b_original_s128_curve_seed2026.json
readonly ORIGINAL_CURVE_LOG_ROOT=$ORIGINAL_CURVE_WORK_ROOT/logs/s128_original_all_anchor_frozen_20260821
readonly ORIGINAL_CURVE_P0=$ORIGINAL_CURVE_LOG_ROOT/certificates/p0_preflight.json
readonly ORIGINAL_CURVE_RESOLVED=$ORIGINAL_CURVE_LOG_ROOT/certificates/p0_resolved_manifest.json
readonly ORIGINAL_CURVE_LEDGER=$ORIGINAL_CURVE_LOG_ROOT/original_s128_curve_execution_ledger.jsonl
readonly ORIGINAL_CURVE_GPUS=$MEMAGENT_ORIGINAL_CURVE_GPU_PAIR
readonly ORIGINAL_CURVE_GPU_A ORIGINAL_CURVE_GPU_B
readonly ORIGINAL_CURVE_GPU_LOCK_A=$ORIGINAL_CURVE_WORK_ROOT/locks/memagent_h20_gpu_${ORIGINAL_CURVE_GPU_A}.lock
readonly ORIGINAL_CURVE_GPU_LOCK_B=$ORIGINAL_CURVE_WORK_ROOT/locks/memagent_h20_gpu_${ORIGINAL_CURVE_GPU_B}.lock

original_curve_require_checkout() {
  local invoked_repo
  invoked_repo=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
  [[ $(cd -- "$ORIGINAL_CURVE_REPO_DIR" && pwd -P) == "$invoked_repo" ]] || {
    echo 'ORIGINAL_S128_CURVE_NO_GO:P0 invoked checkout differs from explicit repository' >&2; exit 68;
  }
  [[ $(cd "$ORIGINAL_CURVE_REPO_DIR" && git branch --show-current) == h20/qwen25-7b-original-all-anchor-s128-frozen-20260821 ]] || {
    echo 'ORIGINAL_S128_CURVE_NO_GO:P0 wrong branch' >&2; exit 70;
  }
  [[ -z $(cd "$ORIGINAL_CURVE_REPO_DIR" && git status --porcelain) ]] || {
    echo 'ORIGINAL_S128_CURVE_NO_GO:P0 dirty worktree' >&2; exit 71;
  }
  [[ $(cd "$ORIGINAL_CURVE_REPO_DIR" && git rev-parse HEAD) == "$ORIGINAL_CURVE_EXPECTED_COMMIT" ]] || {
    echo 'ORIGINAL_S128_CURVE_NO_GO:P0 HEAD differs from expected commit' >&2; exit 64;
  }
}

original_curve_acquire_lock() {
  command -v flock >/dev/null || {
    echo 'ORIGINAL_S128_CURVE_NO_GO:P0 flock is required' >&2; exit 63;
  }
  # The pair is canonical ascending, so every runner migrated to this dynamic
  # pair contract acquires overlapping physical-GPU locks in one global order
  # and cannot deadlock. Legacy fixed-GPU launchers do not implement this
  # contract and must not be launched concurrently.
  mkdir -p "$(dirname "$ORIGINAL_CURVE_GPU_LOCK_A")"
  exec 8>"$ORIGINAL_CURVE_GPU_LOCK_A"
  flock -n 8 || {
    echo "ORIGINAL_S128_CURVE_NO_GO:P0 physical GPU $ORIGINAL_CURVE_GPU_A lock is held" >&2; exit 62;
  }
  exec 9>"$ORIGINAL_CURVE_GPU_LOCK_B"
  flock -n 9 || {
    echo "ORIGINAL_S128_CURVE_NO_GO:P0 physical GPU $ORIGINAL_CURVE_GPU_B lock is held" >&2; exit 62;
  }
}

original_curve_require_idle() {
  command -v nvidia-smi >/dev/null || {
    echo 'ORIGINAL_S128_CURVE_NO_GO:P0 nvidia-smi is required' >&2; exit 78;
  }
  local processes
  processes=$(nvidia-smi -i "$ORIGINAL_CURVE_GPUS" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${processes//[[:space:]]/} ]] || {
    echo "ORIGINAL_S128_CURVE_NO_GO:P0 GPUs $ORIGINAL_CURVE_GPUS are busy; no process was changed: $processes" >&2; exit 79;
  }
}

original_curve_wait_cleanup() {
  local processes poll
  for poll in $(seq 1 45); do
    processes=$(nvidia-smi -i "$ORIGINAL_CURVE_GPUS" --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z ${processes//[[:space:]]/} ]] && return 0
    sleep 2
  done
  echo "ORIGINAL_S128_CURVE_NO_GO:CLEANUP GPUs $ORIGINAL_CURVE_GPUS did not become idle: $processes" >&2
  exit 81
}

original_curve_require_p0() {
  [[ -f $ORIGINAL_CURVE_P0 && -f $ORIGINAL_CURVE_RESOLVED && -f $ORIGINAL_CURVE_LEDGER ]] || {
    echo 'ORIGINAL_S128_CURVE_NO_GO:P0 run standalone P0 before curve evaluation' >&2; exit 61;
  }
  "$ORIGINAL_CURVE_PYTHON" \
    "$ORIGINAL_CURVE_REPO_DIR/tools/h20/preflight_qwen25_7b_original_s128_curve.py" \
    --manifest "$ORIGINAL_CURVE_MANIFEST" --interface-mode --interface I >/dev/null || {
      echo 'ORIGINAL_S128_CURVE_NO_GO:P0 certificate prefix is not authenticated' >&2; exit 80;
    }
}

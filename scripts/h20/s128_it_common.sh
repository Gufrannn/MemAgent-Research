#!/usr/bin/env bash
set -euo pipefail

[[ -n ${MEMAGENT_S128_IT_WORK_ROOT:-} ]] || {
  echo 'S128_IT_NO_GO:P0 set MEMAGENT_S128_IT_WORK_ROOT explicitly' >&2; exit 66;
}
[[ -n ${MEMAGENT_S128_IT_REPO_DIR:-} ]] || {
  echo 'S128_IT_NO_GO:P0 set MEMAGENT_S128_IT_REPO_DIR explicitly' >&2; exit 67;
}
[[ ${MEMAGENT_S128_IT_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'S128_IT_NO_GO:P0 expected commit must be a full Git SHA' >&2; exit 65;
}
[[ $MEMAGENT_S128_IT_WORK_ROOT == /* && $MEMAGENT_S128_IT_REPO_DIR == /* ]] || {
  echo 'S128_IT_NO_GO:P0 task-scoped paths must be absolute' >&2; exit 69;
}

readonly S128_IT_WORK_ROOT=$MEMAGENT_S128_IT_WORK_ROOT
readonly S128_IT_REPO_DIR=$MEMAGENT_S128_IT_REPO_DIR
readonly S128_IT_EXPECTED_COMMIT=$MEMAGENT_S128_IT_EXPECTED_COMMIT
readonly S128_IT_PYTHON=$S128_IT_WORK_ROOT/.venv/bin/python
readonly S128_IT_MANIFEST=$S128_IT_REPO_DIR/manifests/h20/qwen25_7b_s128_it_seed2026.json
readonly S128_IT_LOG_ROOT=$S128_IT_WORK_ROOT/logs/s128_it_original_t25_frozen_20260821
readonly S128_IT_P0=$S128_IT_LOG_ROOT/certificates/p0_preflight.json
readonly S128_IT_RESOLVED=$S128_IT_LOG_ROOT/certificates/p0_resolved_manifest.json
readonly S128_IT_LEDGER=$S128_IT_LOG_ROOT/s128_it_execution_ledger.jsonl
readonly S128_IT_GPUS=6,7
readonly S128_IT_LOCK=$S128_IT_WORK_ROOT/locks/memagent_gate_a_gpu_6_7.lock

s128_it_require_checkout() {
  local invoked_repo
  invoked_repo=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
  [[ $(cd -- "$S128_IT_REPO_DIR" && pwd -P) == "$invoked_repo" ]] || {
    echo 'S128_IT_NO_GO:P0 invoked checkout differs from explicit repository' >&2; exit 68;
  }
  [[ $(cd "$S128_IT_REPO_DIR" && git branch --show-current) == h20/qwen25-7b-original-t25-s128-frozen-20260821 ]] || {
    echo 'S128_IT_NO_GO:P0 wrong branch' >&2; exit 70;
  }
  [[ -z $(cd "$S128_IT_REPO_DIR" && git status --porcelain) ]] || {
    echo 'S128_IT_NO_GO:P0 dirty worktree' >&2; exit 71;
  }
  [[ $(cd "$S128_IT_REPO_DIR" && git rev-parse HEAD) == "$S128_IT_EXPECTED_COMMIT" ]] || {
    echo 'S128_IT_NO_GO:P0 HEAD differs from expected commit' >&2; exit 64;
  }
}

s128_it_acquire_lock() {
  command -v flock >/dev/null || { echo 'S128_IT_NO_GO:P0 flock is required' >&2; exit 63; }
  mkdir -p "$(dirname "$S128_IT_LOCK")"
  exec 8>"$S128_IT_LOCK"
  flock -n 8 || { echo 'S128_IT_NO_GO:P0 GPU6-7 project lock is held' >&2; exit 62; }
}

s128_it_require_idle() {
  command -v nvidia-smi >/dev/null || { echo 'S128_IT_NO_GO:P0 nvidia-smi is required' >&2; exit 78; }
  local processes
  processes=$(nvidia-smi -i "$S128_IT_GPUS" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${processes//[[:space:]]/} ]] || {
    echo "S128_IT_NO_GO:P0 GPU6-7 are busy; no process was changed: $processes" >&2; exit 79;
  }
}

s128_it_wait_cleanup() {
  local processes poll
  for poll in $(seq 1 45); do
    processes=$(nvidia-smi -i "$S128_IT_GPUS" --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z ${processes//[[:space:]]/} ]] && return 0
    sleep 2
  done
  echo "S128_IT_NO_GO:CLEANUP GPU6-7 did not become idle: $processes" >&2
  exit 81
}

s128_it_require_p0() {
  [[ -f $S128_IT_P0 && -f $S128_IT_RESOLVED && -f $S128_IT_LEDGER ]] || {
    echo 'S128_IT_NO_GO:P0 run standalone P0 before I/T evaluation' >&2; exit 61;
  }
  "$S128_IT_PYTHON" "$S128_IT_REPO_DIR/tools/h20/preflight_qwen25_7b_s128_it.py" \
    --manifest "$S128_IT_MANIFEST" --validate-p0-prefix || {
      echo 'S128_IT_NO_GO:P0 certificate prefix is not fully authenticated' >&2; exit 80;
    }
}

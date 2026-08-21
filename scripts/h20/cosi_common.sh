#!/usr/bin/env bash
set -euo pipefail
readonly COSI_BRANCH=h20/qwen25-7b-cosi-t25-frozen-20260822
[[ ${MEMAGENT_COSI_WORK_ROOT:-} == /* && ${MEMAGENT_COSI_REPO_DIR:-} == /* ]] || { echo COSI_NO_GO:explicit_absolute_paths >&2; exit 60; }
[[ ${MEMAGENT_COSI_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || { echo COSI_NO_GO:exact_commit >&2; exit 61; }
[[ ${MEMAGENT_COSI_GPU_PAIR:-} =~ ^[0-9]+,[0-9]+$ ]] || { echo COSI_NO_GO:gpu_pair >&2; exit 62; }
IFS=, read -r COSI_GPU_A COSI_GPU_B <<< "$MEMAGENT_COSI_GPU_PAIR"
(( COSI_GPU_A < COSI_GPU_B )) || { echo COSI_NO_GO:gpu_pair_not_canonical >&2; exit 63; }
cosi_checkout_guard() {
  [[ $(cd "$MEMAGENT_COSI_REPO_DIR" && git branch --show-current) == "$COSI_BRANCH" ]] || { echo COSI_NO_GO:wrong_branch >&2; exit 64; }
  [[ $(cd "$MEMAGENT_COSI_REPO_DIR" && git rev-parse HEAD) == "$MEMAGENT_COSI_EXPECTED_COMMIT" ]] || { echo COSI_NO_GO:wrong_commit >&2; exit 65; }
  [[ -z $(cd "$MEMAGENT_COSI_REPO_DIR" && git status --porcelain) ]] || { echo COSI_NO_GO:dirty_tree >&2; exit 66; }
}
cosi_acquire_gpu_locks() {
  command -v flock >/dev/null || { echo COSI_NO_GO:flock_missing >&2; exit 67; }
  mkdir -p "$MEMAGENT_COSI_WORK_ROOT/locks"
  exec 8>"$MEMAGENT_COSI_WORK_ROOT/locks/memagent_h20_gpu_${COSI_GPU_A}.lock"
  exec 9>"$MEMAGENT_COSI_WORK_ROOT/locks/memagent_h20_gpu_${COSI_GPU_B}.lock"
  flock -n 8 || { echo COSI_NO_GO:gpu_lock_conflict_${COSI_GPU_A} >&2; exit 68; }
  flock -n 9 || { echo COSI_NO_GO:gpu_lock_conflict_${COSI_GPU_B} >&2; exit 69; }
  local active
  active=$(nvidia-smi -i "$MEMAGENT_COSI_GPU_PAIR" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${active//[[:space:]]/} ]] || { echo "COSI_NO_GO:gpu_busy:$active" >&2; exit 70; }
  export CUDA_VISIBLE_DEVICES=$MEMAGENT_COSI_GPU_PAIR
}

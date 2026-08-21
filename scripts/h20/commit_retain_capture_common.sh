#!/usr/bin/env bash
set -euo pipefail

[[ -n ${MEMAGENT_COMMIT_RETAIN_WORK_ROOT:-} ]] || {
  echo 'COMMIT_RETAIN_NO_GO:P0 set MEMAGENT_COMMIT_RETAIN_WORK_ROOT explicitly' >&2; exit 66;
}
[[ -n ${MEMAGENT_COMMIT_RETAIN_REPO_DIR:-} ]] || {
  echo 'COMMIT_RETAIN_NO_GO:P0 set MEMAGENT_COMMIT_RETAIN_REPO_DIR explicitly' >&2; exit 67;
}
[[ ${MEMAGENT_COMMIT_RETAIN_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'COMMIT_RETAIN_NO_GO:P0 expected commit must be a full Git SHA' >&2; exit 65;
}
[[ ${MEMAGENT_COMMIT_RETAIN_RUN_ID:-} =~ ^[a-z0-9][a-z0-9_-]{1,31}$ ]] || {
  echo 'COMMIT_RETAIN_NO_GO:P0 set a task-scoped run ID' >&2; exit 64;
}
[[ ${MEMAGENT_COMMIT_RETAIN_GPU_PAIR:-} =~ ^([0-9]+),([0-9]+)$ ]] || {
  echo 'COMMIT_RETAIN_NO_GO:P0 set MEMAGENT_COMMIT_RETAIN_GPU_PAIR=A,B explicitly' >&2; exit 59;
}
readonly COMMIT_RETAIN_GPU_FIRST=$((10#${BASH_REMATCH[1]}))
readonly COMMIT_RETAIN_GPU_SECOND=$((10#${BASH_REMATCH[2]}))
[[ $MEMAGENT_COMMIT_RETAIN_GPU_PAIR == "$COMMIT_RETAIN_GPU_FIRST,$COMMIT_RETAIN_GPU_SECOND" \
   && $COMMIT_RETAIN_GPU_FIRST -lt $COMMIT_RETAIN_GPU_SECOND ]] || {
  echo 'COMMIT_RETAIN_NO_GO:P0 GPU pair must be canonical ascending distinct A,B' >&2; exit 58;
}
[[ $MEMAGENT_COMMIT_RETAIN_WORK_ROOT == /* && $MEMAGENT_COMMIT_RETAIN_REPO_DIR == /* ]] || {
  echo 'COMMIT_RETAIN_NO_GO:P0 task-scoped paths must be absolute' >&2; exit 69;
}

readonly COMMIT_RETAIN_WORK_ROOT=$MEMAGENT_COMMIT_RETAIN_WORK_ROOT
readonly COMMIT_RETAIN_REPO_DIR=$MEMAGENT_COMMIT_RETAIN_REPO_DIR
readonly COMMIT_RETAIN_EXPECTED_COMMIT=$MEMAGENT_COMMIT_RETAIN_EXPECTED_COMMIT
readonly COMMIT_RETAIN_RUN_ID=$MEMAGENT_COMMIT_RETAIN_RUN_ID
readonly COMMIT_RETAIN_GPUS=$MEMAGENT_COMMIT_RETAIN_GPU_PAIR
readonly COMMIT_RETAIN_GPU_PAIR_SLUG=gpu${COMMIT_RETAIN_GPU_FIRST}_${COMMIT_RETAIN_GPU_SECOND}
readonly COMMIT_RETAIN_LOCK_FIRST=$COMMIT_RETAIN_WORK_ROOT/locks/memagent_gate_a_gpu_${COMMIT_RETAIN_GPU_FIRST}.lock
readonly COMMIT_RETAIN_LOCK_SECOND=$COMMIT_RETAIN_WORK_ROOT/locks/memagent_gate_a_gpu_${COMMIT_RETAIN_GPU_SECOND}.lock
readonly COMMIT_RETAIN_PYTHON=$COMMIT_RETAIN_WORK_ROOT/.venv/bin/python
readonly COMMIT_RETAIN_PROFILE=${COMMIT_RETAIN_CAPTURE_PROFILE:-gpu67}
case $COMMIT_RETAIN_PROFILE in
  gpu67)
    readonly COMMIT_RETAIN_BRANCH=h20/qwen25-7b-commit-retain-capture-20260821
    readonly COMMIT_RETAIN_MANIFEST=$COMMIT_RETAIN_REPO_DIR/manifests/h20/qwen25_7b_commit_retain_capture_seed2026.json
    readonly COMMIT_RETAIN_LOG_ROOT=$COMMIT_RETAIN_WORK_ROOT/logs/commit_retain_capture_frozen_20260821/${COMMIT_RETAIN_RUN_ID}_${COMMIT_RETAIN_GPU_PAIR_SLUG}
    ;;
  gpu45)
    readonly COMMIT_RETAIN_BRANCH=h20/qwen25-7b-commit-retain-capture-gpu45-20260821
    readonly COMMIT_RETAIN_MANIFEST=$COMMIT_RETAIN_REPO_DIR/manifests/h20/qwen25_7b_commit_retain_capture_gpu45_seed2026.json
    readonly COMMIT_RETAIN_LOG_ROOT=$COMMIT_RETAIN_WORK_ROOT/logs/commit_retain_capture_frozen_20260821/${COMMIT_RETAIN_RUN_ID}_${COMMIT_RETAIN_GPU_PAIR_SLUG}
    ;;
  *)
    echo "COMMIT_RETAIN_NO_GO:P0 unknown capture profile: $COMMIT_RETAIN_PROFILE" >&2; exit 60
    ;;
esac
readonly COMMIT_RETAIN_CERT_ROOT=$COMMIT_RETAIN_LOG_ROOT/certificates
readonly COMMIT_RETAIN_P0=$COMMIT_RETAIN_CERT_ROOT/p0_preflight.json
readonly COMMIT_RETAIN_RESOLVED=$COMMIT_RETAIN_CERT_ROOT/p0_resolved_manifest.json
readonly COMMIT_RETAIN_LEDGER=$COMMIT_RETAIN_LOG_ROOT/commit_retain_capture_execution_ledger.jsonl
readonly COMMIT_RETAIN_CREDENTIAL=$COMMIT_RETAIN_LOG_ROOT/credentials/capture_child.json
readonly COMMIT_RETAIN_CAPTURE=$COMMIT_RETAIN_LOG_ROOT/captures/commit_retain_pairs.jsonl
readonly COMMIT_RETAIN_RUN_RECEIPT=$COMMIT_RETAIN_LOG_ROOT/captures/run_receipt.json
readonly COMMIT_RETAIN_FINAL=$COMMIT_RETAIN_CERT_ROOT/commit_retain_capture_final_report.json

commit_retain_sanitize_environment() {
  local inherited prefix
  for prefix in GATE_A_ ORIGINAL_T25_ T25_ STABLE_I_ S128_ SERIAL_CREDIT_ RAY_ VLLM_; do
    while IFS= read -r inherited; do unset "$inherited"; done < <(compgen -v "$prefix" || true)
  done
  for prefix in MEMAGENT_GATEA_ MEMAGENT_T25_ MEMAGENT_S128_ MEMAGENT_STABLE_ MEMAGENT_SERIAL_CREDIT_; do
    while IFS= read -r inherited; do unset "$inherited"; done < <(compgen -v "$prefix" || true)
  done
  unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER PYTHONPATH PYTHONNOUSERSITE \
    TOKENIZERS_PARALLELISM TRAIN_BATCH_SIZE ROLLOUT_N PPO_MINI_BATCH_SIZE \
    PPO_MICRO_BATCH_SIZE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH \
    MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS TEMPERATURE TOP_P TOP_K MIN_P \
    N_GPUS FSDP_SIZE WANDB_MODE
}

commit_retain_require_checkout() {
  local script_repo
  script_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
  [[ $(cd -- "$COMMIT_RETAIN_REPO_DIR" && pwd -P) == "$script_repo" ]] || {
    echo 'COMMIT_RETAIN_NO_GO:P0 invoked checkout differs from explicit repository' >&2; exit 68;
  }
  [[ $(cd "$COMMIT_RETAIN_REPO_DIR" && git branch --show-current) == "$COMMIT_RETAIN_BRANCH" ]] || {
    echo 'COMMIT_RETAIN_NO_GO:P0 wrong branch' >&2; exit 70;
  }
  [[ -z $(cd "$COMMIT_RETAIN_REPO_DIR" && git status --porcelain) ]] || {
    echo 'COMMIT_RETAIN_NO_GO:P0 dirty worktree' >&2; exit 71;
  }
  [[ $(cd "$COMMIT_RETAIN_REPO_DIR" && git rev-parse HEAD) == "$COMMIT_RETAIN_EXPECTED_COMMIT" ]] || {
    echo 'COMMIT_RETAIN_NO_GO:P0 HEAD differs from expected commit' >&2; exit 72;
  }
}

commit_retain_acquire_lock() {
  command -v flock >/dev/null || {
    echo 'COMMIT_RETAIN_NO_GO:P0 flock is required' >&2; exit 63;
  }
  mkdir -p "$(dirname "$COMMIT_RETAIN_LOCK_FIRST")"
  exec 8>"$COMMIT_RETAIN_LOCK_FIRST"
  flock -n 8 || {
    echo "COMMIT_RETAIN_NO_GO:P0 GPU$COMMIT_RETAIN_GPU_FIRST project lock is held" >&2; exit 62;
  }
  exec 9>"$COMMIT_RETAIN_LOCK_SECOND"
  flock -n 9 || {
    echo "COMMIT_RETAIN_NO_GO:P0 GPU$COMMIT_RETAIN_GPU_SECOND project lock is held" >&2; exit 62;
  }
}

commit_retain_validate_gpu_pair() {
  command -v nvidia-smi >/dev/null || {
    echo 'COMMIT_RETAIN_NO_GO:P0 nvidia-smi is required' >&2; exit 78;
  }
  local identities observed
  identities=$(nvidia-smi -i "$COMMIT_RETAIN_GPUS" \
    --query-gpu=index,name --format=csv,noheader,nounits) || {
    echo "COMMIT_RETAIN_NO_GO:P0 GPU pair does not exist: $COMMIT_RETAIN_GPUS" >&2; exit 57;
  }
  observed=$(printf '%s\n' "$identities" | awk -F, '{gsub(/[[:space:]]/,"",$1); print $1}' | paste -sd, -)
  [[ $observed == "$COMMIT_RETAIN_GPUS" ]] || {
    echo "COMMIT_RETAIN_NO_GO:P0 nvidia-smi indices $observed != $COMMIT_RETAIN_GPUS" >&2; exit 57;
  }
  [[ $(printf '%s\n' "$identities" | grep -c 'NVIDIA H20') -eq 2 ]] || {
    echo "COMMIT_RETAIN_NO_GO:P0 GPU pair is not two NVIDIA H20 devices" >&2; exit 56;
  }
}

commit_retain_require_idle() {
  local processes
  processes=$(nvidia-smi -i "$COMMIT_RETAIN_GPUS" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${processes//[[:space:]]/} ]] || {
    echo "COMMIT_RETAIN_NO_GO:P0 GPU$COMMIT_RETAIN_GPUS are busy; no process was changed: $processes" >&2; exit 79;
  }
}

commit_retain_wait_idle() {
  local poll processes
  for poll in $(seq 1 45); do
    processes=$(nvidia-smi -i "$COMMIT_RETAIN_GPUS" --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z ${processes//[[:space:]]/} ]] && return 0
    sleep 2
  done
  echo "COMMIT_RETAIN_NO_GO:CLEANUP GPU$COMMIT_RETAIN_GPUS did not become idle: $processes" >&2
  return 81
}

commit_retain_require_p0() {
  [[ -f $COMMIT_RETAIN_P0 && -f $COMMIT_RETAIN_RESOLVED && -f $COMMIT_RETAIN_LEDGER ]] || {
    echo 'COMMIT_RETAIN_NO_GO:P0 run standalone P0 first' >&2; exit 61;
  }
  "$COMMIT_RETAIN_PYTHON" \
    "$COMMIT_RETAIN_REPO_DIR/tools/h20/preflight_qwen25_7b_commit_retain.py" \
    --manifest "$COMMIT_RETAIN_MANIFEST" --validate-p0-prefix >/dev/null
}

commit_retain_record() {
  "$COMMIT_RETAIN_PYTHON" \
    "$COMMIT_RETAIN_REPO_DIR/tools/h20/preflight_qwen25_7b_commit_retain.py" \
    --manifest "$COMMIT_RETAIN_MANIFEST" "$@" >/dev/null
}

commit_retain_issue_capture_credential() {
  "$COMMIT_RETAIN_PYTHON" \
    "$COMMIT_RETAIN_REPO_DIR/tools/h20/preflight_qwen25_7b_commit_retain.py" \
    --manifest "$COMMIT_RETAIN_MANIFEST" \
    --issue-capture-credential "$COMMIT_RETAIN_CREDENTIAL" \
    --issuer-shell-pid "$$" >/dev/null
}

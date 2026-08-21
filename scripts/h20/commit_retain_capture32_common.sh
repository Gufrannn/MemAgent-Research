#!/usr/bin/env bash
set -euo pipefail

capture32_die() {
  local code=$1
  shift
  echo "CAPTURE32_NO_GO:$*" >&2
  exit "$code"
}

[[ -n ${MEMAGENT_CAPTURE32_WORK_ROOT:-} ]] || \
  capture32_die 66 'P0 set MEMAGENT_CAPTURE32_WORK_ROOT explicitly'
[[ -n ${MEMAGENT_CAPTURE32_REPO_DIR:-} ]] || \
  capture32_die 67 'P0 set MEMAGENT_CAPTURE32_REPO_DIR explicitly'
[[ ${MEMAGENT_CAPTURE32_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || \
  capture32_die 65 'P0 expected commit must be a full Git SHA'
[[ ${MEMAGENT_CAPTURE32_RUN_ID:-} =~ ^[a-z0-9][a-z0-9_-]{1,31}$ ]] || \
  capture32_die 64 'P0 set a task-scoped run ID'
[[ ${MEMAGENT_CAPTURE32_PHYSICAL_GPUS:-} =~ ^(0|[1-9][0-9]*),(0|[1-9][0-9]*)$ ]] || \
  capture32_die 59 'P0 MEMAGENT_CAPTURE32_PHYSICAL_GPUS must be exactly N,M'
[[ $MEMAGENT_CAPTURE32_WORK_ROOT == /* && $MEMAGENT_CAPTURE32_REPO_DIR == /* ]] || \
  capture32_die 69 'P0 task-scoped paths must be absolute'

IFS=, read -r CAPTURE32_GPU0 CAPTURE32_GPU1 <<<"$MEMAGENT_CAPTURE32_PHYSICAL_GPUS"
(( 10#$CAPTURE32_GPU0 < 10#$CAPTURE32_GPU1 )) || \
  capture32_die 58 'P0 physical GPU pair must contain two distinct indices in ascending order'

readonly CAPTURE32_WORK_ROOT=$MEMAGENT_CAPTURE32_WORK_ROOT
readonly CAPTURE32_REPO_DIR=$MEMAGENT_CAPTURE32_REPO_DIR
readonly CAPTURE32_EXPECTED_COMMIT=$MEMAGENT_CAPTURE32_EXPECTED_COMMIT
readonly CAPTURE32_RUN_ID=$MEMAGENT_CAPTURE32_RUN_ID
readonly CAPTURE32_GPUS=$MEMAGENT_CAPTURE32_PHYSICAL_GPUS
readonly CAPTURE32_GPU0 CAPTURE32_GPU1
readonly CAPTURE32_BRANCH=h20/qwen25-7b-paired-effect-pipeline-20260821
readonly CAPTURE32_PYTHON=$CAPTURE32_WORK_ROOT/.venv/bin/python
readonly CAPTURE32_NVIDIA_SMI=/usr/bin/nvidia-smi
readonly CAPTURE32_FLOCK=/usr/bin/flock
readonly CAPTURE32_MANIFEST=$CAPTURE32_REPO_DIR/manifests/h20/qwen25_7b_commit_retain_capture32_seed2026.json
readonly CAPTURE32_LOG_ROOT=$CAPTURE32_WORK_ROOT/logs/commit_retain_capture32_frozen_20260821/$CAPTURE32_RUN_ID
readonly CAPTURE32_CERT_ROOT=$CAPTURE32_LOG_ROOT/certificates
readonly CAPTURE32_P0=$CAPTURE32_CERT_ROOT/p0_preflight.json
readonly CAPTURE32_RESOLVED=$CAPTURE32_CERT_ROOT/p0_resolved_manifest.json
readonly CAPTURE32_LEDGER=$CAPTURE32_LOG_ROOT/commit_retain_capture32_execution_ledger.jsonl
readonly CAPTURE32_CREDENTIAL=$CAPTURE32_LOG_ROOT/credentials/capture_child.json
readonly CAPTURE32_CREDENTIAL_CONSUMPTION=$CAPTURE32_LOG_ROOT/credentials/capture_child_consumed.json
readonly CAPTURE32_CAPTURE=$CAPTURE32_LOG_ROOT/captures/commit_retain_pairs.jsonl
readonly CAPTURE32_RUN_RECEIPT=$CAPTURE32_LOG_ROOT/captures/run_receipt.json
readonly CAPTURE32_FINAL=$CAPTURE32_CERT_ROOT/commit_retain_capture32_final_report.json
readonly CAPTURE32_PREREG_ANCHOR=$CAPTURE32_WORK_ROOT/provenance/commit_retain_capture32/$CAPTURE32_RUN_ID.preregistration.json
readonly CAPTURE32_TERMINAL_ANCHOR=$CAPTURE32_WORK_ROOT/provenance/commit_retain_capture32/$CAPTURE32_RUN_ID.terminal.json

capture32_sanitize_environment() {
  local inherited prefix
  for prefix in GATE_A_ ORIGINAL_T25_ T25_ STABLE_I_ S128_ SERIAL_CREDIT_ RAY_ VLLM_; do
    while IFS= read -r inherited; do unset "$inherited"; done < <(compgen -v "$prefix" || true)
  done
  for prefix in MEMAGENT_GATEA_ MEMAGENT_T25_ MEMAGENT_S128_ MEMAGENT_STABLE_ \
    MEMAGENT_SERIAL_CREDIT_ MEMAGENT_COMMIT_RETAIN_; do
    while IFS= read -r inherited; do unset "$inherited"; done < <(compgen -v "$prefix" || true)
  done
  unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER PYTHONPATH PYTHONNOUSERSITE \
    TOKENIZERS_PARALLELISM TRAIN_BATCH_SIZE ROLLOUT_N PPO_MINI_BATCH_SIZE \
    PPO_MICRO_BATCH_SIZE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH \
    MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS TEMPERATURE TOP_P TOP_K MIN_P \
    N_GPUS FSDP_SIZE WANDB_MODE MEMAGENT_CAPTURE32_LOCK_FDS \
    MEMAGENT_CAPTURE32_LOCK_PATHS MEMAGENT_CAPTURE32_GPU_UUIDS \
    MEMAGENT_CAPTURE32_GPU_NAMES
}

capture32_export_runtime() {
  export CUDA_VISIBLE_DEVICES=$CAPTURE32_GPUS
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export VLLM_USE_V1=0
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export PYTHONNOUSERSITE=1
  export PYTHONPATH=$CAPTURE32_REPO_DIR
  export TOKENIZERS_PARALLELISM=false
}

capture32_require_checkout() {
  local script_repo
  script_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
  [[ $(cd -- "$CAPTURE32_REPO_DIR" && pwd -P) == "$script_repo" ]] || \
    capture32_die 68 'P0 invoked checkout differs from explicit repository'
  [[ $(cd "$CAPTURE32_REPO_DIR" && git branch --show-current) == "$CAPTURE32_BRANCH" ]] || \
    capture32_die 70 'P0 wrong branch'
  [[ -z $(cd "$CAPTURE32_REPO_DIR" && git status --porcelain) ]] || \
    capture32_die 71 'P0 dirty worktree'
  [[ $(cd "$CAPTURE32_REPO_DIR" && git rev-parse HEAD) == "$CAPTURE32_EXPECTED_COMMIT" ]] || \
    capture32_die 72 'P0 HEAD differs from expected commit'
  [[ -x $CAPTURE32_PYTHON ]] || capture32_die 77 'P0 task Python is missing or not executable'
}

capture32_acquire_locks() {
  [[ -x $CAPTURE32_FLOCK ]] || capture32_die 63 'P0 /usr/bin/flock is required'
  local lock_root=$CAPTURE32_WORK_ROOT/locks
  [[ ! -L $lock_root ]] || capture32_die 57 'P0 lock directory must not be a symlink'
  mkdir -p "$lock_root"

  local -a lock_paths
  local lock_path lock_fd lock_ordinal=0 joined_paths= joined_fds=
  lock_paths=(
    "$lock_root/memagent_h20_gpu_$CAPTURE32_GPU0.lock"
    "$lock_root/memagent_h20_gpu_$CAPTURE32_GPU1.lock"
  )
  if [[ $CAPTURE32_GPU0 == 4 || $CAPTURE32_GPU0 == 5 || \
        $CAPTURE32_GPU1 == 4 || $CAPTURE32_GPU1 == 5 ]]; then
    lock_paths+=("$lock_root/memagent_gate_a_gpu_4_5.lock")
  fi
  if [[ $CAPTURE32_GPU0 == 6 || $CAPTURE32_GPU0 == 7 || \
        $CAPTURE32_GPU1 == 6 || $CAPTURE32_GPU1 == 7 ]]; then
    lock_paths+=("$lock_root/memagent_gate_a_gpu_6_7.lock")
  fi

  # This order is part of the P0 receipt: physical locks in ascending GPU
  # order, then intersecting legacy aggregate locks in 4_5, 6_7 order.
  for lock_path in "${lock_paths[@]}"; do
    [[ ! -L $lock_path ]] || capture32_die 57 "P0 lock path is a symlink: $lock_path"
    # Fixed high descriptors work on the H20 Bash and on the older Bash used
    # by CPU review hosts. They also make the inherited descriptor inventory
    # explicit instead of relying on version-specific dynamic-FD behavior.
    case $lock_ordinal in
      0) exec 180>"$lock_path"; lock_fd=180 ;;
      1) exec 181>"$lock_path"; lock_fd=181 ;;
      2) exec 182>"$lock_path"; lock_fd=182 ;;
      3) exec 183>"$lock_path"; lock_fd=183 ;;
      *) capture32_die 52 'P0 internal GPU lock inventory exceeds four locks' ;;
    esac
    "$CAPTURE32_FLOCK" -n "$lock_fd" || \
      capture32_die 62 "P0 GPU lock is held: $lock_path"
    joined_paths+="${joined_paths:+,}$lock_path"
    joined_fds+="${joined_fds:+,}$lock_fd"
    ((lock_ordinal += 1))
  done

  # Bash keeps these descriptors open across exec; the exported inventory lets
  # P0 and the child authenticate the exact locks inherited from this shell.
  export MEMAGENT_CAPTURE32_LOCK_PATHS=$joined_paths
  export MEMAGENT_CAPTURE32_LOCK_FDS=$joined_fds
  readonly MEMAGENT_CAPTURE32_LOCK_PATHS MEMAGENT_CAPTURE32_LOCK_FDS
}

capture32_require_h20_binding() {
  [[ -x $CAPTURE32_NVIDIA_SMI ]] || capture32_die 78 'P0 /usr/bin/nvidia-smi is required'
  local gpu reported name uuid
  local joined_names= joined_uuids=
  for gpu in "$CAPTURE32_GPU0" "$CAPTURE32_GPU1"; do
    reported=$($CAPTURE32_NVIDIA_SMI -i "$gpu" --query-gpu=index --format=csv,noheader,nounits)
    name=$($CAPTURE32_NVIDIA_SMI -i "$gpu" --query-gpu=name --format=csv,noheader)
    uuid=$($CAPTURE32_NVIDIA_SMI -i "$gpu" --query-gpu=uuid --format=csv,noheader)
    [[ $reported != *$'\n'* && $name != *$'\n'* && $uuid != *$'\n'* ]] || \
      capture32_die 56 "P0 physical GPU $gpu did not resolve to exactly one device"
    [[ $reported == "$gpu" ]] || capture32_die 56 "P0 requested GPU $gpu resolved as $reported"
    [[ $name == *'NVIDIA H20'* ]] || capture32_die 55 "P0 GPU $gpu is not NVIDIA H20: $name"
    [[ $uuid =~ ^GPU-[0-9A-Fa-f-]+$ ]] || capture32_die 54 "P0 GPU $gpu returned an invalid UUID"
    joined_names+="${joined_names:+|}$name"
    joined_uuids+="${joined_uuids:+,}$uuid"
  done
  [[ ${joined_uuids%%,*} != ${joined_uuids##*,} ]] || \
    capture32_die 53 'P0 selected physical GPUs resolved to the same UUID'
  export MEMAGENT_CAPTURE32_GPU_NAMES=$joined_names
  export MEMAGENT_CAPTURE32_GPU_UUIDS=$joined_uuids
  readonly MEMAGENT_CAPTURE32_GPU_NAMES MEMAGENT_CAPTURE32_GPU_UUIDS
}

capture32_require_idle() {
  local processes
  processes=$($CAPTURE32_NVIDIA_SMI -i "$CAPTURE32_GPUS" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${processes//[[:space:]]/} ]] || \
    capture32_die 79 "P0 GPU$CAPTURE32_GPUS are busy; no process was changed: $processes"
}

capture32_wait_idle() {
  local poll processes=
  for ((poll = 1; poll <= 45; poll += 1)); do
    processes=$($CAPTURE32_NVIDIA_SMI -i "$CAPTURE32_GPUS" --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z ${processes//[[:space:]]/} ]] && return 0
    sleep 2
  done
  echo "CAPTURE32_NO_GO:CLEANUP GPU$CAPTURE32_GPUS did not become idle: $processes" >&2
  return 81
}

capture32_require_p0() {
  [[ -f $CAPTURE32_P0 && -f $CAPTURE32_RESOLVED && -f $CAPTURE32_LEDGER && \
     -f $CAPTURE32_PREREG_ANCHOR ]] || \
    capture32_die 61 'P0 run standalone capture32 P0 first'
  "$CAPTURE32_PYTHON" \
    "$CAPTURE32_REPO_DIR/tools/h20/preflight_qwen25_7b_commit_retain_capture32.py" \
    --manifest "$CAPTURE32_MANIFEST" --validate-p0-prefix >/dev/null
}

capture32_record_complete() {
  "$CAPTURE32_PYTHON" \
    "$CAPTURE32_REPO_DIR/tools/h20/preflight_qwen25_7b_commit_retain_capture32.py" \
    --manifest "$CAPTURE32_MANIFEST" --record-capture-complete >/dev/null
}

capture32_issue_capture_credential() {
  "$CAPTURE32_PYTHON" \
    "$CAPTURE32_REPO_DIR/tools/h20/preflight_qwen25_7b_commit_retain_capture32.py" \
    --manifest "$CAPTURE32_MANIFEST" \
    --issue-capture-credential "$CAPTURE32_CREDENTIAL" \
    --issuer-shell-pid "$$" >/dev/null
}

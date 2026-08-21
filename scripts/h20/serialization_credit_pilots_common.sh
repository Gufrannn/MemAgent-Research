#!/usr/bin/env bash
set -euo pipefail

[[ -n ${MEMAGENT_SERIAL_CREDIT_WORK_ROOT:-} ]] || {
  echo 'SERIAL_CREDIT_NO_GO:P0 set MEMAGENT_SERIAL_CREDIT_WORK_ROOT explicitly' >&2; exit 66;
}
[[ -n ${MEMAGENT_SERIAL_CREDIT_REPO_DIR:-} ]] || {
  echo 'SERIAL_CREDIT_NO_GO:P0 set MEMAGENT_SERIAL_CREDIT_REPO_DIR explicitly' >&2; exit 67;
}
[[ ${MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'SERIAL_CREDIT_NO_GO:P0 expected commit must be a full Git SHA' >&2; exit 65;
}
[[ ${MEMAGENT_SERIAL_CREDIT_RUN_ID:-} =~ ^[a-z0-9][a-z0-9_-]{1,31}$ ]] || {
  echo 'SERIAL_CREDIT_NO_GO:P0 set a task-scoped MEMAGENT_SERIAL_CREDIT_RUN_ID' >&2; exit 64;
}
[[ $MEMAGENT_SERIAL_CREDIT_WORK_ROOT == /* && $MEMAGENT_SERIAL_CREDIT_REPO_DIR == /* ]] || {
  echo 'SERIAL_CREDIT_NO_GO:P0 task-scoped paths must be absolute' >&2; exit 69;
}

readonly SERIAL_CREDIT_WORK_ROOT=$MEMAGENT_SERIAL_CREDIT_WORK_ROOT
readonly SERIAL_CREDIT_REPO_DIR=$MEMAGENT_SERIAL_CREDIT_REPO_DIR
readonly SERIAL_CREDIT_EXPECTED_COMMIT=$MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT
readonly SERIAL_CREDIT_RUN_ID=$MEMAGENT_SERIAL_CREDIT_RUN_ID
readonly SERIAL_CREDIT_PYTHON=$SERIAL_CREDIT_WORK_ROOT/.venv/bin/python
readonly SERIAL_CREDIT_MANIFEST=$SERIAL_CREDIT_REPO_DIR/manifests/h20/qwen25_7b_serialization_credit_pilots_seed2026.json
readonly SERIAL_CREDIT_LOG_ROOT=$SERIAL_CREDIT_WORK_ROOT/logs/serialization_credit_pilots_gpu23_frozen_20260821/$SERIAL_CREDIT_RUN_ID
readonly SERIAL_CREDIT_CERT_ROOT=$SERIAL_CREDIT_LOG_ROOT/certificates
readonly SERIAL_CREDIT_P0=$SERIAL_CREDIT_CERT_ROOT/p0_preflight.json
readonly SERIAL_CREDIT_RESOLVED=$SERIAL_CREDIT_CERT_ROOT/p0_resolved_manifest.json
readonly SERIAL_CREDIT_PARENT_AUTHORITY=$SERIAL_CREDIT_CERT_ROOT/parent_receipt_authority.secret
readonly SERIAL_CREDIT_READONLY_REAUDIT=$SERIAL_CREDIT_CERT_ROOT/serialization_credit_pilot_readonly_reaudit.json
readonly SERIAL_CREDIT_LEDGER=$SERIAL_CREDIT_LOG_ROOT/serialization_credit_pilot_execution_ledger.jsonl
readonly SERIAL_CREDIT_SMSB_ROOT=$SERIAL_CREDIT_LOG_ROOT/smsb
readonly SERIAL_CREDIT_SMSB_CAPTURES=$SERIAL_CREDIT_SMSB_ROOT/captures.jsonl
readonly SERIAL_CREDIT_SMSB_REPLAYS=$SERIAL_CREDIT_SMSB_ROOT/replays
readonly SERIAL_CREDIT_SMSB_CREDENTIALS=$SERIAL_CREDIT_SMSB_ROOT/credentials
readonly SERIAL_CREDIT_SMSB_RECEIPTS=$SERIAL_CREDIT_SMSB_ROOT/receipts
readonly SERIAL_CREDIT_SMSB_CHILD_LOGS=$SERIAL_CREDIT_SMSB_ROOT/child_logs
readonly SERIAL_CREDIT_SMSB_REPORT=$SERIAL_CREDIT_SMSB_ROOT/adjudication.json
readonly SERIAL_CREDIT_TETRAD_ROOT=$SERIAL_CREDIT_LOG_ROOT/tetrad
readonly SERIAL_CREDIT_TETRAD_AUTHORING=$SERIAL_CREDIT_TETRAD_ROOT/authoring.jsonl
readonly SERIAL_CREDIT_TETRAD_MANIFEST=$SERIAL_CREDIT_TETRAD_ROOT/manifest.jsonl
readonly SERIAL_CREDIT_TETRAD_RESULTS=$SERIAL_CREDIT_TETRAD_ROOT/results
readonly SERIAL_CREDIT_TETRAD_CREDENTIALS=$SERIAL_CREDIT_TETRAD_ROOT/credentials
readonly SERIAL_CREDIT_TETRAD_RECEIPTS=$SERIAL_CREDIT_TETRAD_ROOT/receipts
readonly SERIAL_CREDIT_TETRAD_CHILD_LOGS=$SERIAL_CREDIT_TETRAD_ROOT/child_logs
readonly SERIAL_CREDIT_TETRAD_REPORT=$SERIAL_CREDIT_TETRAD_ROOT/adjudication.json
readonly SERIAL_CREDIT_FINAL_REPORT=$SERIAL_CREDIT_CERT_ROOT/serialization_credit_pilot_final_report.json
readonly SERIAL_CREDIT_GPUS=2,3
readonly SERIAL_CREDIT_LOCK=$SERIAL_CREDIT_WORK_ROOT/locks/memagent_serial_credit_gpu_2_3.lock

serial_credit_sanitize_inherited_environment() {
  local inherited prefix
  for prefix in GATE_A_ ORIGINAL_T25_ T25_ STABLE_I_ S128_ RAY_ VLLM_; do
    while IFS= read -r inherited; do
      unset "$inherited"
    done < <(compgen -v "$prefix" || true)
  done
  for prefix in MEMAGENT_GATEA_ MEMAGENT_T25_ MEMAGENT_S128_ MEMAGENT_STABLE_; do
    while IFS= read -r inherited; do
      unset "$inherited"
    done < <(compgen -v "$prefix" || true)
  done
  unset CUDA_VISIBLE_DEVICES PYTHONPATH TOKENIZERS_PARALLELISM \
    CUDA_DEVICE_ORDER \
    TRAIN_BATCH_SIZE ROLLOUT_N PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE \
    MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS \
    TEMPERATURE TOP_P TOP_K MIN_P N_GPUS FSDP_SIZE WANDB_MODE
}

serial_credit_require_checkout() {
  local script_repo
  script_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
  [[ $(cd -- "$SERIAL_CREDIT_REPO_DIR" && pwd -P) == "$script_repo" ]] || {
    echo 'SERIAL_CREDIT_NO_GO:P0 invoked checkout differs from explicit repository' >&2; exit 68;
  }
  [[ $(cd "$SERIAL_CREDIT_REPO_DIR" && git branch --show-current) == h20/qwen25-7b-serialization-credit-gpu23-20260821 ]] || {
    echo 'SERIAL_CREDIT_NO_GO:P0 wrong branch' >&2; exit 70;
  }
  [[ -z $(cd "$SERIAL_CREDIT_REPO_DIR" && git status --porcelain) ]] || {
    echo 'SERIAL_CREDIT_NO_GO:P0 dirty worktree' >&2; exit 71;
  }
  [[ $(cd "$SERIAL_CREDIT_REPO_DIR" && git rev-parse HEAD) == "$SERIAL_CREDIT_EXPECTED_COMMIT" ]] || {
    echo 'SERIAL_CREDIT_NO_GO:P0 HEAD differs from expected commit' >&2; exit 72;
  }
}

serial_credit_require_p0() {
  [[ -f $SERIAL_CREDIT_P0 && -f $SERIAL_CREDIT_RESOLVED && -f $SERIAL_CREDIT_LEDGER ]] || {
    echo 'SERIAL_CREDIT_NO_GO:P0 run standalone P0 before any GPU mechanism pilot' >&2; exit 61;
  }
  "$SERIAL_CREDIT_PYTHON" \
    "$SERIAL_CREDIT_REPO_DIR/tools/h20/preflight_qwen25_7b_serialization_credit.py" \
    --manifest "$SERIAL_CREDIT_MANIFEST" --validate-p0-prefix >/dev/null || {
      echo 'SERIAL_CREDIT_NO_GO:P0 certificate prefix is not authenticated' >&2; exit 80;
    }
}

serial_credit_acquire_lock() {
  command -v flock >/dev/null || {
    echo 'SERIAL_CREDIT_NO_GO:P0 flock is required' >&2; exit 63;
  }
  mkdir -p "$(dirname "$SERIAL_CREDIT_LOCK")"
  exec 8>"$SERIAL_CREDIT_LOCK"
  flock -n 8 || {
    echo 'SERIAL_CREDIT_NO_GO:P0 GPU2-3 project lock is held' >&2; exit 62;
  }
}

serial_credit_require_idle() {
  command -v nvidia-smi >/dev/null || {
    echo 'SERIAL_CREDIT_NO_GO:P0 nvidia-smi is required' >&2; exit 78;
  }
  local processes
  processes=$(nvidia-smi -i "$SERIAL_CREDIT_GPUS" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${processes//[[:space:]]/} ]] || {
    echo "SERIAL_CREDIT_NO_GO:P0 GPU2-3 are busy; no process was changed: $processes" >&2; exit 79;
  }
}

serial_credit_wait_idle() {
  local processes poll
  for poll in $(seq 1 45); do
    processes=$(nvidia-smi -i "$SERIAL_CREDIT_GPUS" --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z ${processes//[[:space:]]/} ]] && return 0
    sleep 2
  done
  echo "SERIAL_CREDIT_NO_GO:CLEANUP GPU2-3 did not become idle: $processes" >&2
  return 81
}

serial_credit_record() {
  "$SERIAL_CREDIT_PYTHON" \
    "$SERIAL_CREDIT_REPO_DIR/tools/h20/preflight_qwen25_7b_serialization_credit.py" \
    --manifest "$SERIAL_CREDIT_MANIFEST" "$@" >/dev/null
}

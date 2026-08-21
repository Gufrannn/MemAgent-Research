#!/usr/bin/env bash
set -euo pipefail

readonly PRD_BRANCH='h20/qwen25-7b-prd-memrl-t25-frozen-20260822'
readonly PRD_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly PRD_REPO=$(cd -- "$PRD_SCRIPT_DIR/../.." && pwd -P)

prd_die() { echo "PRD_NO_GO:$*" >&2; exit 64; }

prd_require_env() {
  [[ ${WORK_ROOT:-} == /* ]] || prd_die 'WORK_ROOT must be an explicit absolute path'
  [[ ${EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || prd_die 'EXPECTED_COMMIT must be a full SHA'
  [[ ${GPU_PAIR:-} =~ ^[0-9]+,[0-9]+$ ]] || prd_die 'GPU_PAIR must be N,M'
  local first=${GPU_PAIR%,*} second=${GPU_PAIR#*,}
  (( first < second )) || prd_die 'GPU_PAIR must be distinct canonical ascending indices'
  readonly PRD_PYTHON=${PRD_PYTHON:-$WORK_ROOT/.venv/bin/python}
  [[ -x $PRD_PYTHON ]] || prd_die "missing project Python: $PRD_PYTHON"
  [[ $(git -C "$PRD_REPO" branch --show-current) == "$PRD_BRANCH" ]] || prd_die 'wrong branch'
  [[ $(git -C "$PRD_REPO" rev-parse HEAD) == "$EXPECTED_COMMIT" ]] || prd_die 'wrong exact commit'
  [[ -z $(git -C "$PRD_REPO" status --porcelain) ]] || prd_die 'dirty worktree'
}

prd_acquire_gpu_locks() {
  command -v flock >/dev/null || prd_die 'flock is required'
  mkdir -p "$WORK_ROOT/locks"
  local first=${GPU_PAIR%,*} second=${GPU_PAIR#*,}
  readonly PRD_LOCK_FIRST=$WORK_ROOT/locks/memagent_h20_gpu_${first}.lock
  readonly PRD_LOCK_SECOND=$WORK_ROOT/locks/memagent_h20_gpu_${second}.lock
  exec 8>"$PRD_LOCK_FIRST"; flock -n 8 || prd_die "lock conflict: $PRD_LOCK_FIRST"
  exec 9>"$PRD_LOCK_SECOND"; flock -n 9 || prd_die "lock conflict: $PRD_LOCK_SECOND"
  command -v nvidia-smi >/dev/null || prd_die 'nvidia-smi missing'
  local apps
  apps=$(nvidia-smi -i "$GPU_PAIR" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${apps//[[:space:]]/} ]] || prd_die "selected GPU pair is occupied; no process was changed: $apps"
}

prd_paths() {
  [[ ${RUN_ID:-} =~ ^[a-z0-9][a-z0-9._-]{7,127}$ ]] || prd_die 'RUN_ID must be explicit and stable'
  readonly PRD_RUN_ROOT=$WORK_ROOT/logs/prd_memrl/$RUN_ID
  readonly PRD_CERT_ROOT=$PRD_RUN_ROOT/certificates
  readonly PRD_LEDGER=$PRD_RUN_ROOT/prd_memrl_execution_ledger.jsonl
  mkdir -p "$PRD_CERT_ROOT"
}

prd_append_ledger() {
  local event=$1 payload_file=$2
  "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_ledger.py" append \
    --ledger "$PRD_LEDGER" --run-id "$RUN_ID" --event "$event" \
    --git-commit "$EXPECTED_COMMIT" --payload "$payload_file"
}

prd_verify_gate() {
  local path=$1 decision=$2
  "$PRD_PYTHON" - "$path" "$decision" "$EXPECTED_COMMIT" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); e=p.get("evidence",{})
assert p.get("status")=="PASS" and p.get("decision")==sys.argv[2]
assert e.get("git_commit")==sys.argv[3]
PY
}

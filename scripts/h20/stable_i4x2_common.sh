#!/usr/bin/env bash
set -euo pipefail

[[ -n ${MEMAGENT_STABLE_I_WORK_ROOT:-} ]] || {
  echo 'STABLE_I_NO_GO:P0 set MEMAGENT_STABLE_I_WORK_ROOT explicitly' >&2; exit 66;
}
[[ -n ${MEMAGENT_STABLE_I_REPO_DIR:-} ]] || {
  echo 'STABLE_I_NO_GO:P0 set MEMAGENT_STABLE_I_REPO_DIR explicitly' >&2; exit 67;
}
[[ ${MEMAGENT_STABLE_I_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'STABLE_I_NO_GO:P0 MEMAGENT_STABLE_I_EXPECTED_COMMIT must be a full Git SHA' >&2; exit 65;
}
[[ $MEMAGENT_STABLE_I_WORK_ROOT == /* && $MEMAGENT_STABLE_I_REPO_DIR == /* ]] || {
  echo 'STABLE_I_NO_GO:P0 task-scoped paths must be absolute' >&2; exit 69;
}

readonly STABLE_I_WORK_ROOT=$MEMAGENT_STABLE_I_WORK_ROOT
readonly STABLE_I_REPO_DIR=$MEMAGENT_STABLE_I_REPO_DIR
readonly STABLE_I_EXPECTED_COMMIT=$MEMAGENT_STABLE_I_EXPECTED_COMMIT
readonly STABLE_I_PYTHON=$STABLE_I_WORK_ROOT/.venv/bin/python
readonly STABLE_I_MANIFEST=$STABLE_I_REPO_DIR/manifests/h20/qwen25_7b_stable_i4x2_seed2026.json
readonly STABLE_I_LOG_ROOT=$STABLE_I_WORK_ROOT/logs/stable_i4x2_frozen_20260821r2
readonly STABLE_I_CERT_ROOT=$STABLE_I_LOG_ROOT/certificates
readonly STABLE_I_P0=$STABLE_I_CERT_ROOT/p0_preflight.json
readonly STABLE_I_RESOLVED_MANIFEST=$STABLE_I_CERT_ROOT/p0_resolved_manifest.json
readonly STABLE_I_LEDGER=$STABLE_I_LOG_ROOT/stable_identity_execution_ledger.jsonl
readonly STABLE_I_GPU_DECLARATION=6,7
# Reuse the existing GPU6-7 project lock so this canary cannot race the
# already-frozen Gate A wrapper on the shared H20 account.
readonly STABLE_I_RUN_LOCK=$STABLE_I_WORK_ROOT/locks/memagent_gate_a_gpu_6_7.lock

stable_i_require_clean_frozen_checkout() {
  local invoked_repo
  invoked_repo=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
  [[ $(cd -- "$STABLE_I_REPO_DIR" && pwd -P) == "$invoked_repo" ]] || {
    echo "STABLE_I_NO_GO:P0 invoked checkout differs from MEMAGENT_STABLE_I_REPO_DIR: $invoked_repo != $STABLE_I_REPO_DIR" >&2
    exit 68
  }
  [[ $(cd "$STABLE_I_REPO_DIR" && git branch --show-current) == h20/qwen25-7b-stable-eval-i4x2-frozen-20260821 ]] || {
    echo 'STABLE_I_NO_GO:P0 wrong branch' >&2; exit 70;
  }
  [[ -z $(cd "$STABLE_I_REPO_DIR" && git status --porcelain) ]] || {
    echo 'STABLE_I_NO_GO:P0 dirty worktree' >&2; exit 71;
  }
  [[ $(cd "$STABLE_I_REPO_DIR" && git rev-parse HEAD) == "$STABLE_I_EXPECTED_COMMIT" ]] || {
    echo 'STABLE_I_NO_GO:P0 HEAD differs from MEMAGENT_STABLE_I_EXPECTED_COMMIT' >&2; exit 64;
  }
}

stable_i_acquire_run_lock() {
  command -v flock >/dev/null || { echo 'STABLE_I_NO_GO:P0 flock is required' >&2; exit 63; }
  mkdir -p "$(dirname "$STABLE_I_RUN_LOCK")"
  exec 8>"$STABLE_I_RUN_LOCK"
  flock -n 8 || {
    echo "STABLE_I_NO_GO:P0 another process owns the GPU6-7 stable-I lock: $STABLE_I_RUN_LOCK" >&2
    exit 62
  }
}

stable_i_require_declared_gpus_idle() {
  command -v nvidia-smi >/dev/null || { echo 'STABLE_I_NO_GO:P0 nvidia-smi is required' >&2; exit 78; }
  local applications
  applications=$(nvidia-smi -i "$STABLE_I_GPU_DECLARATION" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${applications//[[:space:]]/} ]] || {
    echo "STABLE_I_NO_GO:P0 GPU6-7 are not idle; no process was changed: $applications" >&2
    exit 79
  }
}

stable_i_wait_for_attempt_cleanup() {
  command -v nvidia-smi >/dev/null || { echo 'STABLE_I_NO_GO:P0 nvidia-smi is required' >&2; exit 78; }
  local applications
  local poll
  for poll in $(seq 1 45); do
    applications=$(nvidia-smi -i "$STABLE_I_GPU_DECLARATION" --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ -z ${applications//[[:space:]]/} ]]; then
      return 0
    fi
    sleep 2
  done
  echo "STABLE_I_NO_GO:CLEANUP GPU6-7 did not become idle after repeat_a; no process was killed: $applications" >&2
  exit 81
}

stable_i_require_p0() {
  [[ -f $STABLE_I_P0 && -f $STABLE_I_RESOLVED_MANIFEST ]] || {
    echo 'STABLE_I_NO_GO:P0 run the standalone P0 command before the two attempts' >&2
    exit 61
  }
  "$STABLE_I_PYTHON" - "$STABLE_I_P0" "$STABLE_I_RESOLVED_MANIFEST" "$STABLE_I_LEDGER" "$STABLE_I_EXPECTED_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys

certificate = json.loads(pathlib.Path(sys.argv[1]).read_text())
resolved_path = pathlib.Path(sys.argv[2])
ledger_path = pathlib.Path(sys.argv[3])
expected_commit = sys.argv[4]
resolved_sha = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
certificate_sha = hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()
resolved = json.loads(resolved_path.read_text())
computed_manifest_hash = hashlib.sha256(json.dumps(
    resolved["identity_payload"], ensure_ascii=False, sort_keys=True,
    separators=(",", ":"), allow_nan=False,
).encode("utf-8")).hexdigest()
computed_execution_binding_hash = hashlib.sha256(json.dumps(
    resolved["execution_binding"], ensure_ascii=False, sort_keys=True,
    separators=(",", ":"), allow_nan=False,
).encode("utf-8")).hexdigest()
ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
evidence = certificate.get("evidence", {})
ledger_payload = dict(ledger[0]) if len(ledger) == 1 else {}
ledger_record_sha = ledger_payload.pop("record_sha256", None)
computed_ledger_sha = hashlib.sha256(json.dumps(
    ledger_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode("utf-8")).hexdigest() if ledger_payload else None
valid = (
    certificate.get("status") == "PASS"
    and evidence.get("git_commit") == expected_commit
    and evidence.get("expected_git_commit") == expected_commit
    and evidence.get("resolved_manifest_sha256") == resolved_sha
    and evidence.get("eval_manifest_hash") == computed_manifest_hash
    and resolved.get("eval_manifest_hash") == computed_manifest_hash
    and evidence.get("execution_binding_sha256") == computed_execution_binding_hash
    and len(ledger) == 1
    and ledger[0].get("record_type") == "s0_preflight"
    and ledger[0].get("artifact_sha256") == certificate_sha
    and ledger[0].get("eval_manifest_hash") == computed_manifest_hash
    and ledger[0].get("execution_binding_sha256") == computed_execution_binding_hash
    and ledger[0].get("runtime_binding_sha256") == evidence.get("runtime_binding_sha256")
    and ledger[0].get("git_commit") == expected_commit
    and ledger[0].get("run_id") == evidence.get("run_id")
    and ledger[0].get("record_index") == 0
    and ledger[0].get("previous_record_sha256") == "0" * 64
    and ledger_record_sha == computed_ledger_sha
)
raise SystemExit(0 if valid else 1)
PY
}

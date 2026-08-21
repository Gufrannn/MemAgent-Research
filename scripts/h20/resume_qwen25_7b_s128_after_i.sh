#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly HOTFIX_REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)

[[ -n ${MEMAGENT_S128_IT_WORK_ROOT:-} ]] || {
  echo 'S128_IT_RECOVERY_NO_GO: set MEMAGENT_S128_IT_WORK_ROOT explicitly' >&2; exit 66;
}
[[ -n ${MEMAGENT_S128_IT_REPO_DIR:-} ]] || {
  echo 'S128_IT_RECOVERY_NO_GO: set MEMAGENT_S128_IT_REPO_DIR explicitly' >&2; exit 67;
}
[[ ${MEMAGENT_S128_IT_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'S128_IT_RECOVERY_NO_GO: original expected commit must be a full SHA' >&2; exit 65;
}
[[ ${MEMAGENT_S128_IT_AUDIT_CODE_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'S128_IT_RECOVERY_NO_GO: audit-code commit must be a full SHA' >&2; exit 64;
}

readonly WORK_ROOT=$MEMAGENT_S128_IT_WORK_ROOT
readonly ORIGINAL_REPO=$(cd -- "$MEMAGENT_S128_IT_REPO_DIR" && pwd -P)
readonly ORIGINAL_COMMIT=$MEMAGENT_S128_IT_EXPECTED_COMMIT
readonly AUDIT_COMMIT=$MEMAGENT_S128_IT_AUDIT_CODE_COMMIT
readonly PY=$WORK_ROOT/.venv/bin/python
readonly MANIFEST=$ORIGINAL_REPO/manifests/h20/qwen25_7b_s128_it_seed2026.json
readonly LOG_ROOT=$WORK_ROOT/logs/s128_it_original_t25_frozen_20260821
readonly LEDGER=$LOG_ROOT/s128_it_execution_ledger.jsonl
readonly I_ROOT=$LOG_ROOT/interface_i_base
readonly T_ROOT=$LOG_ROOT/interface_t25_original
readonly LOCK=$WORK_ROOT/locks/memagent_gate_a_gpu_6_7.lock
readonly PREFLIGHT=$HOTFIX_REPO/tools/h20/preflight_qwen25_7b_s128_it.py
readonly AUDIT=$HOTFIX_REPO/tools/h20/audit_qwen25_7b_s128_it.py

[[ $(git -C "$HOTFIX_REPO" rev-parse HEAD) == "$AUDIT_COMMIT" ]] || {
  echo 'S128_IT_RECOVERY_NO_GO: hotfix HEAD differs from explicit audit commit' >&2; exit 63;
}
[[ -z $(git -C "$HOTFIX_REPO" status --porcelain) ]] || {
  echo 'S128_IT_RECOVERY_NO_GO: hotfix worktree is dirty' >&2; exit 62;
}
[[ $(git -C "$ORIGINAL_REPO" rev-parse HEAD) == "$ORIGINAL_COMMIT" ]] || {
  echo 'S128_IT_RECOVERY_NO_GO: original evaluation HEAD changed' >&2; exit 61;
}
[[ $(git -C "$ORIGINAL_REPO" branch --show-current) == h20/qwen25-7b-original-t25-s128-frozen-20260821 ]] || {
  echo 'S128_IT_RECOVERY_NO_GO: original evaluation branch changed' >&2; exit 60;
}
[[ -z $(git -C "$ORIGINAL_REPO" status --porcelain) ]] || {
  echo 'S128_IT_RECOVERY_NO_GO: original evaluation worktree is dirty' >&2; exit 59;
}

command -v flock >/dev/null || { echo 'S128_IT_RECOVERY_NO_GO: flock is required' >&2; exit 58; }
mkdir -p "$(dirname "$LOCK")"
exec 8>"$LOCK"
flock -n 8 || { echo 'S128_IT_RECOVERY_NO_GO: GPU6-7 project lock is held' >&2; exit 57; }

require_idle() {
  local processes
  processes=$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${processes//[[:space:]]/} ]] || {
    echo "S128_IT_RECOVERY_NO_GO: GPU6-7 are busy: $processes" >&2; exit 56;
  }
}

wait_cleanup() {
  local poll processes
  for poll in $(seq 1 45); do
    processes=$(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z ${processes//[[:space:]]/} ]] && return 0
    sleep 2
  done
  echo "S128_IT_RECOVERY_NO_GO: GPU6-7 did not become idle: $processes" >&2
  exit 55
}

readonly state=$(
  LEDGER_PATH=$LEDGER "$PY" -c '
import json, os
rows=[json.loads(x) for x in open(os.environ["LEDGER_PATH"]) if x.strip()]
print(json.dumps([[r.get("record_type"), r.get("interface_id")] for r in rows], separators=(",",":")))
'
)
require_idle
case "$state" in
  '[["s0_preflight",null],["interface_start","I"]]')
    echo 'Authenticating the completed Base-I artifacts; Base-I will not be rerun.'
    "$PY" "$PREFLIGHT" --manifest "$MANIFEST" \
      --record-interface-event finish --interface I
    ;;
  '[["s0_preflight",null],["interface_start","I"],["interface_finish","I"]]')
    echo 'Base-I finish is already authenticated; continuing without rerun.'
    ;;
  *)
    echo "S128_IT_RECOVERY_NO_GO: unsupported ledger state $state" >&2
    exit 54
    ;;
esac

[[ ! -e $T_ROOT ]] || {
  echo "S128_IT_RECOVERY_NO_GO: T25 path already exists: $T_ROOT" >&2; exit 53;
}
require_idle

readonly RAY_TMP=/tmp/ms128_${UID}_T25_recovery_$$
readonly OVERRIDES=$RAY_TMP/trainer_overrides.txt
readonly RUN_LOG=$T_ROOT/run.log
mkdir -p "$RAY_TMP"
"$PY" "$PREFLIGHT" --manifest "$MANIFEST" \
  --emit-trainer-overrides --interface T25 >"$OVERRIDES"
mapfile -t trainer_overrides <"$OVERRIDES"
[[ ${#trainer_overrides[@]} -gt 0 ]] || {
  echo 'S128_IT_RECOVERY_NO_GO: empty frozen T25 argv' >&2; exit 52;
}
"$PY" "$PREFLIGHT" --manifest "$MANIFEST" \
  --record-interface-event start --interface T25
mkdir -p "$T_ROOT"

while IFS= read -r inherited_gate_a_variable; do
  unset "$inherited_gate_a_variable"
done < <(compgen -v GATE_A_ || true)
unset ORIGINAL_T25_EXPECTED_RUNTIME_CONFIG_SHA256
unset ORIGINAL_T25_TRAINER_OVERRIDE_ARGV_SHA256
export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
export GATE_A_WEIGHT_DIGEST_SAMPLES=256
unset RAY_ADDRESS
export CUDA_VISIBLE_DEVICES=6,7
export RAY_TMPDIR=$RAY_TMP TMPDIR=$RAY_TMP
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline VLLM_USE_V1=0 NCCL_DEBUG=WARN

cd "$ORIGINAL_REPO"
"$PY" -m verl.trainer.main_ppo "${trainer_overrides[@]}" 2>&1 | tee -a "$RUN_LOG"
wait_cleanup

"$PY" "$PREFLIGHT" --manifest "$MANIFEST" \
  --record-interface-event finish --interface T25
"$PY" "$AUDIT" --manifest "$MANIFEST" --write-report
"$PY" "$AUDIT" --manifest "$MANIFEST"

echo 'S128_IT_RECOVERY_PASS: reused completed Base-I, ran only T25, and certified the paired result'

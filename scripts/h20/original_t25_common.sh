#!/usr/bin/env bash
set -euo pipefail

[[ -n ${MEMAGENT_T25_WORK_ROOT:-} ]] || {
  echo 'ORIGINAL_T25_NO_GO:P0 set MEMAGENT_T25_WORK_ROOT explicitly' >&2; exit 66;
}
[[ -n ${MEMAGENT_T25_REPO_DIR:-} ]] || {
  echo 'ORIGINAL_T25_NO_GO:P0 set MEMAGENT_T25_REPO_DIR explicitly' >&2; exit 67;
}
[[ ${MEMAGENT_T25_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'ORIGINAL_T25_NO_GO:P0 MEMAGENT_T25_EXPECTED_COMMIT must be a full Git SHA' >&2; exit 65;
}
[[ $MEMAGENT_T25_WORK_ROOT == /* && $MEMAGENT_T25_REPO_DIR == /* ]] || {
  echo 'ORIGINAL_T25_NO_GO:P0 task-scoped paths must be absolute' >&2; exit 69;
}

readonly T25_WORK_ROOT=$MEMAGENT_T25_WORK_ROOT
readonly T25_REPO_DIR=$MEMAGENT_T25_REPO_DIR
readonly T25_EXPECTED_COMMIT=$MEMAGENT_T25_EXPECTED_COMMIT
export MEMAGENT_T25_WORK_ROOT MEMAGENT_T25_REPO_DIR MEMAGENT_T25_EXPECTED_COMMIT
readonly T25_PYTHON=$T25_WORK_ROOT/.venv/bin/python
readonly T25_BRANCH=h20/qwen25-7b-original-t25-s128-frozen-20260821
readonly T25_MANIFEST=$T25_REPO_DIR/manifests/h20/qwen25_7b_original_t25_seed2026.json
readonly T25_EXPERIMENT=qwen25_7b_h20_corrected_original_style_2gpu_pilot_resume3to25_strictvllm_naive_indseed_seed2026_20260821
readonly T25_SOURCE=$T25_WORK_ROOT/logs/memory_agent/qwen25_7b_h20_2gpu_gatea_resume2to3_strictvllm_naive_indseed_seed2026_20260821r5/global_step_3
readonly T25_OUTPUT=$T25_WORK_ROOT/logs/memory_agent/$T25_EXPERIMENT
readonly T25_LOG_ROOT=$T25_WORK_ROOT/logs/original_t25_2gpu_frozen_20260821
readonly T25_CERT_ROOT=$T25_LOG_ROOT/certificates
readonly T25_P0=$T25_CERT_ROOT/p0_preflight.json
readonly T25_FINAL=$T25_CERT_ROOT/original_t25_final_report.json
readonly T25_LEDGER=$T25_LOG_ROOT/original_t25_execution_ledger.jsonl
readonly T25_LOG=$T25_LOG_ROOT/$T25_EXPERIMENT.log
readonly T25_GPU_DECLARATION=6,7
readonly T25_LOCK=$T25_WORK_ROOT/locks/memagent_gate_a_gpu_6_7.lock
readonly T25_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight

t25_require_clean_checkout() {
  local invoked_repo
  invoked_repo=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
  [[ $(cd -- "$T25_REPO_DIR" && pwd -P) == "$invoked_repo" ]] || {
    echo 'ORIGINAL_T25_NO_GO:P0 invoked checkout differs from MEMAGENT_T25_REPO_DIR' >&2; exit 68;
  }
  [[ $(cd "$T25_REPO_DIR" && git branch --show-current) == "$T25_BRANCH" ]] || {
    echo 'ORIGINAL_T25_NO_GO:P0 wrong branch' >&2; exit 70;
  }
  [[ -z $(cd "$T25_REPO_DIR" && git status --porcelain) ]] || {
    echo 'ORIGINAL_T25_NO_GO:P0 dirty worktree' >&2; exit 71;
  }
  [[ $(cd "$T25_REPO_DIR" && git rev-parse HEAD) == "$T25_EXPECTED_COMMIT" ]] || {
    echo 'ORIGINAL_T25_NO_GO:P0 HEAD differs from MEMAGENT_T25_EXPECTED_COMMIT' >&2; exit 64;
  }
}

t25_acquire_run_lock() {
  command -v flock >/dev/null || { echo 'ORIGINAL_T25_NO_GO:P0 flock missing' >&2; exit 63; }
  mkdir -p "$(dirname "$T25_LOCK")"
  exec 9>"$T25_LOCK"
  flock -n 9 || {
    echo "ORIGINAL_T25_NO_GO:P0 another T25 process owns $T25_LOCK" >&2; exit 62;
  }
}

t25_require_p0() {
  [[ -f $T25_P0 ]] || { echo 'ORIGINAL_T25_NO_GO:P0 run standalone P0 first' >&2; exit 61; }
  "$T25_PYTHON" -c '
import json,re,sys
r=json.load(open(sys.argv[1])); e=r.get("evidence",{})
ok=(r.get("status")=="PASS" and r.get("decision")=="T25_P0_PASS"
    and e.get("git_commit")==sys.argv[2] and e.get("expected_git_commit")==sys.argv[2]
    and re.fullmatch(r"[0-9a-f]{32}",str(e.get("run_id",""))) is not None
    and re.fullmatch(r"[0-9a-f]{64}",str(e.get("resolved_trainer_config_sha256",""))) is not None
    and re.fullmatch(r"[0-9a-f]{64}",str(e.get("trainer_override_argv_sha256",""))) is not None)
raise SystemExit(0 if ok else 1)
' "$T25_P0" "$T25_EXPECTED_COMMIT" || {
    echo 'ORIGINAL_T25_NO_GO:P0 certificate is not bound to this exact config/commit' >&2; exit 80;
  }
}

t25_require_gpus_idle() {
  command -v nvidia-smi >/dev/null || { echo 'ORIGINAL_T25_NO_GO:P0 nvidia-smi missing' >&2; exit 78; }
  local apps
  apps=$(nvidia-smi -i "$T25_GPU_DECLARATION" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${apps//[[:space:]]/} ]] || {
    echo "ORIGINAL_T25_NO_GO:P0 GPU6-7 are not idle; no process changed: $apps" >&2; exit 79;
  }
}

t25_export_execution_evidence() {
  local inherited_gate_a_variable
  while IFS= read -r inherited_gate_a_variable; do
    unset "$inherited_gate_a_variable"
  done < <(compgen -v GATE_A_ || true)
  unset MEMAGENT_STABLE_I_WORK_ROOT MEMAGENT_STABLE_I_REPO_DIR \
    MEMAGENT_STABLE_I_EXPECTED_COMMIT MEMAGENT_S128_IT_WORK_ROOT \
    MEMAGENT_S128_IT_REPO_DIR MEMAGENT_S128_IT_EXPECTED_COMMIT \
    ORIGINAL_T25_EXPECTED_RUNTIME_CONFIG_SHA256 \
    ORIGINAL_T25_TRAINER_OVERRIDE_ARGV_SHA256
  export GATE_A_FROZEN_AUDIT=1
  export GATE_A_EXECUTION_LEDGER=$T25_LEDGER
  export GATE_A_EXPERIMENT_NAME=$T25_EXPERIMENT
  export GATE_A_GIT_COMMIT=$T25_EXPECTED_COMMIT
  export GATE_A_RUN_ID
  GATE_A_RUN_ID=$(
    "$T25_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["evidence"]["run_id"])' "$T25_P0"
  )
  export GATE_A_WEIGHT_DIGEST_PARAMETERS=$T25_DIGEST_PARAMETERS
  export GATE_A_WEIGHT_DIGEST_SAMPLES=256
  export ORIGINAL_T25_EXPECTED_RUNTIME_CONFIG_SHA256
  ORIGINAL_T25_EXPECTED_RUNTIME_CONFIG_SHA256=$(
    "$T25_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["evidence"]["resolved_trainer_config_sha256"])' "$T25_P0"
  )
  export ORIGINAL_T25_TRAINER_OVERRIDE_ARGV_SHA256
  ORIGINAL_T25_TRAINER_OVERRIDE_ARGV_SHA256=$(
    "$T25_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["evidence"]["trainer_override_argv_sha256"])' "$T25_P0"
  )
  export CUDA_VISIBLE_DEVICES=$T25_GPU_DECLARATION
}

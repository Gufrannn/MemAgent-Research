#!/usr/bin/env bash
set -euo pipefail

[[ -n ${WORK_ROOT:-} ]] || {
  echo 'GATE_A_NO_GO:P0 WORK_ROOT must be explicitly set; no runtime path is selected automatically' >&2; exit 66;
}
[[ -n ${REPO_DIR:-} ]] || {
  echo 'GATE_A_NO_GO:P0 REPO_DIR must be explicitly set; resolve the repository-path conflict before P0' >&2; exit 67;
}
[[ $WORK_ROOT == /* && $REPO_DIR == /* ]] || {
  echo 'GATE_A_NO_GO:P0 WORK_ROOT and REPO_DIR must be absolute paths' >&2; exit 69;
}
readonly WORK_ROOT
readonly REPO_DIR
readonly CODE=$REPO_DIR
readonly PYTHON=$WORK_ROOT/.venv/bin/python
readonly MANIFEST=$CODE/manifests/h20/qwen25_7b_gatea_seed2026.yaml
readonly LOG_ROOT=$WORK_ROOT/logs/gate_a_2gpu_frozen_20260820
readonly CERTIFICATE_ROOT=$LOG_ROOT/certificates
readonly EXECUTION_LEDGER=$LOG_ROOT/gate_a_execution_ledger.jsonl
readonly FRESH_EXP=qwen25_7b_h20_2gpu_gatea_fresh2_strictvllm_naive_indseed_seed2026_20260820
readonly RESUME_EXP=qwen25_7b_h20_2gpu_gatea_resume2to3_strictvllm_naive_indseed_seed2026_20260820
readonly FRESH_OUTPUT=$WORK_ROOT/logs/memory_agent/$FRESH_EXP
readonly RESUME_OUTPUT=$WORK_ROOT/logs/memory_agent/$RESUME_EXP
readonly RESUME_SOURCE=$FRESH_OUTPUT/global_step_2
readonly FROZEN_GPU_DECLARATION=6,7
readonly DIGEST_PARAMETERS=model.layers.0.input_layernorm.weight,model.layers.0.post_attention_layernorm.weight,model.layers.27.input_layernorm.weight,model.layers.27.post_attention_layernorm.weight

gatea_require_clean_frozen_checkout() {
  local invoked_repo
  invoked_repo=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
  [[ $(cd -- "$CODE" && pwd -P) == "$invoked_repo" ]] || {
    echo "GATE_A_NO_GO:P0 invoked checkout differs from explicit REPO_DIR: $invoked_repo != $CODE" >&2; exit 68;
  }
  [[ $(cd "$CODE" && git branch --show-current) == h20/qwen25-7b-gatea-2gpu-frozen-20260820 ]] || {
    echo 'GATE_A_NO_GO:P0 wrong branch' >&2; exit 70;
  }
  [[ -z $(cd "$CODE" && git status --porcelain) ]] || {
    echo 'GATE_A_NO_GO:P0 dirty worktree' >&2; exit 71;
  }
}

gatea_export_audit_environment() {
  export GATE_A_FROZEN_AUDIT=1
  export GATE_A_EXECUTION_LEDGER=$EXECUTION_LEDGER
  export GATE_A_GIT_COMMIT
  GATE_A_GIT_COMMIT=$(cd "$CODE" && git rev-parse HEAD)
  export GATE_A_WEIGHT_DIGEST_PARAMETERS=$DIGEST_PARAMETERS
  export GATE_A_WEIGHT_DIGEST_SAMPLES=256
  export CUDA_VISIBLE_DEVICES=$FROZEN_GPU_DECLARATION
}

gatea_require_declared_gpus_idle() {
  command -v nvidia-smi >/dev/null || { echo 'GATE_A_NO_GO:P0 nvidia-smi missing' >&2; exit 78; }
  local apps
  apps=$(nvidia-smi -i "$FROZEN_GPU_DECLARATION" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${apps//[[:space:]]/} ]] || {
    echo "GATE_A_NO_GO:P0 declared GPU set is not idle; no process was changed: $apps" >&2
    exit 79
  }
}

gatea_require_p0_commit() {
  local current
  current=$(cd "$CODE" && git rev-parse HEAD)
  "$PYTHON" -c 'import json,sys; data=json.load(open(sys.argv[1])); sys.exit(data["status"] != "PASS" or data["evidence"]["git_commit"] != sys.argv[2])' \
    "$CERTIFICATE_ROOT/p0_preflight.json" "$current" || {
      echo 'GATE_A_NO_GO:P0 current commit differs from the P0 frozen commit' >&2
      exit 80
    }
}

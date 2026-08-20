#!/usr/bin/env bash
set -euo pipefail

[[ -n ${MEMAGENT_GATEA_WORK_ROOT:-} ]] || {
  echo 'GATE_A_NO_GO:P0 MEMAGENT_GATEA_WORK_ROOT must be explicitly set; no runtime path is selected automatically' >&2; exit 66;
}
[[ -n ${MEMAGENT_GATEA_REPO_DIR:-} ]] || {
  echo 'GATE_A_NO_GO:P0 MEMAGENT_GATEA_REPO_DIR must be explicitly set; resolve the repository-path conflict before P0' >&2; exit 67;
}
[[ $MEMAGENT_GATEA_WORK_ROOT == /* && $MEMAGENT_GATEA_REPO_DIR == /* ]] || {
  echo 'GATE_A_NO_GO:P0 MEMAGENT_GATEA_WORK_ROOT and MEMAGENT_GATEA_REPO_DIR must be absolute paths' >&2; exit 69;
}
readonly GATEA_WORK_ROOT=$MEMAGENT_GATEA_WORK_ROOT
readonly GATEA_REPO_DIR=$MEMAGENT_GATEA_REPO_DIR
readonly GATEA_CODE=$GATEA_REPO_DIR
readonly GATEA_PYTHON=$GATEA_WORK_ROOT/.venv/bin/python
readonly GATEA_MANIFEST=$GATEA_CODE/manifests/h20/qwen25_7b_gatea_seed2026.yaml
readonly GATEA_LOG_ROOT=$GATEA_WORK_ROOT/logs/gate_a_2gpu_frozen_20260821r2
readonly GATEA_CERTIFICATE_ROOT=$GATEA_LOG_ROOT/certificates
readonly GATEA_EXECUTION_LEDGER=$GATEA_LOG_ROOT/gate_a_execution_ledger.jsonl
readonly GATEA_FRESH_EXP=qwen25_7b_h20_2gpu_gatea_fresh2_strictvllm_naive_indseed_seed2026_20260821r2
readonly GATEA_RESUME_EXP=qwen25_7b_h20_2gpu_gatea_resume2to3_strictvllm_naive_indseed_seed2026_20260821r2
readonly GATEA_FRESH_OUTPUT=$GATEA_WORK_ROOT/logs/memory_agent/$GATEA_FRESH_EXP
readonly GATEA_RESUME_OUTPUT=$GATEA_WORK_ROOT/logs/memory_agent/$GATEA_RESUME_EXP
readonly GATEA_RESUME_SOURCE=$GATEA_FRESH_OUTPUT/global_step_2
readonly GATEA_FROZEN_GPU_DECLARATION=6,7
readonly GATEA_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight

gatea_require_clean_frozen_checkout() {
  local invoked_repo
  invoked_repo=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
  [[ $(cd -- "$GATEA_CODE" && pwd -P) == "$invoked_repo" ]] || {
    echo "GATE_A_NO_GO:P0 invoked checkout differs from explicit MEMAGENT_GATEA_REPO_DIR: $invoked_repo != $GATEA_CODE" >&2; exit 68;
  }
  [[ $(cd "$GATEA_CODE" && git branch --show-current) == h20/qwen25-7b-gatea-2gpu-frozen-20260820 ]] || {
    echo 'GATE_A_NO_GO:P0 wrong branch' >&2; exit 70;
  }
  [[ -z $(cd "$GATEA_CODE" && git status --porcelain) ]] || {
    echo 'GATE_A_NO_GO:P0 dirty worktree' >&2; exit 71;
  }
}

gatea_export_audit_environment() {
  export GATE_A_FROZEN_AUDIT=1
  export GATE_A_EXECUTION_LEDGER=$GATEA_EXECUTION_LEDGER
  export GATE_A_GIT_COMMIT
  GATE_A_GIT_COMMIT=$(cd "$GATEA_CODE" && git rev-parse HEAD)
  export GATE_A_WEIGHT_DIGEST_PARAMETERS=$GATEA_DIGEST_PARAMETERS
  export GATE_A_WEIGHT_DIGEST_SAMPLES=256
  export CUDA_VISIBLE_DEVICES=$GATEA_FROZEN_GPU_DECLARATION
}

gatea_require_declared_gpus_idle() {
  command -v nvidia-smi >/dev/null || { echo 'GATE_A_NO_GO:P0 nvidia-smi missing' >&2; exit 78; }
  local apps
  apps=$(nvidia-smi -i "$GATEA_FROZEN_GPU_DECLARATION" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${apps//[[:space:]]/} ]] || {
    echo "GATE_A_NO_GO:P0 declared GPU set is not idle; no process was changed: $apps" >&2
    exit 79
  }
}

gatea_require_p0_commit() {
  local current
  current=$(cd "$GATEA_CODE" && git rev-parse HEAD)
  "$GATEA_PYTHON" -c 'import json,sys; data=json.load(open(sys.argv[1])); sys.exit(data["status"] != "PASS" or data["evidence"]["git_commit"] != sys.argv[2])' \
    "$GATEA_CERTIFICATE_ROOT/p0_preflight.json" "$current" || {
      echo 'GATE_A_NO_GO:P0 current commit differs from the P0 frozen commit' >&2
      exit 80
    }
}

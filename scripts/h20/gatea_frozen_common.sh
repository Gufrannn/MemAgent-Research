#!/usr/bin/env bash
set -euo pipefail

readonly WORK_ROOT=/data/cw/memagent_work
readonly CODE=/data/cw/memagent_work/MemAgent-Research
readonly PYTHON=/data/cw/memagent_work/.venv/bin/python
readonly MANIFEST=$CODE/manifests/h20/qwen25_7b_gatea_seed2026.yaml
readonly LOG_ROOT=$WORK_ROOT/logs/gate_a_frozen_20260820
readonly CERTIFICATE_ROOT=$LOG_ROOT/certificates
readonly EXECUTION_LEDGER=$LOG_ROOT/gate_a_execution_ledger.jsonl
readonly FRESH_EXP=qwen25_7b_h20_gatea_fresh2_strictvllm_naive_indseed_seed2026_20260820
readonly RESUME_EXP=qwen25_7b_h20_gatea_resume2to3_strictvllm_naive_indseed_seed2026_20260820
readonly FRESH_OUTPUT=$WORK_ROOT/logs/memory_agent/$FRESH_EXP
readonly RESUME_OUTPUT=$WORK_ROOT/logs/memory_agent/$RESUME_EXP
readonly RESUME_SOURCE=$FRESH_OUTPUT/global_step_2
readonly FROZEN_GPU_DECLARATION=4,5,6,7
readonly DIGEST_PARAMETERS=model.layers.0.input_layernorm.weight,model.layers.0.post_attention_layernorm.weight,model.layers.27.input_layernorm.weight,model.layers.27.post_attention_layernorm.weight

gatea_require_clean_frozen_checkout() {
  [[ $(cd "$CODE" && git branch --show-current) == h20/qwen25-7b-gatea-frozen-20260820 ]] || {
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

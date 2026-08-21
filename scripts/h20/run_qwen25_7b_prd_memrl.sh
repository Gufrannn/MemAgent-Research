#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$SCRIPT_DIR/prd_memrl_common.sh"
prd_require_env
prd_paths

readonly ACTION=${1:-}
readonly E0_CERT=$PRD_CERT_ROOT/e0.json
readonly E1_CERT=$PRD_CERT_ROOT/e1.json
readonly PAPER_CERT=$PRD_CERT_ROOT/paper_review.json
readonly P0_CERT=$PRD_CERT_ROOT/p0.json

case "$ACTION" in
  e0)
    [[ ! -e $E0_CERT ]] || prd_die 'E0 certificate already exists; use a new RUN_ID'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_gate.py" e0 --output "$E0_CERT"
    ;;
  e1)
    [[ ${E1_ROWS:-} == /* && -f ${E1_ROWS:-} ]] || prd_die 'E1_ROWS must name frozen Original JSONL'
    [[ ! -e $E1_CERT ]] || prd_die 'E1 certificate already exists; use a new RUN_ID'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_gate.py" e1 --rows "$E1_ROWS" --output "$E1_CERT"
    ;;
  preflight)
    [[ ! -e $P0_CERT ]] || prd_die 'P0 certificate already exists; use a new RUN_ID'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_gate.py" preflight \
      --expected-commit "$EXPECTED_COMMIT" --gpu-pair "$GPU_PAIR" --e0 "$E0_CERT" \
      --e1 "$E1_CERT" --paper-review "$PAPER_CERT" --output "$P0_CERT"
    ;;
  train-t5|continue-t25)
    prd_die 'RELEASE_NO_GO: production prior worker, dual checkpoint, and reviewer GO are not yet implemented'
    [[ ${CAPACITY_NATS:-} =~ ^[0-9]+([.][0-9]+)?$ ]] || prd_die 'CAPACITY_NATS must be explicit'
    case ",$CAPACITY_NATS," in ,128.0,|,256.0,|,512.0,) ;; *) prd_die 'CAPACITY_NATS is not in the frozen frontier' ;; esac
    prd_verify_gate "$P0_CERT" PRD_P0_PASS
    if [[ $ACTION == continue-t25 ]]; then
      prd_verify_gate "$PRD_CERT_ROOT/t5_gate.json" PRD_T5_GATE_PASS
      [[ ${RESUME_CHECKPOINT:-} == "$PRD_RUN_ROOT"/capacity_*/global_step_5 ]] || prd_die 'continuation must resume this run exact step 5'
    else
      [[ -z ${RESUME_CHECKPOINT:-} ]] || prd_die 'Method-T5 must start fresh; resume/warm-start forbidden'
    fi
    prd_acquire_gpu_locks
    export CUDA_VISIBLE_DEVICES=$GPU_PAIR
    resume_mode=disable
    [[ $ACTION == continue-t25 ]] && resume_mode=resume_path
    exec "$PRD_PYTHON" -m verl.trainer.main_ppo \
      --config-path="$PRD_REPO/verl/trainer/config" --config-name=ppo_trainer \
      "actor_rollout_ref.model.path=$WORK_ROOT/models/Qwen2.5-7B-Instruct" \
      actor_rollout_ref.actor.prd_memrl.enable=true \
      "actor_rollout_ref.actor.prd_memrl.capacity_nats=$CAPACITY_NATS" \
      actor_rollout_ref.actor.prd_memrl.dual_value=0.0 \
      "trainer.resume_mode=$resume_mode" \
      "trainer.resume_from_path=${RESUME_CHECKPOINT:-}" \
      "trainer.total_training_steps=$([[ $ACTION == train-t5 ]] && echo 5 || echo 25)" \
      "trainer.default_local_dir=$PRD_RUN_ROOT/training" \
      "trainer.experiment_name=$RUN_ID"
    ;;
  *) prd_die 'action must be e0, e1, preflight, train-t5, or continue-t25' ;;
esac

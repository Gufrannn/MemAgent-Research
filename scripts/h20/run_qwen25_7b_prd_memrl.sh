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
    [[ ${PRD_PRIOR_MODEL:-} == /* ]] || prd_die 'PRD_PRIOR_MODEL must be explicit for P0'
    [[ ! -e $P0_CERT ]] || prd_die 'P0 certificate already exists; use a new RUN_ID'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_gate.py" preflight \
      --expected-commit "$EXPECTED_COMMIT" --gpu-pair "$GPU_PAIR" --e0 "$E0_CERT" \
      --e1 "$E1_CERT" --paper-review "$PAPER_CERT" --prior-model "$PRD_PRIOR_MODEL" --output "$P0_CERT"
    ;;
  bind)
    [[ ${BASELINE_CERT:-} == /* && -f ${BASELINE_CERT:-} ]] || prd_die 'BASELINE_CERT must name the read-only certified import'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_orchestrator.py" bind \
      --run-root "$PRD_RUN_ROOT" --run-id "$RUN_ID" --commit "$EXPECTED_COMMIT" \
      --gpu-pair "$GPU_PAIR" --baseline "$BASELINE_CERT" --p0 "$P0_CERT"
    ;;
  prepare-t5)
    [[ ${CAPACITY_NATS:-} =~ ^[0-9]+([.][0-9]+)?$ ]] || prd_die 'CAPACITY_NATS must be explicit'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_orchestrator.py" stage t5 \
      --run-root "$PRD_RUN_ROOT" --run-id "$RUN_ID" --commit "$EXPECTED_COMMIT" --capacity "$CAPACITY_NATS"
    ;;
  prepare-continuation)
    [[ ${CAPACITY_NATS:-} =~ ^[0-9]+([.][0-9]+)?$ ]] || prd_die 'CAPACITY_NATS must be explicit'
    [[ ${RESUME_CHECKPOINT:-} == /* ]] || prd_die 'RESUME_CHECKPOINT must be explicit'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_orchestrator.py" stage continue \
      --run-root "$PRD_RUN_ROOT" --run-id "$RUN_ID" --commit "$EXPECTED_COMMIT" \
      --capacity "$CAPACITY_NATS" --resume "$RESUME_CHECKPOINT"
    ;;
  evaluate)
    [[ ${CAPACITY_NATS:-} =~ ^[0-9]+([.][0-9]+)?$ ]] || prd_die 'CAPACITY_NATS must be explicit'
    [[ ${EVAL_INPUT_TEMPLATE:-} == /* && ${EVAL_INPUT_TEMPLATE:-} == *'{anchor}'* ]] || prd_die 'EVAL_INPUT_TEMPLATE must be absolute and contain {anchor}'
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_orchestrator.py" evaluate \
      --run-root "$PRD_RUN_ROOT" --capacity "$CAPACITY_NATS" --input-template "$EVAL_INPUT_TEMPLATE" \
      --anchors "${EVAL_ANCHORS:-5,10,15,20,25}"
    ;;
  t5-gate)
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_orchestrator.py" t5-gate --run-root "$PRD_RUN_ROOT"
    ;;
  final-audit)
    "$PRD_PYTHON" "$PRD_REPO/tools/h20/prd_memrl_orchestrator.py" audit \
      --run-root "$PRD_RUN_ROOT" --ledger "$PRD_LEDGER" --output "$PRD_CERT_ROOT/final_audit.json"
    ;;
  train-t5|continue-t25)
    [[ ${CAPACITY_NATS:-} =~ ^[0-9]+([.][0-9]+)?$ ]] || prd_die 'CAPACITY_NATS must be explicit'
    case ",$CAPACITY_NATS," in ,128.0,|,256.0,|,512.0,) ;; *) prd_die 'CAPACITY_NATS is not in the frozen frontier' ;; esac
    prd_verify_gate "$P0_CERT" PRD_P0_PASS
    if [[ $ACTION == continue-t25 ]]; then
      prd_verify_gate "$PRD_CERT_ROOT/t5_gate.json" PRD_T5_GATE_PASS
      cid=c${CAPACITY_NATS%.*}
      [[ ${RESUME_CHECKPOINT:-} == "$PRD_RUN_ROOT/frontier/$cid/checkpoints/global_step_5" ]] || prd_die 'continuation must resume this run/capacity exact step 5'
    else
      [[ -z ${RESUME_CHECKPOINT:-} ]] || prd_die 'Method-T5 must start fresh; resume/warm-start forbidden'
    fi
    prd_acquire_gpu_locks
    export CUDA_VISIBLE_DEVICES=$GPU_PAIR
    [[ ${PRD_PRIOR_MODEL:-} == /* ]] || prd_die 'PRD_PRIOR_MODEL must be explicit'
    cid=c${CAPACITY_NATS%.*}
    stage=t5
    [[ $ACTION == continue-t25 ]] && stage=continue
    "$PRD_PYTHON" - "$PRD_RUN_ROOT/resolved_run.json" "$PRD_RUN_ROOT/frontier/$cid/launch_${stage}.json" "$RUN_ID" "$EXPECTED_COMMIT" "$GPU_PAIR" "$CAPACITY_NATS" "$PRD_PRIOR_MODEL" <<'PY'
import hashlib,json,pathlib,sys
run=json.load(open(sys.argv[1])); launch=json.load(open(sys.argv[2]))
assert run["run_id"]==sys.argv[3] and run["git_commit"]==sys.argv[4]
assert run["gpu_pair"]==sys.argv[5] and launch["gpu_pair"]==sys.argv[5]
assert float(launch["capacity_nats"])==float(sys.argv[6])
assert launch["frontier_id"]=="c"+str(int(float(sys.argv[6])))
prior=pathlib.Path(sys.argv[7]).resolve(); frozen=run["prior_model"]
assert str(prior)==frozen["path"] and launch["prior_model"]==frozen
for item in frozen["files"]:
 p=prior/item["path"]; assert p.is_file() and not p.is_symlink() and p.stat().st_size==item["size"]
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item["sha256"]
PY
    phase=fresh
    [[ $ACTION == continue-t25 ]] && phase=resume
    exec env WORK_ROOT="$WORK_ROOT" CODE="$PRD_REPO" PYTHON="$PRD_PYTHON" \
      PHASE="$phase" EXP="${RUN_ID}_${cid}" RUN_SEED=2026 \
      TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 N_GPUS=2 FSDP_SIZE=2 \
      REWARD_MANAGER=naive GPU_MEMORY_UTILIZATION=0.55 \
      FRESH_TOTAL_STEPS=5 RESUME_TOTAL_STEPS=25 RESUME_SOURCE_STEP=5 \
      RESUME_FROM="${RESUME_CHECKPOINT:-}" SAVE_FREQ=5 MAX_ACTOR_CKPT_TO_KEEP=5 \
      OUTPUT_ROOT="$PRD_RUN_ROOT/frontier/$cid/checkpoints" \
      PRD_MEMRL_ENABLE=1 PRD_MEMRL_CAPACITY="$CAPACITY_NATS" \
      PRD_RUN_ID="$RUN_ID" PRD_FRONTIER_ID="$cid" PRD_GIT_COMMIT="$EXPECTED_COMMIT" \
      PRD_EXECUTION_LEDGER="$PRD_LEDGER" \
      PRD_PRIOR_MODEL="$PRD_PRIOR_MODEL" \
      bash "$PRD_REPO/experiments/7b_gate_a/run_gate_a.sh"
    ;;
  *) prd_die 'unknown action (expected e0/e1/preflight/bind/prepare-t5/prepare-continuation/evaluate/t5-gate/final-audit/train-t5/continue-t25)' ;;
esac

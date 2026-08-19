#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE=${CODE:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
PYTHON=${PYTHON:-python}
IDEA_ARM=${IDEA_ARM:-qa_only_original}
PHASE=${PHASE:-fresh2}
RUN_SEED=${RUN_SEED:-2026}
N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
WORK_ROOT=${WORK_ROOT:-$CODE/runs}
TRAIN=${TRAIN:?Set TRAIN to the existing project HotpotQA training parquet}
VAL=${VAL:?Set VAL to the existing project HotpotQA validation parquet}
EXP=${EXP:?Set a unique append-only experiment name}
OUT=${OUT:-$WORK_ROOT/$EXP}
IDEA_EVIDENCE_LEDGER=${IDEA_EVIDENCE_LEDGER:-}
IDEA_RUN_LEDGER=${IDEA_RUN_LEDGER:-$WORK_ROOT/idea_run_ledger.jsonl}
IDEA_REWARD_AUDIT=${IDEA_REWARD_AUDIT:-$OUT/reward_tuple_audit.jsonl}
EXACT_NOOP_V2_MANIFEST=${EXACT_NOOP_V2_MANIFEST:-}
SHAPEA_HORIZON_PRIMARY_MANIFEST=${SHAPEA_HORIZON_PRIMARY_MANIFEST:-}
STOP_RULE_MANIFEST=${STOP_RULE_MANIFEST:-}

if [[ ${W4_GRADIENT_PILOT_REQUEST:-0} == 1 || ${W4_OPTIMIZER_STEPS:-0} != 0 ]]; then
  [[ -n ${IDEA_EVIDENCE_LEDGER:-} && -f ${IDEA_EVIDENCE_LEDGER:-/nonexistent} ]] || { echo "PENDING_EVIDENCE_NO_SELECTION: W4 request requires evidence ledger" >&2; exit 77; }
  [[ -n ${MECHANISM_EXTENSION_DECISION:-} && -f ${MECHANISM_EXTENSION_DECISION:-/nonexistent} ]] || { echo "PENDING_NO_EXTENSION: W4 request requires single-extension router decision" >&2; exit 78; }
  echo "W4_NO_GO: gradient pilot and optimizer steps are not currently authorized; highest_level=W3" >&2
  exit 79
fi

if [[ ${MEMORY_R2_BASELINE_REQUEST:-0} == 1 ]]; then
  [[ -n ${IDEA_EVIDENCE_LEDGER:-} && -f ${IDEA_EVIDENCE_LEDGER:-/nonexistent} ]] || { echo "PENDING_EVIDENCE_NO_SELECTION: Memory-R2-like request requires evidence ledger" >&2; exit 71; }
  [[ -n ${MECHANISM_EXTENSION_DECISION:-} && -f ${MECHANISM_EXTENSION_DECISION:-/nonexistent} ]] || { echo "PENDING_NO_EXTENSION: Memory-R2-like request requires single-extension router decision" >&2; exit 72; }
  "$PYTHON" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("decision")=="SELECT_ONE" and not x.get("training_authorized", True)' "$MECHANISM_EXTENSION_DECISION" || { echo "NO_MECHANISM_EXTENSION: router did not select exactly one outcome-blind extension" >&2; exit 73; }
  echo "NO_METHOD: Memory-R2-like/LoGo-GRPO implementation and training are not authorized" >&2
  exit 74
fi

if [[ ${MECHANISM_EXTENSION_REQUEST:-none} != none ]]; then
  echo "PENDING_NO_EXTENSION: training launcher never starts mechanism extensions; run the outcome-blind extension adjudicator first" >&2
  exit 70
fi

[[ "$MODEL" == *Qwen2.5-7B-Instruct* ]] || { echo "Only Qwen2.5-7B-Instruct is supported" >&2; exit 60; }
[[ "$N_GPUS" -eq 8 ]] || { echo "Default scientific target is one node with 8 H20 GPUs; override is smoke-only" >&2; [[ ${ALLOW_SMOKE_GPU_OVERRIDE:-0} == 1 ]] || exit 61; }
[[ "$PHASE" == fresh2 || "$PHASE" == resume3 || "$PHASE" == extended ]] || { echo "PHASE must be fresh2, resume3, or extended" >&2; exit 62; }
if [[ "$PHASE" == extended && ${CONFIRM_EXTENDED_RUN:-0} != 1 ]]; then echo "25+ steps require CONFIRM_EXTENDED_RUN=1" >&2; exit 63; fi
if [[ "$PHASE" == fresh2 && -e "$OUT" ]]; then echo "Refusing to overwrite run directory: $OUT" >&2; exit 64; fi
[[ "$IDEA_ARM" == qa_only_original || -n "$IDEA_EVIDENCE_LEDGER" ]] || { echo "PENDING_EVIDENCE_NO_SELECTION"; exit 3; }
if [[ "$IDEA_ARM" != qa_only_original ]]; then
  : "${IDEA_REWARD_MANIFEST:?Non-Original arms require IDEA_REWARD_MANIFEST}"
  : "${IDEA_MANIFEST_HASH:?Non-Original arms require IDEA_MANIFEST_HASH}"
  [[ ${#IDEA_MANIFEST_HASH} -eq 64 ]] || { echo "NO_METHOD: IDEA_MANIFEST_HASH must be SHA-256" >&2; exit 67; }
  [[ -n "$EXACT_NOOP_V2_MANIFEST" && -f "$EXACT_NOOP_V2_MANIFEST" ]] || { echo "E_QUALIFICATION_FAIL: Non-Original arms require EXACT_NOOP_V2_MANIFEST; legacy replay is ineligible" >&2; exit 75; }
  PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$SCRIPT_DIR/analysis/validate_exact_noop_v2_manifest_20260819.py" --manifest "$EXACT_NOOP_V2_MANIFEST" --require-shape-a-e \
    || { echo "E_QUALIFICATION_FAIL: exact-NOOP v2 replay preflight failed" >&2; exit 76; }
  [[ -n "$SHAPEA_HORIZON_PRIMARY_MANIFEST" && -f "$SHAPEA_HORIZON_PRIMARY_MANIFEST" ]] || { echo "HORIZON_ENDPOINT_SELECTION_INVALID: missing outcome-blind Shape A primary freeze" >&2; exit 82; }
  "$PYTHON" "$SCRIPT_DIR/analysis/validate_shapeA_horizon_primary_20260819.py" --manifest "$SHAPEA_HORIZON_PRIMARY_MANIFEST" \
    || { echo "HORIZON_ENDPOINT_SELECTION_INVALID: Shape A horizon/endpoint primary preflight failed" >&2; exit 83; }
  if [[ "$IDEA_ARM" == ncr_certified_routing || "$IDEA_ARM" == generic_frozen_judge_tournament || "$IDEA_ARM" == information_matched_raw_judge ]]; then
    : "${NCR_FROZEN_READOUT_HASH:?This arm requires a frozen readout SHA-256}"
    [[ ${#NCR_FROZEN_READOUT_HASH} -eq 64 ]] || { echo "NO_METHOD: frozen readout hash must be SHA-256" >&2; exit 68; }
  fi
fi
export IDEA_ARM IDEA_EVIDENCE_LEDGER IDEA_REWARD_MANIFEST IDEA_REWARD_AUDIT IDEA_MANIFEST_HASH NCR_FROZEN_READOUT_HASH EXACT_NOOP_V2_MANIFEST SHAPEA_HORIZON_PRIMARY_MANIFEST
cd "$CODE"
PREFLIGHT_ARGS=(--arm "$IDEA_ARM")
[[ -z "$IDEA_EVIDENCE_LEDGER" ]] || PREFLIGHT_ARGS+=(--evidence-ledger "$IDEA_EVIDENCE_LEDGER")
"$PYTHON" "$SCRIPT_DIR/verify_7b_idea.py" "${PREFLIGHT_ARGS[@]}"

case "$IDEA_ARM" in
  typed_boundary_prompt_control) echo "NO_METHOD: diagnostic-only; use verify --diagnostic-only"; exit 3 ;;
  cerc_native_credit|target_aligned_repair) echo "NO_METHOD: control/concept module has no training authorization"; exit 3 ;;
esac
TOTAL_STEPS=2; RESUME_ARGS=()
if [[ "$PHASE" == resume3 ]]; then
  TOTAL_STEPS=3; RESUME_FROM=${RESUME_FROM:-$OUT/global_step_2}
  [[ -d "$RESUME_FROM/actor" && -f "$RESUME_FROM/data.pt" ]] || { echo "Incomplete source checkpoint: $RESUME_FROM" >&2; exit 65; }
  RESUME_ARGS=(trainer.resume_mode=resume_path trainer.resume_from_path="$RESUME_FROM")
elif [[ "$PHASE" == extended ]]; then
  [[ ${TERMINAL_RULE_FROZEN:-false} == true ]] || { echo "Extended anchors require TERMINAL_RULE_FROZEN=true before metric unblinding" >&2; exit 69; }
  [[ -n "$STOP_RULE_MANIFEST" && -f "$STOP_RULE_MANIFEST" ]] || { echo "NO_METHOD: extended anchors require frozen adaptive stop v4 manifest" >&2; exit 80; }
  "$PYTHON" "$SCRIPT_DIR/analysis/validate_adaptive_stop_rule_v4_20260819.py" --manifest "$STOP_RULE_MANIFEST" \
    || { echo "NO_METHOD: adaptive stop v4 preflight failed" >&2; exit 81; }
  TOTAL_STEPS=${EXTENDED_STEPS:?Set EXTENDED_STEPS explicitly}; [[ "$TOTAL_STEPS" =~ ^(25|50|100|200)$ ]] || { echo "Allowed frozen anchors: 25/50/100/200; never automatic 400" >&2; exit 66; }
fi

mkdir -p "$OUT"
printf '{"run_id":"%s","arm":"%s","lambda":"%s","stratum":"%s","readout_hash":"%s","manifest_hash":"%s","seed_schedule":{"base":%s,"mode":"independent"},"phase":"%s","terminal_rule_frozen":%s}\n' \
 "$EXP" "$IDEA_ARM" "${IDEA_LAMBDA:-0}" "${IDEA_STRATUM:-qa_exact_tie}" "${NCR_FROZEN_READOUT_HASH:-none}" "${IDEA_MANIFEST_HASH:-none}" "$RUN_SEED" "$PHASE" "${TERMINAL_RULE_FROZEN:-false}" >> "$IDEA_RUN_LEDGER"

exec "$PYTHON" -m verl.trainer.main_ppo \
 recurrent.enable=memory algorithm.adv_estimator=grpo algorithm.grpo_use_adv=False \
 actor_rollout_ref.model.path="$MODEL" actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.n="${ROLLOUT_N:-16}" +actor_rollout_ref.rollout.seed="$RUN_SEED" \
 +actor_rollout_ref.rollout.trajectory_seed_mode=independent data.train_files="$TRAIN" data.val_files="$VAL" \
 data.shuffle=False data.train_batch_size="${TRAIN_BATCH_SIZE:-128}" data.max_prompt_length=8192 data.max_response_length=1024 \
 custom_reward_function.path="$CODE/recurrent/research/hotpotqa_dense_reward.py" custom_reward_function.name=compute_score \
 actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-128}" \
 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.fsdp_config.fsdp_size="$N_GPUS" actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 trainer.n_gpus_per_node="$N_GPUS" trainer.nnodes=1 trainer.total_training_steps="$TOTAL_STEPS" \
 trainer.save_freq=1 trainer.test_freq=-1 trainer.experiment_name="$EXP" trainer.default_local_dir="$OUT" \
 "${RESUME_ARGS[@]}" "$@"

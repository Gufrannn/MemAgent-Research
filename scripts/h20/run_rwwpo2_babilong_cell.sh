#!/usr/bin/env bash
set -euo pipefail

for name in RWWPO_WORK_ROOT RWWPO_REPO_DIR RWWPO_EXPECTED_COMMIT \
  RWWPO_TRAINING_COMMIT GPU_PAIR RWWPO_BABILONG_BUNDLE_ROOT \
  RWWPO_BABILONG_BUNDLE_AUDIT RWWPO_BABILONG_BUNDLE_AUDIT_SHA256 \
  RWWPO_BABILONG_MODEL RWWPO_BABILONG_CHECKPOINT \
  RWWPO_BABILONG_CELL RWWPO_BABILONG_STEP RWWPO_BABILONG_PARTITION \
  RWWPO_BABILONG_EXPERIMENT_SEED RWWPO_BABILONG_PIPELINE_ROOT; do
  [[ -n ${!name:-} ]] || { echo "RWWPO2_BABILONG_CELL_NO_GO:missing $name" >&2; exit 60; }
done
[[ $RWWPO_BABILONG_CELL =~ ^(B|D|E)$ \
   && $RWWPO_BABILONG_STEP =~ ^(20|50|400)$ \
   && $RWWPO_BABILONG_EXPERIMENT_SEED =~ ^(202[6-9]|203[0-3])$ \
   && $RWWPO_BABILONG_PARTITION =~ ^(development|confirmation)$ ]] || {
  echo 'RWWPO2_BABILONG_CELL_NO_GO:cell/step/seed/partition' >&2; exit 61;
}
if [[ $RWWPO_BABILONG_STEP == 400 ]]; then
  [[ $RWWPO_BABILONG_PARTITION == confirmation ]] || {
    echo 'RWWPO2_BABILONG_CELL_NO_GO:R400 must use confirmation' >&2; exit 62;
  }
else
  [[ $RWWPO_BABILONG_PARTITION == development ]] || {
    echo 'RWWPO2_BABILONG_CELL_NO_GO:R20/R50 must use development' >&2; exit 62;
  }
fi
[[ $RWWPO_BABILONG_PIPELINE_ROOT == /* && ! -e $RWWPO_BABILONG_PIPELINE_ROOT ]] || {
  echo 'RWWPO2_BABILONG_CELL_NO_GO:one-use absolute pipeline root' >&2; exit 63;
}
mkdir -p "$RWWPO_BABILONG_PIPELINE_ROOT/manifests" \
  "$RWWPO_BABILONG_PIPELINE_ROOT/evaluations" \
  "$RWWPO_BABILONG_PIPELINE_ROOT/certificates"
if [[ $RWWPO_BABILONG_PARTITION == development ]]; then SUFFIX=DEV; else SUFFIX=CONFIRM; fi
CELL=$RWWPO_BABILONG_CELL
STEP=$RWWPO_BABILONG_STEP
ATTEMPT_AUDIT_ARGS=()
if [[ $STEP != 20 ]]; then
  for name in RWWPO_BABILONG_TRAINING_ATTEMPT_AUDIT \
    RWWPO_BABILONG_TRAINING_ATTEMPT_AUDIT_SHA256; do
    [[ -n ${!name:-} ]] || {
      echo "RWWPO2_BABILONG_CELL_NO_GO:missing $name" >&2; exit 64;
    }
  done
  [[ $RWWPO_BABILONG_TRAINING_ATTEMPT_AUDIT_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
    echo 'RWWPO2_BABILONG_CELL_NO_GO:attempt audit SHA' >&2; exit 64;
  }
  ATTEMPT_AUDIT_ARGS=(
    --training-attempt-audit "$RWWPO_BABILONG_TRAINING_ATTEMPT_AUDIT"
    --training-attempt-audit-sha256 "$RWWPO_BABILONG_TRAINING_ATTEMPT_AUDIT_SHA256"
  )
fi
for LENGTH in 32k 128k; do
  UPPER=${LENGTH^^}
  BASE_RESOLVED=$RWWPO_BABILONG_BUNDLE_ROOT/babilong_${LENGTH}_${RWWPO_BABILONG_PARTITION}_resolved.json
  VALIDATION=$RWWPO_BABILONG_BUNDLE_ROOT/babilong_${LENGTH}_${RWWPO_BABILONG_PARTITION}.parquet
  BASE_SHA=$(sha256sum "$BASE_RESOLVED" | awk '{print $1}')
  INTERFACE=RWWPO2_${CELL}_seed${RWWPO_BABILONG_EXPERIMENT_SEED}_R${STEP}_BABILONG_${UPPER}_${SUFFIX}
  ATTEMPT=rwwpo2-babilong-eval-${CELL,,}-seed${RWWPO_BABILONG_EXPERIMENT_SEED}-r${STEP}-${LENGTH}-${RWWPO_BABILONG_PARTITION}01
  EVAL_ROOT=$RWWPO_BABILONG_PIPELINE_ROOT/evaluations/${CELL}_${LENGTH}_r${STEP}
  RESOLVED=$RWWPO_BABILONG_PIPELINE_ROOT/manifests/${CELL}_${LENGTH}_r${STEP}_resolved.json
  "$RWWPO_WORK_ROOT/.venv/bin/python" \
    "$RWWPO_REPO_DIR/tools/h20/materialize_rwwpo2_babilong_eval.py" \
    --base-resolved "$BASE_RESOLVED" --base-resolved-sha256 "$BASE_SHA" \
    --bundle-audit "$RWWPO_BABILONG_BUNDLE_AUDIT" \
    --bundle-audit-sha256 "$RWWPO_BABILONG_BUNDLE_AUDIT_SHA256" \
    --checkpoint "$RWWPO_BABILONG_CHECKPOINT" --validation "$VALIDATION" \
    --model "$RWWPO_BABILONG_MODEL" --training-commit "$RWWPO_TRAINING_COMMIT" \
    --cell "$CELL" --experiment-seed "$RWWPO_BABILONG_EXPERIMENT_SEED" --step "$STEP" \
    --eval-root "$EVAL_ROOT" --interface-id "$INTERFACE" \
    --attempt-id "$ATTEMPT" --expected-commit "$RWWPO_EXPECTED_COMMIT" \
    "${ATTEMPT_AUDIT_ARGS[@]}" \
    --output "$RESOLVED"
  RESOLVED_SHA=$(sha256sum "$RESOLVED" | awk '{print $1}')
  RWWPO_BABILONG_DATA=$VALIDATION \
  RWWPO_BABILONG_RESOLVED=$RESOLVED \
  RWWPO_BABILONG_RESOLVED_SHA256=$RESOLVED_SHA \
  RWWPO_BABILONG_EVAL_ROOT=$EVAL_ROOT \
  RWWPO_BABILONG_INTERFACE_ID=$INTERFACE \
  RWWPO_BABILONG_ATTEMPT_ID=$ATTEMPT \
    bash "$RWWPO_REPO_DIR/scripts/h20/run_rwwpo2_babilong_eval.sh"
  "$RWWPO_WORK_ROOT/.venv/bin/python" \
    "$RWWPO_REPO_DIR/tools/h20/audit_rwwpo2_babilong_eval.py" \
    --eval-root "$EVAL_ROOT" --resolved-manifest "$RESOLVED" \
    --resolved-manifest-sha256 "$RESOLVED_SHA" --validation "$VALIDATION" \
    --checkpoint "$RWWPO_BABILONG_CHECKPOINT" --model "$RWWPO_BABILONG_MODEL" \
    --interface-id "$INTERFACE" --attempt-id "$ATTEMPT" \
    --expected-commit "$RWWPO_EXPECTED_COMMIT" \
    --metric-rows-output "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/${CELL}_${LENGTH}_r${STEP}.rows.jsonl" \
    --output "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/${CELL}_${LENGTH}_r${STEP}.json"
done
touch "$RWWPO_BABILONG_PIPELINE_ROOT/PIPELINE_PASS"
echo "RWWPO2 BABILONG $CELL R$STEP $RWWPO_BABILONG_PARTITION PIPELINE PASS"

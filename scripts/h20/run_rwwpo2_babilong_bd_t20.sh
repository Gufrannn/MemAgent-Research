#!/usr/bin/env bash
set -euo pipefail

for name in RWWPO_WORK_ROOT RWWPO_REPO_DIR RWWPO_EXPECTED_COMMIT \
  RWWPO_TRAINING_COMMIT GPU_PAIR RWWPO_BABILONG_BUNDLE_ROOT \
  RWWPO_BABILONG_BUNDLE_AUDIT RWWPO_BABILONG_BUNDLE_AUDIT_SHA256 \
  RWWPO_BABILONG_MODEL RWWPO_BABILONG_B_CHECKPOINT \
  RWWPO_BABILONG_D_CHECKPOINT RWWPO_BABILONG_PIPELINE_ROOT; do
  [[ -n ${!name:-} ]] || { echo "RWWPO2_BABILONG_BD_NO_GO:missing $name" >&2; exit 60; }
done
[[ $RWWPO_BABILONG_PIPELINE_ROOT == /* && ! -e $RWWPO_BABILONG_PIPELINE_ROOT ]] || {
  echo 'RWWPO2_BABILONG_BD_NO_GO:one-use absolute pipeline root' >&2; exit 61;
}
mkdir -p "$RWWPO_BABILONG_PIPELINE_ROOT/manifests" \
  "$RWWPO_BABILONG_PIPELINE_ROOT/evaluations" \
  "$RWWPO_BABILONG_PIPELINE_ROOT/certificates"

for CELL in B D; do
  if [[ $CELL == B ]]; then CHECKPOINT=$RWWPO_BABILONG_B_CHECKPOINT; else CHECKPOINT=$RWWPO_BABILONG_D_CHECKPOINT; fi
  for LENGTH in 32k 128k; do
    UPPER=${LENGTH^^}
    BASE_RESOLVED=$RWWPO_BABILONG_BUNDLE_ROOT/babilong_${LENGTH}_development_resolved.json
    VALIDATION=$RWWPO_BABILONG_BUNDLE_ROOT/babilong_${LENGTH}_development.parquet
    BASE_SHA=$(sha256sum "$BASE_RESOLVED" | awk '{print $1}')
    INTERFACE=RWWPO2_${CELL}_seed2026_R20_BABILONG_${UPPER}_DEV
    ATTEMPT=rwwpo2-babilong-eval-${CELL,,}-r20-${LENGTH}-dev01
    EVAL_ROOT=$RWWPO_BABILONG_PIPELINE_ROOT/evaluations/${CELL}_${LENGTH}_r20
    RESOLVED=$RWWPO_BABILONG_PIPELINE_ROOT/manifests/${CELL}_${LENGTH}_r20_resolved.json
    "$RWWPO_WORK_ROOT/.venv/bin/python" \
      "$RWWPO_REPO_DIR/tools/h20/materialize_rwwpo2_babilong_eval.py" \
      --base-resolved "$BASE_RESOLVED" --base-resolved-sha256 "$BASE_SHA" \
      --bundle-audit "$RWWPO_BABILONG_BUNDLE_AUDIT" \
      --bundle-audit-sha256 "$RWWPO_BABILONG_BUNDLE_AUDIT_SHA256" \
      --checkpoint "$CHECKPOINT" --validation "$VALIDATION" \
      --model "$RWWPO_BABILONG_MODEL" \
      --training-commit "$RWWPO_TRAINING_COMMIT" \
      --cell "$CELL" --experiment-seed 2026 --step 20 \
      --eval-root "$EVAL_ROOT" --interface-id "$INTERFACE" \
      --attempt-id "$ATTEMPT" --expected-commit "$RWWPO_EXPECTED_COMMIT" \
      --output "$RESOLVED"
    RESOLVED_SHA=$(sha256sum "$RESOLVED" | awk '{print $1}')
    RWWPO_BABILONG_CHECKPOINT=$CHECKPOINT \
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
      --checkpoint "$CHECKPOINT" --model "$RWWPO_BABILONG_MODEL" \
      --interface-id "$INTERFACE" --attempt-id "$ATTEMPT" \
      --expected-commit "$RWWPO_EXPECTED_COMMIT" \
      --metric-rows-output "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/${CELL}_${LENGTH}_r20.rows.jsonl" \
      --output "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/${CELL}_${LENGTH}_r20.json"
  done
done

"$RWWPO_WORK_ROOT/.venv/bin/python" \
  "$RWWPO_REPO_DIR/tools/h20/compare_rwwpo2_babilong.py" \
  --left-name B-R20 --right-name D-R20 \
  --left-rows \
    "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/B_32k_r20.rows.jsonl" \
    "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/B_128k_r20.rows.jsonl" \
  --right-rows \
    "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/D_32k_r20.rows.jsonl" \
    "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/D_128k_r20.rows.jsonl" \
  --left-reports \
    "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/B_32k_r20.json" \
    "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/B_128k_r20.json" \
  --right-reports \
    "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/D_32k_r20.json" \
    "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/D_128k_r20.json" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_BABILONG_PIPELINE_ROOT/certificates/B_minus_D_r20.json"
touch "$RWWPO_BABILONG_PIPELINE_ROOT/PIPELINE_PASS"
echo "RWWPO2 BABILONG B/D R20 DEVELOPMENT PIPELINE PASS"

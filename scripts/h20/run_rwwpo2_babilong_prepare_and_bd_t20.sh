#!/usr/bin/env bash
set -euo pipefail

for name in RWWPO_WORK_ROOT RWWPO_REPO_DIR RWWPO_EXPECTED_COMMIT \
  RWWPO_TRAINING_COMMIT GPU_PAIR RWWPO_BABILONG_RUN_ID \
  RWWPO_BABILONG_MODEL RWWPO_BABILONG_B_CHECKPOINT \
  RWWPO_BABILONG_D_CHECKPOINT; do
  [[ -n ${!name:-} ]] || {
    echo "RWWPO2_BABILONG_PREPARE_NO_GO:missing $name" >&2; exit 60;
  }
done
[[ $RWWPO_WORK_ROOT == /* && $RWWPO_REPO_DIR == /* \
   && $RWWPO_BABILONG_MODEL == /* \
   && $RWWPO_BABILONG_B_CHECKPOINT == /* \
   && $RWWPO_BABILONG_D_CHECKPOINT == /* \
   && $RWWPO_EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ \
   && $RWWPO_TRAINING_COMMIT =~ ^[0-9a-f]{40}$ \
   && $GPU_PAIR =~ ^[0-9]+,[0-9]+$ \
   && $RWWPO_BABILONG_RUN_ID =~ ^rwwpo2_babilong_[a-z0-9_-]{8,80}$ ]] || {
  echo 'RWWPO2_BABILONG_PREPARE_NO_GO:path/identity syntax' >&2; exit 61;
}
cd "$RWWPO_REPO_DIR"
[[ $(git rev-parse HEAD) == "$RWWPO_EXPECTED_COMMIT" \
   && $(git branch --show-current) == \
      h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822 \
   && -z $(git status --porcelain) ]] || {
  echo 'RWWPO2_BABILONG_PREPARE_NO_GO:checkout' >&2; exit 62;
}

RWWPO_PYTHON="$RWWPO_WORK_ROOT/.venv/bin/python"
export RWWPO_MANIFEST="$RWWPO_REPO_DIR/manifests/h20/qwen25_7b_rwwpo2_r400_k2_seed2026.json"
export RWWPO_MANIFEST_SHA256
RWWPO_MANIFEST_SHA256=$(sha256sum "$RWWPO_MANIFEST" | awk '{print $1}')
BABILONG_MANIFEST="$RWWPO_REPO_DIR/manifests/h20/rwwpo2_babilong_pilot_v1.json"
BABILONG_MANIFEST_SHA256=$(sha256sum "$BABILONG_MANIFEST" | awk '{print $1}')
ROOT="$RWWPO_WORK_ROOT/logs/rwwpo2_babilong/$RWWPO_BABILONG_RUN_ID"
RELEASE_ROOT="$RWWPO_WORK_ROOT/logs/rwwpo2_release_tests/${RWWPO_BABILONG_RUN_ID}_release"
SOURCE_ROOT="$ROOT/source_bundle"
DEVELOPMENT_ROOT="$ROOT/development_bundle"
CERT_ROOT="$ROOT/certificates"
TRAIN_DATA="$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet"

for path in "$RWWPO_PYTHON" "$RWWPO_MANIFEST" "$BABILONG_MANIFEST" \
  "$RWWPO_BABILONG_MODEL/config.json" \
  "$TRAIN_DATA" \
  "$RWWPO_BABILONG_B_CHECKPOINT/data.pt" \
  "$RWWPO_BABILONG_B_CHECKPOINT/actor/model_world_size_2_rank_0.pt" \
  "$RWWPO_BABILONG_B_CHECKPOINT/actor/model_world_size_2_rank_1.pt" \
  "$RWWPO_BABILONG_D_CHECKPOINT/data.pt" \
  "$RWWPO_BABILONG_D_CHECKPOINT/actor/model_world_size_2_rank_0.pt" \
  "$RWWPO_BABILONG_D_CHECKPOINT/actor/model_world_size_2_rank_1.pt"; do
  [[ -f $path && ! -L $path ]] || {
    echo "RWWPO2_BABILONG_PREPARE_NO_GO:missing/symlink $path" >&2; exit 63;
  }
done
[[ ! -e $ROOT && ! -e $RELEASE_ROOT ]] || {
  echo 'RWWPO2_BABILONG_PREPARE_NO_GO:one-use root consumed' >&2; exit 64;
}
mkdir -p "$CERT_ROOT"

echo '=== 1/6 SOURCE FIREWALL ==='
"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_source_firewall.py \
  --manifest "$RWWPO_MANIFEST" --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$CERT_ROOT/source_firewall.json"

echo '=== 2/6 AUTHENTICATED RELEASE TESTS ==='
"$RWWPO_PYTHON" tools/h20/run_rwwpo2_release_tests.py \
  --manifest "$RWWPO_MANIFEST" --manifest-sha256 "$RWWPO_MANIFEST_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" --work-root "$RWWPO_WORK_ROOT" \
  --output-root "$RELEASE_ROOT"
export RWWPO_RELEASE_TEST_RECEIPT="$RELEASE_ROOT/release_tests.json"
export RWWPO_RELEASE_TEST_RECEIPT_SHA256
RWWPO_RELEASE_TEST_RECEIPT_SHA256=$(sha256sum \
  "$RWWPO_RELEASE_TEST_RECEIPT" | awk '{print $1}')
"$RWWPO_PYTHON" tools/h20/verify_rwwpo2_release_tests.py \
  --receipt "$RWWPO_RELEASE_TEST_RECEIPT" \
  --receipt-sha256 "$RWWPO_RELEASE_TEST_RECEIPT_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" --manifest "$RWWPO_MANIFEST" \
  --manifest-sha256 "$RWWPO_MANIFEST_SHA256" --work-root "$RWWPO_WORK_ROOT"

echo '=== 3/6 SIX-CELL MANUAL FIXTURE AUDIT ==='
"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_babilong_fixtures.py \
  --manifest "$BABILONG_MANIFEST" \
  --manifest-sha256 "$BABILONG_MANIFEST_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$CERT_ROOT/fixture_audit.json"

echo '=== 4/6 PINNED OFFICIAL SOURCE + DEVELOPMENT MATERIALIZATION ==='
"$RWWPO_PYTHON" tools/h20/fetch_rwwpo2_babilong_source.py \
  --manifest "$BABILONG_MANIFEST" \
  --manifest-sha256 "$BABILONG_MANIFEST_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" --output-root "$SOURCE_ROOT"
SOURCE_MANIFEST_SHA256=$(sha256sum "$SOURCE_ROOT/bundle_manifest.json" | awk '{print $1}')
"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_babilong_data_boundary.py \
  --manifest "$RWWPO_MANIFEST" --manifest-sha256 "$RWWPO_MANIFEST_SHA256" \
  --adapter-manifest-sha256 "$BABILONG_MANIFEST_SHA256" \
  --train "$TRAIN_DATA" --tokenizer-root "$RWWPO_BABILONG_MODEL" \
  --source-root "$SOURCE_ROOT" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$CERT_ROOT/data_boundary.json"
"$RWWPO_PYTHON" tools/h20/materialize_rwwpo2_babilong.py \
  --manifest "$BABILONG_MANIFEST" \
  --manifest-sha256 "$BABILONG_MANIFEST_SHA256" \
  --source-root "$SOURCE_ROOT" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --tokenizer-root "$RWWPO_BABILONG_MODEL" --partition development \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" --output-root "$DEVELOPMENT_ROOT"

echo '=== 5/6 INDEPENDENT DEVELOPMENT BUNDLE AUDIT ==='
MATERIALIZATION_REPORT_SHA256=$(sha256sum \
  "$DEVELOPMENT_ROOT/materialization_report.json" | awk '{print $1}')
"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_babilong_bundle.py \
  --manifest "$BABILONG_MANIFEST" \
  --manifest-sha256 "$BABILONG_MANIFEST_SHA256" \
  --source-root "$SOURCE_ROOT" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --tokenizer-root "$RWWPO_BABILONG_MODEL" \
  --bundle-root "$DEVELOPMENT_ROOT" \
  --materialization-report-sha256 "$MATERIALIZATION_REPORT_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$CERT_ROOT/development_bundle_audit.json"

echo '=== 6/6 B-R20 THEN D-R20 ON FROZEN DEVELOPMENT ROWS ==='
export RWWPO_BABILONG_BUNDLE_ROOT="$DEVELOPMENT_ROOT"
export RWWPO_BABILONG_BUNDLE_AUDIT="$CERT_ROOT/development_bundle_audit.json"
export RWWPO_BABILONG_BUNDLE_AUDIT_SHA256
RWWPO_BABILONG_BUNDLE_AUDIT_SHA256=$(sha256sum \
  "$RWWPO_BABILONG_BUNDLE_AUDIT" | awk '{print $1}')
export RWWPO_BABILONG_PIPELINE_ROOT="$ROOT/bd_r20"
bash scripts/h20/run_rwwpo2_babilong_bd_t20.sh

touch "$ROOT/PIPELINE_PASS"
echo "RWWPO2 BABILONG PREPARE + B/D R20 PASS: $ROOT"

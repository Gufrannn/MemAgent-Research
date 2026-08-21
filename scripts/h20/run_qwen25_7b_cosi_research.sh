#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/cosi_common.sh"
cosi_checkout_guard
readonly PYTHON=$MEMAGENT_COSI_WORK_ROOT/.venv/bin/python
readonly CERT=$MEMAGENT_COSI_WORK_ROOT/logs/cosi_preflight/certificates
mkdir -p "$CERT"
[[ ! -e $CERT/coral_e0.json ]] || { echo COSI_NO_GO:e0_output_exists >&2; exit 71; }
"$PYTHON" "$MEMAGENT_COSI_REPO_DIR/tools/h20/preflight_qwen25_7b_cosi.py" --manifest "$MEMAGENT_COSI_REPO_DIR/manifests/h20/qwen25_7b_cosi_seed2026.json" --stage research --write-certificate
"$PYTHON" "$MEMAGENT_COSI_REPO_DIR/tools/h20/coral_e0.py" --output "$CERT/coral_e0.json"
"$PYTHON" "$MEMAGENT_COSI_REPO_DIR/tools/h20/import_cosi_original_baseline.py" \
  --bundle-index "$MEMAGENT_COSI_BASELINE_INDEX" \
  --expected-bundle-index-sha256 "$MEMAGENT_COSI_BASELINE_INDEX_SHA256" \
  --expected-eval-manifest-sha256 351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a \
  --s128-resolved-manifest "$MEMAGENT_COSI_S128_RESOLVED_MANIFEST" \
  --expected-s128-resolved-manifest-sha256 "$MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256" \
  --validation "$MEMAGENT_COSI_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  --expected-validation-sha256 54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6 \
  --output "$CERT/baseline_import.json"
echo CORAL_CPU_RESEARCH_GATES_COMPLETE
echo "NEXT_GPU_GATE:$MEMAGENT_COSI_REPO_DIR/scripts/h20/run_qwen25_7b_coral_e1_producer.sh"

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
"$PYTHON" "$MEMAGENT_COSI_REPO_DIR/tools/h20/materialize_cosi_original_baseline.py" \
  --manifest "$MEMAGENT_COSI_REPO_DIR/manifests/h20/qwen25_7b_cosi_seed2026.json" \
  --output "$CERT/baseline_import.json"
echo CORAL_CPU_RESEARCH_GATES_COMPLETE
echo "NEXT_GPU_GATE:$MEMAGENT_COSI_REPO_DIR/scripts/h20/run_qwen25_7b_coral_e1_producer.sh"

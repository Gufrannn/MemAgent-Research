#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/mic_common.sh"
mic_require_checkout
mkdir -p "$MIC_CERT"
"$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/audit_mic_source_firewall.py"
"$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" p0 \
  --manifest "$MIC_MANIFEST" --output "$MIC_P0" --check-runtime
"$MIC_PYTHON" "$MEMAGENT_MIC_REPO_DIR/tools/h20/mic_pipeline.py" e0 --output "$MIC_E0"

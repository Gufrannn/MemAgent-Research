#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/cosi_common.sh"
cosi_checkout_guard
readonly PYTHON=$MEMAGENT_COSI_WORK_ROOT/.venv/bin/python
readonly RUN_ID=${MEMAGENT_COSI_RUN_ID:-coral_seed2026_primary_v1}
readonly EXP=qwen25_7b_coral_fresh_t25_seed2026_${RUN_ID}
readonly RUN_ROOT=$MEMAGENT_COSI_WORK_ROOT/logs/coral/$RUN_ID
readonly TRAIN_ROOT=$MEMAGENT_COSI_WORK_ROOT/logs/memory_agent/$EXP
"$PYTHON" "$MEMAGENT_COSI_REPO_DIR/tools/h20/audit_qwen25_7b_cosi.py" \
  --run-root "$RUN_ROOT" --training-root "$TRAIN_ROOT" --stage final \
  --output "$RUN_ROOT/certificates/final_audit.json"

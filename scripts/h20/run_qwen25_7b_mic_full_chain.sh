#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$SCRIPT_DIR/preflight_qwen25_7b_mic.sh"
bash "$SCRIPT_DIR/run_qwen25_7b_mic_t5.sh"
bash "$SCRIPT_DIR/eval_audit_qwen25_7b_mic.sh" 5
for pair in '5 10' '10 15' '15 20' '20 25'; do
  read -r source target <<<"$pair"
  bash "$SCRIPT_DIR/continue_qwen25_7b_mic.sh" "$source" "$target"
  bash "$SCRIPT_DIR/eval_audit_qwen25_7b_mic.sh" "$target"
done

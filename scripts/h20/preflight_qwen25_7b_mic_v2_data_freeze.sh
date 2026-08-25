#!/usr/bin/env bash
set -euo pipefail

for name in \
  MEMAGENT_MIC_V2_WORK_ROOT \
  MEMAGENT_MIC_V2_REPO_DIR \
  MEMAGENT_MIC_V2_EXPECTED_COMMIT \
  MEMAGENT_MIC_V2_DATA_RUN_ID
do
  if [[ -z "${!name:-}" ]]; then
    echo "MIC_V2_NO_GO: missing required environment $name" >&2
    exit 40
  fi
done

if [[ "$MEMAGENT_MIC_V2_REPO_DIR" != /* || "$MEMAGENT_MIC_V2_WORK_ROOT" != /* ]]; then
  echo "MIC_V2_NO_GO: work/repository roots must be absolute" >&2
  exit 41
fi
if [[ ! "$MEMAGENT_MIC_V2_DATA_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "MIC_V2_NO_GO: data run ID is not a safe stable identifier" >&2
  exit 45
fi
if [[ -n "${PYTHONOPTIMIZE:-}" ]]; then
  echo "MIC_V2_NO_GO: PYTHONOPTIMIZE is forbidden" >&2
  exit 46
fi

MIC_V2_PYTHON="$MEMAGENT_MIC_V2_WORK_ROOT/.venv/bin/python"
MIC_V2_OUTPUT_ROOT="$MEMAGENT_MIC_V2_WORK_ROOT/logs/mic_v2_data_freeze/$MEMAGENT_MIC_V2_DATA_RUN_ID"

# Identity materialization and overlap audit are intentionally CPU-only.
export CUDA_VISIBLE_DEVICES=""

test -x "$MIC_V2_PYTHON"
test "$(git -C "$MEMAGENT_MIC_V2_REPO_DIR" rev-parse HEAD)" = \
  "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" || {
    echo "MIC_V2_NO_GO: exact Git commit mismatch" >&2
    exit 42
  }
test -z "$(git -C "$MEMAGENT_MIC_V2_REPO_DIR" status --porcelain)" || {
  echo "MIC_V2_NO_GO: worktree is dirty" >&2
  exit 43
}

if [[ -e "$MIC_V2_OUTPUT_ROOT" ]]; then
  echo "MIC_V2_NO_GO: data-freeze attempt root already exists: $MIC_V2_OUTPUT_ROOT" >&2
  exit 44
fi
mkdir -p "$MEMAGENT_MIC_V2_WORK_ROOT/logs/mic_v2_data_freeze"
if ! mkdir "$MIC_V2_OUTPUT_ROOT"; then
  echo "MIC_V2_NO_GO: concurrent data-freeze allocation conflict" >&2
  exit 47
fi
mkdir "$MIC_V2_OUTPUT_ROOT/certificates"

cd "$MEMAGENT_MIC_V2_REPO_DIR"
PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" -m py_compile \
  recurrent/research/mic_v2.py \
  tools/h20/mic_v2_data_freeze.py

PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" \
  tools/h20/mic_v2_data_freeze.py materialize \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" \
  --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --output-root "$MIC_V2_OUTPUT_ROOT" \
  --run-id "$MEMAGENT_MIC_V2_DATA_RUN_ID"

PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" \
  tools/h20/mic_v2_data_freeze.py verify \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" \
  --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --output-root "$MIC_V2_OUTPUT_ROOT" \
  --run-id "$MEMAGENT_MIC_V2_DATA_RUN_ID"

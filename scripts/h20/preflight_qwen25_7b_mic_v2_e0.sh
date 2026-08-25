#!/usr/bin/env bash
set -euo pipefail

for name in \
  MEMAGENT_MIC_V2_WORK_ROOT \
  MEMAGENT_MIC_V2_REPO_DIR \
  MEMAGENT_MIC_V2_EXPECTED_COMMIT \
  MEMAGENT_MIC_V2_RUN_ID
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
if [[ ! "$MEMAGENT_MIC_V2_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "MIC_V2_NO_GO: run ID is not a safe stable identifier" >&2
  exit 45
fi

MIC_V2_PYTHON="$MEMAGENT_MIC_V2_WORK_ROOT/.venv/bin/python"
MIC_V2_OUTPUT_ROOT="$MEMAGENT_MIC_V2_WORK_ROOT/logs/mic_v2/$MEMAGENT_MIC_V2_RUN_ID"
MIC_V2_CERTIFICATE="$MIC_V2_OUTPUT_ROOT/certificates/e0.json"

if [[ -n "${PYTHONOPTIMIZE:-}" ]]; then
  echo "MIC_V2_NO_GO: PYTHONOPTIMIZE is forbidden for oracle execution" >&2
  exit 46
fi

# E0 is intentionally CPU-only; it must not consume or even select an H20.
export CUDA_VISIBLE_DEVICES=""

test -x "$MIC_V2_PYTHON"
test -d "$MEMAGENT_MIC_V2_REPO_DIR/.git" || \
  git -C "$MEMAGENT_MIC_V2_REPO_DIR" rev-parse --git-dir >/dev/null
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
  echo "MIC_V2_NO_GO: E0 attempt root already exists: $MIC_V2_OUTPUT_ROOT" >&2
  exit 44
fi
mkdir -p "$MEMAGENT_MIC_V2_WORK_ROOT/logs/mic_v2"
if ! mkdir "$MIC_V2_OUTPUT_ROOT"; then
  echo "MIC_V2_NO_GO: concurrent E0 attempt allocation conflict" >&2
  exit 47
fi
mkdir "$MIC_V2_OUTPUT_ROOT/certificates"

cd "$MEMAGENT_MIC_V2_REPO_DIR"
PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" -m py_compile \
  recurrent/research/mic_v2.py \
  tools/h20/mic_v2_pipeline.py

PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" \
  tools/h20/mic_v2_pipeline.py e0 \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" \
  --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --output "$MIC_V2_CERTIFICATE" \
  --run-id "$MEMAGENT_MIC_V2_RUN_ID"

PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" \
  tools/h20/mic_v2_pipeline.py verify-e0 \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" \
  --expected-commit "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" \
  --output "$MIC_V2_CERTIFICATE" \
  --run-id "$MEMAGENT_MIC_V2_RUN_ID"

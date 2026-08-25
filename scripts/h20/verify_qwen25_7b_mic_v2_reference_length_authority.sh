#!/usr/bin/env bash
set -euo pipefail

for name in \
  MEMAGENT_MIC_V2_WORK_ROOT \
  MEMAGENT_MIC_V2_REPO_DIR \
  MEMAGENT_MIC_V2_EXPECTED_COMMIT
do
  if [[ -z "${!name:-}" ]]; then
    echo "MIC_V2_NO_GO: missing required environment $name" >&2
    exit 40
  fi
done

if [[ "$MEMAGENT_MIC_V2_WORK_ROOT" != /* || "$MEMAGENT_MIC_V2_REPO_DIR" != /* ]]; then
  echo "MIC_V2_NO_GO: work/repository roots must be absolute" >&2
  exit 41
fi
if [[ -n "${PYTHONOPTIMIZE:-}" ]]; then
  echo "MIC_V2_NO_GO: PYTHONOPTIMIZE is forbidden" >&2
  exit 42
fi

readonly MIC_V2_PYTHON="$MEMAGENT_MIC_V2_WORK_ROOT/.venv/bin/python"
test -x "$MIC_V2_PYTHON"
test "$(git -C "$MEMAGENT_MIC_V2_REPO_DIR" rev-parse HEAD)" = \
  "$MEMAGENT_MIC_V2_EXPECTED_COMMIT" || {
    echo "MIC_V2_NO_GO: exact Git commit mismatch" >&2
    exit 43
  }
test -z "$(git -C "$MEMAGENT_MIC_V2_REPO_DIR" status --porcelain)" || {
  echo "MIC_V2_NO_GO: worktree is dirty" >&2
  exit 44
}

cd "$MEMAGENT_MIC_V2_REPO_DIR"
PYTHONPATH="$MEMAGENT_MIC_V2_REPO_DIR" "$MIC_V2_PYTHON" \
  tools/h20/mic_v2_reference_length_authority.py \
  --repo "$MEMAGENT_MIC_V2_REPO_DIR" \
  --work-root "$MEMAGENT_MIC_V2_WORK_ROOT"

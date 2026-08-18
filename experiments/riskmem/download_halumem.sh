#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/data/cw/memagent_work}
DATA_DIR=${HALUMEM_DIR:-$WORK_ROOT/datasets/HaluMem}
mkdir -p "$DATA_DIR"

BASE=https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main
curl -L --fail --retry 5 --continue-at - \
  "$BASE/HaluMem-Medium.jsonl?download=true" \
  -o "$DATA_DIR/HaluMem-Medium.jsonl"

python - "$DATA_DIR/HaluMem-Medium.jsonl" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
users = questions = memories = 0
with path.open() as handle:
    for line in handle:
        row = json.loads(line); users += 1
        for session in row.get("sessions", []):
            questions += len(session.get("questions", []))
            memories += len(session.get("memory_points", []))
print({"path": str(path), "bytes": path.stat().st_size, "users": users,
       "questions": questions, "memory_points": memories})
PY

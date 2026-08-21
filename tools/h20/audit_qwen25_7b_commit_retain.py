#!/usr/bin/env python3
"""Read-only recomputation for the four COMMIT/RETAIN capture pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.commit_retain_capture import canonical_json  # noqa: E402
from recurrent.research.serialization_credit_pilots import write_json_exclusive  # noqa: E402
from tools.h20.preflight_qwen25_7b_commit_retain import (  # noqa: E402
    MANIFEST_REL,
    build_final_audit_report,
    load_manifest,
    record_stage,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST_REL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write-final", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    expected = build_final_audit_report(args.manifest)
    final_path = Path(manifest["paths"]["final_report"])
    if args.write_final:
        write_json_exclusive(final_path, expected)
        record_stage(args.manifest, record_type="audit_result", artifact=final_path)
    elif args.verify_existing:
        actual = json.loads(final_path.read_text(encoding="utf-8"))
        if canonical_json(actual) != canonical_json(expected):
            raise ValueError("existing final report differs from read-only recomputation")
    print(json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if expected["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

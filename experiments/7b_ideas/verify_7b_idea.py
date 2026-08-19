#!/usr/bin/env python3
"""Preflight/adjudication plus fresh2/resume3 artifact verification."""

from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recurrent.research.idea_admissibility import PENDING, append_run_ledger, require_arm

SCIENTIFIC_ANCHORS = (2, 25, 50, 100, 200)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--evidence-ledger", type=Path)
    p.add_argument("--diagnostic-only", action="store_true")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--phase", choices=("preflight", "fresh2", "resume3"), default="preflight")
    p.add_argument("--log", type=Path)
    args = p.parse_args()
    try:
        decision = require_arm(args.arm, args.evidence_ledger, diagnostic_only=args.diagnostic_only)
    except ValueError as exc:
        print(str(exc)); return 3
    result = {"status": decision.status, "training_authorized": decision.training_authorized,
              "selected_arm": decision.selected_arm, "ledger_hash": decision.ledger_hash,
              "anchors": SCIENTIFIC_ANCHORS, "auto_step400": False}
    if args.phase != "preflight":
        if not args.run_dir or not args.log:
            raise SystemExit("--run-dir and --log required for artifact verification")
        step = 2 if args.phase == "fresh2" else 3
        text = args.log.read_text(errors="replace")
        required = [args.run_dir / f"global_step_{step}" / "actor", args.run_dir / f"global_step_{step}" / "data.pt"]
        missing = [str(x) for x in required if not x.exists()]
        if "actor/grad_norm:" not in text: missing.append("actor update marker")
        if "After sync model weights in sharding manager" not in text and "vLLM load weights" not in text:
            missing.append("vLLM weight-sync marker")
        if args.phase == "resume3" and not re.search(r"Resuming from.*global_step_2", text, re.S):
            missing.append("explicit source checkpoint global_step_2")
        result["artifact_failures"] = missing
        if missing: print(json.dumps(result, indent=2)); return 4
    print(json.dumps(result, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())

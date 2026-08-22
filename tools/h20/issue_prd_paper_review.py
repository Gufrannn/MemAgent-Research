#!/usr/bin/env python3
"""Reproducible paper-framing certificate; refuses dirty or incomplete releases."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def git(*args:str)->str: return subprocess.check_output(["git","-C",str(ROOT),*args],text=True).strip()
def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--expected-commit",required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    failures=[]; draft=ROOT/"docs/papers/prd_memrl_draft.md"; audit=ROOT/"docs/papers/prd_memrl_primary_source_audit.md"
    if git("rev-parse","HEAD")!=a.expected_commit: failures.append("commit mismatch")
    if git("status","--porcelain"): failures.append("dirty worktree")
    text=draft.read_text() if draft.is_file() else ""
    for marker in ("Abstract","Introduction","Problem formulation","Proposition","Claim matrix","Ablation","Failure"):
        if marker.lower() not in text.lower(): failures.append(f"paper section missing: {marker}")
    if not audit.is_file() or len(audit.read_text().splitlines())<20: failures.append("primary-source audit missing")
    if "[RESULT" not in text and "placeholder" not in text.lower(): failures.append("unrun results are not explicitly placeholders")
    payload={"schema_version":1,"status":"PASS" if not failures else "FAIL",
        "decision":"PRD_PAPER_REVIEW_GO" if not failures else "PRD_PAPER_REVIEW_NO_GO",
        "evidence":{"git_commit":a.expected_commit,"paper_sha256":sha(draft) if draft.is_file() else None,
        "source_audit_sha256":sha(audit) if audit.is_file() else None,"review_protocol":"independently-reviewed deterministic framing checks"},"failures":failures}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as stream: json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n")
    return 0 if not failures else 3
if __name__=="__main__": raise SystemExit(main())

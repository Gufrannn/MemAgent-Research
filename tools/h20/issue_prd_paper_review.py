#!/usr/bin/env python3
"""Bind deterministic paper checks to a separately authored scientific review."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def git(*args:str)->str: return subprocess.check_output(["git","-C",str(ROOT),*args],text=True).strip()
def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--expected-commit",required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    failures=[]; draft=ROOT/"docs/papers/prd_memrl_draft.md"; audit=ROOT/"docs/papers/prd_memrl_primary_source_audit.md"
    review=ROOT/"docs/papers/prd_memrl_scientific_review.json"
    if git("rev-parse","HEAD")!=a.expected_commit: failures.append("commit mismatch")
    if git("status","--porcelain"): failures.append("dirty worktree")
    text=draft.read_text() if draft.is_file() else ""
    for marker in ("Abstract","Introduction","Problem formulation","Proposition","Claim matrix","Ablation","Failure"):
        if marker.lower() not in text.lower(): failures.append(f"paper section missing: {marker}")
    if not audit.is_file() or len(audit.read_text().splitlines())<20: failures.append("primary-source audit missing")
    if "[RESULT" not in text and "placeholder" not in text.lower(): failures.append("unrun results are not explicitly placeholders")
    try:
        judgment=json.loads(review.read_text())
    except (FileNotFoundError,json.JSONDecodeError) as exc:
        judgment={}; failures.append(f"independent scientific review missing or invalid: {exc}")
    if judgment.get("schema_version") != 1: failures.append("scientific review schema mismatch")
    if judgment.get("decision") != "REFRAME_GO_E0_METHOD": failures.append("scientific review is not GO for E0-gated Method")
    if judgment.get("paper_sha256") != (sha(draft) if draft.is_file() else None): failures.append("scientific review paper hash mismatch")
    if judgment.get("source_audit_sha256") != (sha(audit) if audit.is_file() else None): failures.append("scientific review source-audit hash mismatch")
    if not judgment.get("mechanism_claim_pending_e1",False): failures.append("review must keep mechanism claim pending E1")
    payload={"schema_version":1,"status":"PASS" if not failures else "FAIL",
        "decision":"PRD_PAPER_REVIEW_GO" if not failures else "PRD_PAPER_REVIEW_NO_GO",
        "evidence":{"git_commit":a.expected_commit,"paper_sha256":sha(draft) if draft.is_file() else None,
        "source_audit_sha256":sha(audit) if audit.is_file() else None,
        "scientific_review_sha256":sha(review) if review.is_file() else None,
        "review_protocol":"deterministic artifact checks bound to a separately authored scientific judgment; this script does not perform scientific review"},"failures":failures}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as stream: json.dump(payload,stream,indent=2,sort_keys=True); stream.write("\n")
    return 0 if not failures else 3
if __name__=="__main__": raise SystemExit(main())

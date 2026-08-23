#!/usr/bin/env python3
"""Commit-bound E0 for TF-RWWPO gradient and controller closure."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# This file is a user-facing H20 entrypoint, so it must work when invoked as
# ``python tools/h20/run_tf_rwwpo_e0.py`` without relying on an ambient
# PYTHONPATH.  Python otherwise puts tools/h20, rather than the repository
# root, on sys.path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.rwwpo_transaction import ALPHA_GRID, largest_tested_feasible


def main():
    p=argparse.ArgumentParser(); p.add_argument("--expected-commit",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if head!=a.expected_commit or subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise SystemExit("TF_RWWPO_E0_NO_GO:checkout")
    with tempfile.TemporaryDirectory() as raw:
        base=Path(raw)/"base.json"
        subprocess.run([sys.executable,str(Path(__file__).with_name("run_rwwpo_e0.py")),
                        "--expected-commit",head,"--output",str(base)],check=True)
        original=json.loads(base.read_text())
    cases={
        "alpha_one":largest_tested_feasible({1.0:True}).alpha,
        "half":largest_tested_feasible({1.0:False,.5:True}).alpha,
        "smallest":largest_tested_feasible({x:x==1/32 for x in ALPHA_GRID}).alpha,
        "all_reject":largest_tested_feasible({x:False for x in ALPHA_GRID}).alpha,
        "descending_prefix":largest_tested_feasible({1.0:False,.5:False,.25:True}).alpha,
        "zero_proposal":largest_tested_feasible({1.0:True},proposal_zero=True).alpha,
    }
    expected={"alpha_one":1.0,"half":.5,"smallest":1/32,"all_reject":0.0,
              "descending_prefix":.25,"zero_proposal":0.0}
    status="PASS" if original["status"]=="PASS" and cases==expected else "FAIL"
    report={"status":status,"decision":"RWWPO_E0_PASS" if status=="PASS" else "TF_RWWPO_E0_NO_GO",
            "git_commit":head,"base_e0_report_sha256":original["report_sha256"],"controller_cases":cases,
            "frozen_alpha_grid":list(ALPHA_GRID)}
    raw=json.dumps(report,sort_keys=True,separators=(",",":")); report["report_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    raise SystemExit(0 if status=="PASS" else 1)

if __name__=="__main__": main()

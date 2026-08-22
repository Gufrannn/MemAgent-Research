#!/usr/bin/env python3
"""Commit-bound E0 for TF-RWWPO gradient and controller closure."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

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
        "alpha_one":largest_tested_feasible({x:x==1 for x in ALPHA_GRID}).alpha,
        "half":largest_tested_feasible({x:x<=.5 for x in ALPHA_GRID}).alpha,
        "smallest":largest_tested_feasible({x:x==1/32 for x in ALPHA_GRID}).alpha,
        "all_reject":largest_tested_feasible({x:False for x in ALPHA_GRID}).alpha,
        "nonmonotone":largest_tested_feasible({x:x in (.5,.125) for x in ALPHA_GRID}).alpha,
        "zero_proposal":largest_tested_feasible({x:True for x in ALPHA_GRID},proposal_zero=True).alpha,
    }
    expected={"alpha_one":1.0,"half":.5,"smallest":1/32,"all_reject":0.0,"nonmonotone":.5,"zero_proposal":0.0}
    status="PASS" if original["status"]=="PASS" and cases==expected else "FAIL"
    report={"status":status,"decision":"RWWPO_E0_PASS" if status=="PASS" else "TF_RWWPO_E0_NO_GO",
            "git_commit":head,"base_e0_report_sha256":original["report_sha256"],"controller_cases":cases,
            "frozen_alpha_grid":list(ALPHA_GRID)}
    raw=json.dumps(report,sort_keys=True,separators=(",",":")); report["report_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    raise SystemExit(0 if status=="PASS" else 1)

if __name__=="__main__": main()

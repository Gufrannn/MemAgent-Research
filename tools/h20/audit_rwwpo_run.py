#!/usr/bin/env python3
import argparse, hashlib, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.h20.audit_rwwpo_actual_loss import audit

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-root",required=True); p.add_argument("--actual-ledger-dir",required=True); p.add_argument("--execution-ledger",required=True); p.add_argument("--expected-commit",required=True); p.add_argument("--target-step",type=int,required=True); p.add_argument("--output",required=True); a=p.parse_args()
    ck=Path(a.run_root)/f"global_step_{a.target_step}"
    if not (ck.joinpath("actor").is_dir() and ck.joinpath("data.pt").is_file()): raise SystemExit("RWWPO_AUDIT_NO_GO:checkpoint")
    ledgers=sorted(Path(a.actual_ledger_dir).glob("actual_loss_rank*.jsonl"))
    actual=audit(ledgers,require_method=True)
    if actual["modes"] != ["rwwpo_method"]: raise SystemExit("RWWPO_AUDIT_NO_GO:wrong ledger mode")
    events=[json.loads(x) for x in Path(a.execution_ledger).read_text().splitlines() if x.strip()]
    text=json.dumps(events,sort_keys=True)
    if a.expected_commit not in text: raise SystemExit("RWWPO_AUDIT_NO_GO:commit")
    for token in ("post_actor_update","weight"):
        if token not in text: raise SystemExit(f"RWWPO_AUDIT_NO_GO:{token}")
    if actual["min_prefix_ess"] < .5: raise SystemExit("RWWPO_AUDIT_NO_GO:constraint")
    report={"status":"PASS","decision":f"RWWPO_T{a.target_step}_HEALTH_PASS","git_commit":a.expected_commit,"target_step":a.target_step,"checkpoint":str(ck.resolve()),"actual_loss":actual,"execution_ledger_sha256":hashlib.sha256(Path(a.execution_ledger).read_bytes()).hexdigest()}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
if __name__=="__main__": main()

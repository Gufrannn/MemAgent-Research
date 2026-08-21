#!/usr/bin/env python3
import argparse, hashlib, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.h20.audit_rwwpo_actual_loss import audit
from recurrent.research.gate_a_execution import validate_jsonl_chain,checkpoint_inventory

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-root",required=True); p.add_argument("--actual-ledger-dir",required=True); p.add_argument("--execution-ledger",required=True); p.add_argument("--expected-commit",required=True); p.add_argument("--target-step",type=int,required=True); p.add_argument("--output",required=True); a=p.parse_args()
    ck=Path(a.run_root)/f"global_step_{a.target_step}"
    if not (ck.joinpath("actor").is_dir() and ck.joinpath("data.pt").is_file()): raise SystemExit("RWWPO_AUDIT_NO_GO:checkpoint")
    ledgers=sorted(Path(a.actual_ledger_dir).glob("actual_loss_rank*.jsonl"))
    actual=audit(ledgers,require_method=True)
    if actual["modes"] != ["rwwpo_method"]: raise SystemExit("RWWPO_AUDIT_NO_GO:wrong ledger mode")
    events=[json.loads(x) for x in Path(a.execution_ledger).read_text().splitlines() if x.strip()]
    failures=validate_jsonl_chain(events)
    if failures or any(row.get("git_commit")!=a.expected_commit for row in events): raise SystemExit("RWWPO_AUDIT_NO_GO:ledger chain/commit")
    sync=[row for row in events if row.get("record_type")=="weight_sync_summary" and row.get("sync_kind")=="post_actor_update" and int(row.get("global_step",-1))<=a.target_step]
    if not sync or sync[-1].get("worker_ranks")!=[0,1] or not re.fullmatch(r"[0-9a-f]{64}",str(sync[-1].get("sampled_tensor_digest",""))): raise SystemExit("RWWPO_AUDIT_NO_GO:weight sync closure")
    inventories=[row for row in events if row.get("record_type")=="checkpoint_inventory" and row.get("global_step")==a.target_step]
    current_inventory=checkpoint_inventory(ck)
    if not inventories or inventories[-1].get("inventory")!=current_inventory: raise SystemExit("RWWPO_AUDIT_NO_GO:checkpoint inventory")
    if actual["accepted_fraction"] <= 0 or actual["max_proposed_update"] <= 1e-10: raise SystemExit("RWWPO_AUDIT_NO_GO:post-step acceptance/aperture")
    report={"status":"PASS","decision":f"RWWPO_T{a.target_step}_HEALTH_PASS","git_commit":a.expected_commit,"target_step":a.target_step,"checkpoint":str(ck.resolve()),"actual_loss":actual,"execution_ledger_sha256":hashlib.sha256(Path(a.execution_ledger).read_bytes()).hexdigest()}
    report["report_sha256"]=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
if __name__=="__main__": main()

#!/usr/bin/env python3
import argparse, hashlib, json, re, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.h20.audit_rwwpo_actual_loss import audit
from recurrent.research.gate_a_execution import validate_jsonl_chain,checkpoint_inventory

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-root",required=True); p.add_argument("--actual-ledger-dir",required=True); p.add_argument("--execution-ledger",required=True); p.add_argument("--expected-commit",required=True); p.add_argument("--expected-schema-version",required=True); p.add_argument("--expected-objective",required=True); p.add_argument("--expected-controller",required=True); p.add_argument("--target-step",type=int,required=True); p.add_argument("--output",required=True); a=p.parse_args()
    ck=Path(a.run_root)/f"global_step_{a.target_step}"
    if not (ck.joinpath("actor").is_dir() and ck.joinpath("data.pt").is_file()): raise SystemExit("RWWPO_AUDIT_NO_GO:checkpoint")
    ledgers=sorted(Path(a.actual_ledger_dir).glob("actual_loss_rank*.jsonl"))
    actual=audit(ledgers,require_method=True)
    if actual.get("schema_versions")!=[a.expected_schema_version]: raise SystemExit("RWWPO_AUDIT_NO_GO:schema identity")
    if actual.get("objective_variants")!=[a.expected_objective]: raise SystemExit("RWWPO_AUDIT_NO_GO:objective identity")
    if actual.get("controller_variants")!=[a.expected_controller]: raise SystemExit("RWWPO_AUDIT_NO_GO:controller identity")
    actual_rows=[]
    for ledger in ledgers:
        actual_rows.extend(json.loads(line) for line in ledger.read_text().splitlines() if line.strip())
    actual_identities={(row["attempt_id"],int(row["rank"]),int(row["global_step"]),int(row["epoch"]),int(row["minibatch"])):row for row in actual_rows}
    markers=sorted(Path(a.actual_ledger_dir).glob("transaction_rank*.jsonl"))
    if len(markers)!=2: raise SystemExit("RWWPO_AUDIT_NO_GO:transaction marker ranks")
    marker_completions={}
    marker_intents={}
    for marker in markers:
        records=[json.loads(line) for line in marker.read_text().splitlines() if line.strip()]
        previous="0"*64; pending=None
        for record in records:
            declared=record.pop("record_sha256",None)
            if record.get("previous_record_sha256")!=previous or hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":")).encode()).hexdigest()!=declared:
                raise SystemExit("RWWPO_AUDIT_NO_GO:transaction marker chain")
            previous=declared
            identity=tuple(record[key] for key in ("attempt_id","rank","global_step","epoch","minibatch"))
            if record["phase"]=="intent":
                if pending is not None: raise SystemExit("RWWPO_AUDIT_NO_GO:nested transaction intent")
                pending=identity
                if identity in marker_intents: raise SystemExit("RWWPO_AUDIT_NO_GO:duplicate transaction intent")
                marker_intents[identity]=record["model_digest"]
            elif pending!=identity: raise SystemExit("RWWPO_AUDIT_NO_GO:orphan transaction completion")
            else:
                if identity in marker_completions: raise SystemExit("RWWPO_AUDIT_NO_GO:duplicate transaction completion")
                marker_completions[identity]=record["model_digest"]; pending=None
        if pending is not None: raise SystemExit("RWWPO_AUDIT_NO_GO:interrupted trial transaction")
    if set(marker_intents)!=set(actual_identities) or set(marker_completions)!=set(actual_identities):
        raise SystemExit("RWWPO_AUDIT_NO_GO:transaction marker/actual identity bijection")
    for identity,row in actual_identities.items():
        if marker_intents[identity]!=row["pre_digests"]["model"]:
            raise SystemExit("RWWPO_AUDIT_NO_GO:transaction intent model digest mismatch")
        if marker_completions[identity]!=row["commit_digests"]["model"]:
            raise SystemExit("RWWPO_AUDIT_NO_GO:transaction completion model digest mismatch")
    if actual["modes"] != ["rwwpo_method"]: raise SystemExit("RWWPO_AUDIT_NO_GO:wrong ledger mode")
    events=[json.loads(x) for x in Path(a.execution_ledger).read_text().splitlines() if x.strip()]
    failures=validate_jsonl_chain(events)
    if failures or any(row.get("git_commit")!=a.expected_commit for row in events): raise SystemExit("RWWPO_AUDIT_NO_GO:ledger chain/commit")
    sync=[row for row in events if row.get("record_type")=="weight_sync_summary" and row.get("sync_kind")=="post_actor_update" and int(row.get("global_step",-1))<=a.target_step]
    if not sync or sync[-1].get("worker_ranks")!=[0,1] or not re.fullmatch(r"[0-9a-f]{64}",str(sync[-1].get("sampled_tensor_digest",""))): raise SystemExit("RWWPO_AUDIT_NO_GO:weight sync closure")
    inventories=[row for row in events if row.get("record_type")=="checkpoint_inventory" and row.get("global_step")==a.target_step]
    current_inventory=checkpoint_inventory(ck)
    if not inventories or inventories[-1].get("inventory")!=current_inventory: raise SystemExit("RWWPO_AUDIT_NO_GO:checkpoint inventory")
    anchors=inventories[-1].get("rwwpo_ledger_anchors",{})
    expected_anchor_names={f"{kind}_rank{rank}.jsonl" for kind in ("actual_loss","transaction") for rank in (0,1)}
    if set(anchors)!=expected_anchor_names: raise SystemExit("RWWPO_AUDIT_NO_GO:checkpoint ledger anchors")
    for name,anchor in anchors.items():
        path=Path(a.actual_ledger_dir)/name
        lines=[line for line in path.read_bytes().splitlines(keepends=True) if line.strip()]
        count=int(anchor.get("record_count",-1))
        if count<1 or len(lines)<count: raise SystemExit("RWWPO_AUDIT_NO_GO:checkpoint ledger anchor count")
        prefix=b"".join(lines[:count])
        if hashlib.sha256(prefix).hexdigest()!=anchor.get("prefix_sha256"):
            raise SystemExit("RWWPO_AUDIT_NO_GO:checkpoint ledger prefix SHA")
        if json.loads(lines[count-1])["record_sha256"]!=anchor.get("tail_sha256"):
            raise SystemExit("RWWPO_AUDIT_NO_GO:checkpoint ledger tail")
    transactional=a.expected_controller=="feasible_backtracking"
    if transactional and a.target_step==5:
        early=[alpha for step in range(1,6)
               for alpha in actual["steps"].get(str(step),{}).get("alpha_committed",[])
               if float(alpha)>0]
        if len(early)<4: raise SystemExit("RWWPO_AUDIT_NO_GO:T5_NONZERO_COMMIT_COUNT")
        if statistics.median(early)<0.125: raise SystemExit("RWWPO_AUDIT_NO_GO:T5_MEDIAN_ALPHA")
        if sum(alpha<=1/32 for alpha in early)>len(early)/2: raise SystemExit("RWWPO_AUDIT_NO_GO:PSEUDO_ACTIVITY")
    elif not transactional:
        target_actual=actual["steps"].get(str(a.target_step),{})
        if target_actual.get("accepted_fraction",0) <= 0 or target_actual.get("max_proposed_update",0) <= 1e-10: raise SystemExit("RWWPO_AUDIT_NO_GO:target-step acceptance/aperture")
    report={"status":"PASS","decision":f"RWWPO_T{a.target_step}_HEALTH_PASS","git_commit":a.expected_commit,"target_step":a.target_step,"checkpoint":str(ck.resolve()),"actual_loss":actual,"execution_ledger_sha256":hashlib.sha256(Path(a.execution_ledger).read_bytes()).hexdigest()}
    report["report_sha256"]=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
if __name__=="__main__": main()

#!/usr/bin/env python3
import argparse, hashlib, json, re, statistics, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.h20.audit_rwwpo_actual_loss import audit,hydrate_authenticated_v3_receipt
from recurrent.research.gate_a_execution import validate_jsonl_chain,checkpoint_inventory
from recurrent.research.actor_batch import stable_identity_int64
from recurrent.research.rwwpo_ledger import tensor_shard_inventory

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-root",required=True); p.add_argument("--actual-ledger-dir",required=True); p.add_argument("--execution-ledger",required=True); p.add_argument("--expected-commit",required=True); p.add_argument("--expected-schema-version",required=True); p.add_argument("--expected-objective",required=True); p.add_argument("--expected-controller",required=True); p.add_argument("--target-step",type=int,required=True); p.add_argument("--output",required=True); a=p.parse_args()
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain"],text=True).strip()
    if head!=a.expected_commit: raise SystemExit("RWWPO_AUDIT_NO_GO:checkout commit")
    if dirty: raise SystemExit("RWWPO_AUDIT_NO_GO:dirty checkout")
    ck=Path(a.run_root)/f"global_step_{a.target_step}"
    if not (ck.joinpath("actor").is_dir() and ck.joinpath("data.pt").is_file()): raise SystemExit("RWWPO_AUDIT_NO_GO:checkpoint")
    ledgers=sorted(Path(a.actual_ledger_dir).glob("actual_loss_rank*.jsonl"))
    actual=audit(ledgers,require_method=True)
    if actual.get("schema_versions")!=[a.expected_schema_version]: raise SystemExit("RWWPO_AUDIT_NO_GO:schema identity")
    if actual.get("objective_variants")!=[a.expected_objective]: raise SystemExit("RWWPO_AUDIT_NO_GO:objective identity")
    if actual.get("controller_variants")!=[a.expected_controller]: raise SystemExit("RWWPO_AUDIT_NO_GO:controller identity")
    actual_rows=[]
    for ledger in ledgers:
        for line in ledger.read_text().splitlines():
            if not line.strip(): continue
            receipt=json.loads(line)
            actual_rows.append(hydrate_authenticated_v3_receipt(receipt,ledger)
                               if receipt.get("schema_version")=="rwwpo-actual-loss-v3"
                               else receipt)
    seed_path=Path(a.run_root)/"rollout_seed_audit.jsonl"
    if not seed_path.is_file(): raise SystemExit("RWWPO_AUDIT_NO_GO:stable rollout identity ledger")
    seed_rows=[json.loads(line) for line in seed_path.read_text().splitlines() if line.strip()]
    turn_identity={}
    for row in seed_rows:
        if row.get("record_type")!="trajectory_turn_seed": continue
        try:
            key=(int(row["global_step"]),int(row["sample_index"]),int(row["turn"]))
            example_id=str(row["stable_example_id"]); trajectory_id=str(row["trajectory_id"])
            expected_example=f"frozen_train_row:{int(row['dataset_index'])}"
            expected_trajectory=f"{expected_example}:seed:{int(row['trajectory_seed'])}"
        except (KeyError,TypeError,ValueError):
            raise SystemExit("RWWPO_AUDIT_NO_GO:malformed stable rollout identity")
        if example_id!=expected_example or trajectory_id!=expected_trajectory:
            raise SystemExit("RWWPO_AUDIT_NO_GO:non-reconstructible stable rollout identity")
        if key in turn_identity: raise SystemExit("RWWPO_AUDIT_NO_GO:duplicate stable rollout identity")
        turn_identity[key]=(stable_identity_int64(example_id),stable_identity_int64(trajectory_id))
    if not turn_identity: raise SystemExit("RWWPO_AUDIT_NO_GO:empty stable rollout identity ledger")
    for actual_row in actual_rows:
        step=int(actual_row["global_step"])
        columns=zip(actual_row["sample_index"],actual_row["trajectory_turn"],
                    actual_row["example_identity_hash"],actual_row["trajectory_identity_hash"])
        for sample_index,turn,example_hash,trajectory_hash in columns:
            expected=turn_identity.get((step,int(sample_index),int(turn)))
            if expected is None or expected!=(int(example_hash),int(trajectory_hash)):
                raise SystemExit("RWWPO_AUDIT_NO_GO:actual/stable rollout identity mismatch")
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
            if a.expected_schema_version=="rwwpo-actual-loss-v3":
                actual_row=actual_identities.get(identity)
                if actual_row is None or record.get("schema_version")!="rwwpo-transaction-v2":
                    raise SystemExit("RWWPO_AUDIT_NO_GO:RWWPO-2 marker schema/identity")
                if (int(record.get("inner_id",0))!=int(actual_row["inner_id"]) or
                        int(record.get("proposal_clock",0))!=int(actual_row["proposal_clock"])):
                    raise SystemExit("RWWPO_AUDIT_NO_GO:RWWPO-2 marker proposal coordinate")
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
    if a.expected_schema_version=="rwwpo-actual-loss-v3":
        start_round=min(int(row["global_step"]) for row in actual_rows)
        tensor_inventory=tensor_shard_inventory(
            a.actual_ledger_dir,start_round=start_round,through_round=a.target_step)
        if inventories[-1].get("rwwpo_tensor_inventory")!=tensor_inventory:
            raise SystemExit("RWWPO_AUDIT_NO_GO:checkpoint tensor-shard inventory")
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
    report={"status":"PASS","decision":f"RWWPO_T{a.target_step}_HEALTH_PASS","git_commit":a.expected_commit,"target_step":a.target_step,"checkpoint":str(ck.resolve()),"actual_loss":actual,"execution_ledger_sha256":hashlib.sha256(Path(a.execution_ledger).read_bytes()).hexdigest(),"rollout_seed_audit_sha256":hashlib.sha256(seed_path.read_bytes()).hexdigest()}
    report["report_sha256"]=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
if __name__=="__main__": main()

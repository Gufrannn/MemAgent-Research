#!/usr/bin/env python3
"""Dependency-free real-entry smoke for the TF-RWWPO v2 auditor."""
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.h20.audit_rwwpo_actual_loss import audit
from recurrent.research.actor_batch import stable_identity_int64
from recurrent.research.gate_a_execution import append_jsonl, checkpoint_inventory


def signed(row):
    raw=json.dumps(row,sort_keys=True,separators=(",",":"))
    row=dict(row); row["record_sha256"]=hashlib.sha256(raw.encode()).hexdigest(); return row


def make_row(rank):
    value=0.1 if rank==0 else -0.1
    weights=[math.exp(0.1),math.exp(-0.1)]; total=sum(weights); weights=[x/total for x in weights]
    chi2=2*sum(x*x for x in weights)-1
    pre={"turn":0,"batch_size":2,"ess_fraction":1.0,"chi2":0.0,
         "max_abs_log_ratio":0.0,"mean_log_ratio":0.0}
    post={"turn":0,"batch_size":2,"ess_fraction":1/(1+chi2),"chi2":chi2,
          "max_abs_log_ratio":0.1,"mean_log_ratio":0.0}
    prefix={"turn":0,"sample_index":rank,"log_ratio":0.0,"prefix_token_count":1}
    trial_prefix={"turn":0,"sample_index":rank,"log_ratio":value,"prefix_token_count":1}
    dig={"model":"a"*64,"optimizer":"b"*64,"scheduler":"c"*64,
         "scaler":"not_applicable_bfloat16","rng":"d"*64}
    committed=dict(dig); committed["model"]="e"*64
    return signed({
        "schema_version":"rwwpo-actual-loss-v2","attempt_id":"tf_smoke","mode":"rwwpo_method",
        "objective_variant":"whole_prefix","controller_variant":"feasible_backtracking",
        "global_step":1,"rank":rank,"epoch":0,"minibatch":0,
        "old_log_prob":[[0.0]],"current_log_prob":[[0.0]],
        "proposed_post_log_prob":[[value]],"committed_log_prob":[[value]],
        "response_mask":[[1.0]],"writer_mask":[[1.0]],"answer_mask":[[0.0]],
        "trajectory_turn":[0],"sample_index":[rank],
        "example_identity_hash":[stable_identity_int64(f"frozen_train_row:{17+rank}")],
        "trajectory_identity_hash":[stable_identity_int64(f"frozen_train_row:{17+rank}:seed:{101+rank}")],
        "advantages":[[1.0]],"denominator":1,
        "prefix_rows":[prefix],"prefix_stats":[pre],"post_prefix_rows":[trial_prefix],
        "post_prefix_stats":[post],"q_min":0.5,"writer_log_ratio_cap":4.0,
        "constraint_pass":True,"accepted":True,"alpha_grid":[1.0,.5,.25,.125,.0625,.03125],
        "alpha_test_order":[1.0],"alpha_committed":1.0,"accepted_nonzero":True,
        "proposal_zero":False,"trial_evidence":[{"alpha":1.0,"feasible":True,
            "log_prob":[[value]],"prefix_rows":[trial_prefix],"prefix_stats":[post]}],
        "full_parameter_displacement_norm":1.0,"committed_parameter_displacement_norm":1.0,
        "full_writer_logprob_movement":0.1,"committed_writer_logprob_movement":0.1,
        "pre_digests":dig,"commit_digests":committed,"trial_forward_wall_seconds":1.0,
        "gradient_norm":1.0,"mechanism_diagnostics":{},"previous_record_sha256":"0"*64,
    })


def main():
    with tempfile.TemporaryDirectory() as raw:
        root=Path(raw); paths=[]
        for rank in (0,1):
            path=root/f"actual_loss_rank{rank}.jsonl"
            path.write_text(json.dumps(make_row(rank),sort_keys=True,separators=(",",":"))+"\n")
            paths.append(path)
        result=audit(paths,require_method=True)
        assert result["status"]=="PASS" and result["schema_versions"]==["rwwpo-actual-loss-v2"]
        try: audit(paths[:1],require_method=True)
        except ValueError as exc: assert "rank0 and rank1" in str(exc)
        else: raise AssertionError("missing rank was accepted")
        forged=json.loads(paths[0].read_text()); forged["committed_log_prob"]=[[0.05]]
        paths[0].write_text(json.dumps(signed({k:v for k,v in forged.items() if k!="record_sha256"}),sort_keys=True,separators=(",",":"))+"\n")
        try: audit(paths,require_method=True)
        except ValueError as exc: assert "post-step prefix" in str(exc) or "selected trial" in str(exc)
        else: raise AssertionError("forged committed logprob was accepted")
    # Full formal CLI: actual rows, transaction markers, execution chain,
    # checkpoint inventory, ledger anchors, weight sync, and checkout binding.
    with tempfile.TemporaryDirectory() as raw:
        root=Path(raw); ledger=root/"actual"; ledger.mkdir(); run=root/"run"; ck=run/"global_step_1"
        (ck/"actor").mkdir(parents=True); (ck/"actor"/"model_world_size_2_rank_0.pt").write_bytes(b"m0")
        (ck/"actor"/"model_world_size_2_rank_1.pt").write_bytes(b"m1"); (ck/"data.pt").write_bytes(b"data")
        anchors={}
        for rank in (0,1):
            actual_path=ledger/f"actual_loss_rank{rank}.jsonl"
            actual_payload=(json.dumps(make_row(rank),sort_keys=True,separators=(",",":"))+"\n").encode()
            actual_path.write_bytes(actual_payload)
            identity={"schema_version":"rwwpo-transaction-v1","attempt_id":"tf_smoke","rank":rank,
                      "global_step":1,"epoch":0,"minibatch":0}
            marker_rows=[]; previous="0"*64
            for phase,model in (("intent","a"*64),("complete","e"*64)):
                row={**identity,"phase":phase,"model_digest":model,"previous_record_sha256":previous}
                row=signed(row); previous=row["record_sha256"]; marker_rows.append(row)
            transaction_path=ledger/f"transaction_rank{rank}.jsonl"
            transaction_payload=b"".join((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode() for row in marker_rows)
            transaction_path.write_bytes(transaction_payload)
            for path,payload,tail,count in ((actual_path,actual_payload,json.loads(actual_payload)["record_sha256"],1),
                                            (transaction_path,transaction_payload,marker_rows[-1]["record_sha256"],2)):
                anchors[path.name]={"record_count":count,"prefix_sha256":hashlib.sha256(payload).hexdigest(),"tail_sha256":tail}
        head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
        seed_path=run/"rollout_seed_audit.jsonl"
        seed_path.write_text("".join(json.dumps({
            "record_type":"trajectory_turn_seed","global_step":1,"sample_index":rank,
            "turn":0,"stable_example_id":f"frozen_train_row:{17+rank}",
            "trajectory_id":f"frozen_train_row:{17+rank}:seed:{101+rank}",
            "dataset_index":17+rank,"trajectory_seed":101+rank,
        },sort_keys=True,separators=(",",":"))+"\n" for rank in (0,1)))
        execution=root/"execution.jsonl"
        append_jsonl(execution,{"record_type":"weight_sync_summary","git_commit":head,"global_step":1,
                                "sync_kind":"post_actor_update","worker_ranks":[0,1],"sampled_tensor_digest":"f"*64})
        append_jsonl(execution,{"record_type":"checkpoint_inventory","git_commit":head,"global_step":1,
                                "inventory":checkpoint_inventory(ck),"rwwpo_ledger_anchors":anchors})
        output=root/"health.json"
        command=[sys.executable,"tools/h20/audit_rwwpo_run.py","--run-root",str(run),
                 "--actual-ledger-dir",str(ledger),"--execution-ledger",str(execution),
                 "--expected-commit",head,"--expected-schema-version","rwwpo-actual-loss-v2",
                 "--expected-objective","whole_prefix","--expected-controller","feasible_backtracking",
                 "--target-step","1","--output",str(output)]
        subprocess.run(command,check=True)
        assert json.loads(output.read_text())["status"]=="PASS"
        bad=command.copy(); bad[bad.index("whole_prefix")]="original_tokenwise"
        assert subprocess.run(bad,capture_output=True,text=True).returncode!=0
        forged=json.loads(seed_path.read_text().splitlines()[0]); forged["trajectory_id"]="forged"
        seed_path.write_text(json.dumps(forged)+"\n"+seed_path.read_text().splitlines()[1]+"\n")
        assert subprocess.run(command,capture_output=True,text=True).returncode!=0
    print("TF_RWWPO_AUDIT_SMOKE_PASS")


if __name__=="__main__": main()

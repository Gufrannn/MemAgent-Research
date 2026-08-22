#!/usr/bin/env python3
"""Dependency-free real-entry smoke for the TF-RWWPO v2 auditor."""
import hashlib
import json
import math
import tempfile
from pathlib import Path

from tools.h20.audit_rwwpo_actual_loss import audit


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
        "trajectory_turn":[0],"sample_index":[rank],"example_identity_hash":[100+rank],
        "trajectory_identity_hash":[200+rank],"advantages":[[1.0]],"denominator":1,
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
    print("TF_RWWPO_AUDIT_SMOKE_PASS")


if __name__=="__main__": main()

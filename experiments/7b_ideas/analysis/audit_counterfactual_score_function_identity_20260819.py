#!/usr/bin/env python3
"""CPU identity audit for the W4 v8 control-variate reference."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

EXPECTED={
  "schema_version":"counterfactual-score-function-identity-v8",
  "exact_noop_role":"control_variate_not_new_action_value_target",
  "noop_coupling_frozen_before_candidate":True,
  "noop_coupling_exogenous_given_state":True,
  "reader_seed_derivation":"pre_candidate_state_coupling_manifest",
  "rng_advance_candidate_length_dependent":False,
  "validity_frozen_before_tau":True,"truncation_frozen_before_tau":True,
  "row_selection_frozen_before_tau":True,"tau_or_outcome_conditioned_selection":False,
  "optimizer_steps":0,"new_rollouts":False,
}

def _fail(reason):raise ValueError(f"W4_NO_GO: {reason}; highest_level=W3")
def _close(a,b,tol=1e-10):return all(abs(x-y)<=tol for x,y in zip(a,b))

def audit(value):
    wrong={key:(value.get(key),expected) for key,expected in EXPECTED.items() if value.get(key)!=expected}
    if wrong:_fail(f"v8 exogeneity/pre-return freeze contract failed {wrong}")
    rows=value.get("candidates",[])
    if len(rows)<2:_fail("identity audit requires full candidate support")
    probabilities=[float(row["probability"]) for row in rows]
    if any(p<=0 or not math.isfinite(p) for p in probabilities) or abs(sum(probabilities)-1)>1e-10:
        _fail("candidate probabilities must be positive and sum to one")
    if any(row.get("selected") is not True for row in rows):_fail("post-candidate row filtering is not allowed in identity audit")
    baselines=[float(row["noop_baseline"]) for row in rows]
    if max(baselines)-min(baselines)>1e-12:_fail("candidate-dependent NOOP baseline")
    dimensions={len(row.get("score_gradient",[])) for row in rows}
    if len(dimensions)!=1 or not dimensions or next(iter(dimensions))==0:_fail("score gradient dimension mismatch")
    dimension=next(iter(dimensions));mean_score=[0.0]*dimension;direct=[0.0]*dimension;cf=[0.0]*dimension
    for probability,row,baseline in zip(probabilities,rows,baselines):
        policy=list(row.get("policy_controlled_token_kinds",[]));masked=list(row.get("score_mask_token_kinds",[]))
        if policy!=masked or "eos_or_stop" not in policy:
            _fail("writer score mask omits policy-controlled tokens or EOS/stop")
        gradient=[float(x) for x in row["score_gradient"]]
        commit=float(row["commit_return"])
        if not all(math.isfinite(x) for x in gradient+[commit,baseline]):_fail("non-finite identity input")
        for j,g in enumerate(gradient):
            mean_score[j]+=probability*g;direct[j]+=probability*commit*g
            cf[j]+=probability*(commit-baseline)*g
    if not _close(mean_score,[0.0]*dimension):_fail("score mean is not zero on full candidate support")
    supplied=[float(x) for x in value.get("direct_commit_return_gradient",[])]
    if len(supplied)!=dimension or not _close(direct,supplied):_fail("direct commit-return gradient control mismatch")
    if not _close(cf,direct):_fail("constant-baseline score-function identity mismatch")
    return {"status":"CSFGW_IDENTITY_V8_PASS","constant_baseline_control":True,
      "direct_commit_return_gradient_equal":True,"writer_mask_complete_including_stop":True,
      "pre_candidate_exogenous_coupling":True,"pre_tau_selection_freeze":True,
      "exact_noop_role":"control_variate_not_new_or_truer_action_value_target",
      "same_expected_writer_gradient":True,"algorithm_novelty":False,
      "highest_claim_level":"W3","w4_claim_authorized":False,"training_authorized":False,
      "optimizer_steps":0,"new_rollouts":False}

def self_test():
    rows=[{"candidate_id":"c0","probability":.5,"commit_return":1.0,"noop_baseline":.25,
      "score_gradient":[.5],"selected":True,"policy_controlled_token_kinds":["token","eos_or_stop"],
      "score_mask_token_kinds":["token","eos_or_stop"]},
      {"candidate_id":"c1","probability":.5,"commit_return":0.0,"noop_baseline":.25,
      "score_gradient":[-.5],"selected":True,"policy_controlled_token_kinds":["token","eos_or_stop"],
      "score_mask_token_kinds":["token","eos_or_stop"]}]
    base={**EXPECTED,"direct_commit_return_gradient":[.25],"candidates":rows}
    assert audit(base)["status"]=="CSFGW_IDENTITY_V8_PASS"
    negatives=[]
    bad=json.loads(json.dumps(base));bad["candidates"][1]["noop_baseline"]=.5;negatives.append(bad)
    bad=json.loads(json.dumps(base));bad["tau_or_outcome_conditioned_selection"]=True;bad["candidates"][1]["selected"]=False;negatives.append(bad)
    bad=json.loads(json.dumps(base));bad["candidates"][0]["score_mask_token_kinds"]=["token"];negatives.append(bad)
    for bad in negatives:
        try:audit(bad)
        except ValueError as exc:assert "W4_NO_GO" in str(exc)
        else:raise AssertionError("negative identity control passed")
    print("counterfactual_score_function_identity_v8_self_test=ok")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",type=Path);parser.add_argument("--self-test",action="store_true");args=parser.parse_args()
    if args.self_test:self_test();return
    if not args.manifest:parser.error("--manifest required")
    print(json.dumps(audit(json.loads(args.manifest.read_text())),indent=2,sort_keys=True))
if __name__=="__main__":main()

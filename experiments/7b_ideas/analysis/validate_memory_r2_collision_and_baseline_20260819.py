#!/usr/bin/env python3
"""Fail-closed Memory-R2 novelty/baseline preflight; never launches LoGo-GRPO."""
import argparse,json
from pathlib import Path
COLLIDING={"generic_blocked_within_state_comparison","same_anchor_local_rerollout","global_local_group_relative_credit","CERC_as_method","short_to_long_session_curriculum_as_method"}
MATCHED=("writer_mask","local_group_size","reward_information","rollout_budget","token_budget","scale")
def validate(value):
    collision=sorted(set(value.get("proposed_claims",[]))&COLLIDING)
    if collision: raise ValueError(f"NO_METHOD: Memory-R2 direct collision: {collision}")
    if value.get("candidate_unique_and_all_evidence_passed") is not True:
        return {"status":"PENDING_NO_BASELINE_IMPLEMENTATION","implementation_authorized":False,"training_authorized":False,"long_training_authorized":False}
    candidate=value.get("candidate_method",{}); baseline=value.get("memory_r2_like_baseline",{})
    missing=[key for key in MATCHED if candidate.get(key)!=baseline.get(key)]
    if missing: raise ValueError(f"NO_METHOD: Memory-R2-like baseline not matched: {missing}")
    if value.get("evidence_ledger_passed") is not True or value.get("single_extension_router_decision")!="SELECT_ONE":
        raise ValueError("NO_METHOD: evidence ledger and single-extension router must both pass")
    return {"status":"SCHEMA_READY_REQUIRES_SEPARATE_AUTHORIZATION","implementation_authorized":False,"training_authorized":False,"long_training_authorized":False}
def self_test():
    try: validate({"proposed_claims":["same_anchor_local_rerollout"]})
    except ValueError as exc: assert "Memory-R2 direct collision" in str(exc)
    else: raise AssertionError("collision accepted")
    assert validate({})["status"]=="PENDING_NO_BASELINE_IMPLEMENTATION"
    matched={key:"same" for key in MATCHED}
    out=validate({"candidate_unique_and_all_evidence_passed":True,"candidate_method":matched,"memory_r2_like_baseline":dict(matched),"evidence_ledger_passed":True,"single_extension_router_decision":"SELECT_ONE"})
    assert not out["implementation_authorized"] and not out["training_authorized"]
    print("memory_r2_collision_baseline_self_test=ok")
def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest");p.add_argument("--self-test",action="store_true");a=p.parse_args()
    if a.self_test:self_test();return
    if not a.manifest:p.error("--manifest required")
    print(json.dumps(validate(json.loads(Path(a.manifest).read_text())),indent=2,sort_keys=True))
if __name__=="__main__":main()

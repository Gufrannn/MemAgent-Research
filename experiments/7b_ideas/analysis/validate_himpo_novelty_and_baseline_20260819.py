#!/usr/bin/env python3
"""Fail-closed HiMPO collision/baseline preflight; never launches training."""
import argparse, json
from pathlib import Path

COLLIDING = {"local_counterfactual_memory_credit", "updated_vs_previous_target_answerability",
 "hindsight_filtered_memory_blame", "memory_token_only_local_advantage"}
MATCHED = ("writer_mask", "eligible_rows", "oracle_information", "forward_budget", "token_budget", "scale")
T0_FORBIDDEN = {"candidate_text", "materialized_updated_memory", "m_t", "new_vs_old_answerability",
 "updated_vs_previous_utility", "hindsight_target_score", "candidate_acceptance_score"}
def validate(value):
    if value.get("analysis_phase") == "T0_SHAPEA":
        features = set(value.get("feature_columns", []))
        leaked = sorted(features & T0_FORBIDDEN)
        if leaked or value.get("candidate_accessed") is not False:
            raise ValueError(f"NO_METHOD: T1 candidate-aware columns leaked into T0 manifest: {leaked}")
        allowed = {"pre_action_old_state", "direction_blind_raw_marginals", "P2_audit"}
        if not features <= allowed: raise ValueError(f"NO_METHOD: unregistered T0 feature columns: {sorted(features - allowed)}")
        return {"status": "T0_SHAPEA_CANDIDATE_FREE", "implementation_authorized": False,
          "training_authorized": False, "himpo_is_t0_baseline": False}
    proposed = set(value.get("proposed_claims", []))
    if proposed & COLLIDING: raise ValueError(f"NO_METHOD: HiMPO direct collision: {sorted(proposed & COLLIDING)}")
    if value.get("credit_definition") in {"updated_vs_previous_target_answerability_writer_reward", "old_new_answerability_delta"}:
        raise ValueError("NO_METHOD: updated-vs-previous target-answerability writer reward")
    if value.get("candidate_unique_and_all_evidence_passed") is not True:
        return {"status": "PENDING_NO_BASELINE_IMPLEMENTATION", "implementation_authorized": False,
          "training_authorized": False, "step400_authorized": False, "C256_authorized": False}
    baseline = value.get("himpo_like_baseline", {})
    missing = [key for key in MATCHED if baseline.get(key) != value.get("candidate_method", {}).get(key)]
    if missing: raise ValueError(f"NO_METHOD: HiMPO-like baseline not information/compute matched: {missing}")
    return {"status": "SCHEMA_READY_REQUIRES_SEPARATE_AUTHORIZATION", "implementation_authorized": False,
      "training_authorized": False, "step400_authorized": False, "C256_authorized": False}
def self_test():
    try: validate({"proposed_claims": ["local_counterfactual_memory_credit"]})
    except ValueError as exc: assert "HiMPO direct collision" in str(exc)
    else: raise AssertionError("collision accepted")
    assert validate({})["status"] == "PENDING_NO_BASELINE_IMPLEMENTATION"
    assert validate({"analysis_phase": "T0_SHAPEA", "candidate_accessed": False,
      "feature_columns": ["pre_action_old_state", "direction_blind_raw_marginals", "P2_audit"]})["status"] == "T0_SHAPEA_CANDIDATE_FREE"
    try: validate({"analysis_phase": "T0_SHAPEA", "candidate_accessed": True, "feature_columns": ["m_t"]})
    except ValueError as exc: assert "leaked into T0" in str(exc)
    else: raise AssertionError("T1 column accepted in T0")
    candidate = {key: "same" for key in MATCHED}
    out = validate({"candidate_unique_and_all_evidence_passed": True, "candidate_method": candidate,
      "himpo_like_baseline": dict(candidate)})
    assert out["status"] == "SCHEMA_READY_REQUIRES_SEPARATE_AUTHORIZATION" and not out["training_authorized"]
    print("himpo_novelty_baseline_self_test=ok")
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--manifest"); parser.add_argument("--self-test",action="store_true"); args=parser.parse_args()
    if args.self_test: self_test(); return
    if not args.manifest: parser.error("--manifest required")
    print(json.dumps(validate(json.loads(Path(args.manifest).read_text())),indent=2,sort_keys=True))
if __name__=="__main__": main()

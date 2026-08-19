#!/usr/bin/env python3
"""Fail-closed W4 counterfactual score-function gradient witness validator."""
from __future__ import annotations
import argparse,json,math,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from recurrent.research.counterfactual_gradient_witness import FORBIDDEN_EVIDENCE_BASES,vector_hash,witness_vectors

SHA256=re.compile(r"^[0-9a-f]{64}$")

GLOBAL_REQUIRED={
  "schema_version":"counterfactual-gradient-witness-v2","on_policy_same_checkpoint_candidate":True,
  "exact_noop_v2_qualified":True,"noop_baseline_candidate_independent":True,
  "noop_rng_independent":True,"noop_cache_independent":True,"writer_token_mask_exact":True,
  "noop_coupling_frozen_before_candidate":True,"noop_coupling_exogenous_given_state":True,
  "reader_seed_derivation":"pre_candidate_state_coupling_manifest",
  "rng_advance_candidate_length_dependent":False,
  "validity_frozen_before_tau":True,"truncation_frozen_before_tau":True,
  "row_selection_frozen_before_tau":True,"tau_or_outcome_conditioned_selection":False,
  "shared_suffix_endpoint_frozen":True,"actual_group_reconstructable":True,
  "actual_bonus_reconstructable":True,"actual_logprob_reconstructable":True,
  "actual_clip_reconstructable":True,"actual_kl_reconstructable":True,
  "optimizer_steps":0,"new_rollouts":False,
}

def _no_go(reason):raise ValueError(f"W4_NO_GO: {reason}; highest_level=W3")
def _dot(a,b):return sum(x*y for x,y in zip(a,b))
def _norm(a):return math.sqrt(_dot(a,a))

def validate(value:dict)->dict:
    wrong={key:(value.get(key),expected) for key,expected in GLOBAL_REQUIRED.items() if value.get(key)!=expected}
    if wrong:_no_go(f"reference/reconstruction contract failed {wrong}")
    if not SHA256.fullmatch(str(value.get("exact_noop_v2_manifest_hash",""))):_no_go("missing/invalid exact-NOOP v2 manifest hash")
    evidence=set(value.get("evidence_basis",[]))
    forbidden=sorted(evidence&FORBIDDEN_EVIDENCE_BASES)
    if forbidden:_no_go(f"forbidden evidence basis {forbidden}")
    threshold=value.get("material_effect_threshold")
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or threshold<=0:
        _no_go("material_effect_threshold must be positive and frozen")
    events=value.get("events")
    if not isinstance(events,list) or not events:_no_go("missing capture events")
    identities=[];metrics=[];material=0;total_mass=silent_mass=opposing_mass=0.0
    for index,event in enumerate(events):
        required=("stable_id","group_id","candidate_hash","exact_noop_pair_key_hash","checkpoint_hash",
          "subspace_hash","y_commit","y_retain","writer_token_score_gradients","writer_token_mask",
          "actual_grpo_writer_gradient","actual_candidate_hash","actual_group_id","actual_checkpoint_hash",
          "actual_subspace_hash","score_gradient_hash","actual_gradient_hash","policy_controlled_token_kinds",
          "writer_score_mask_includes_eos_or_stop","writer_score_mask_complete")
        missing=[key for key in required if key not in event]
        if missing:_no_go(f"event={index} missing={missing}")
        identity=(event["stable_id"],event["candidate_hash"],event["checkpoint_hash"])
        if identity in identities:_no_go(f"duplicate independent material event={identity}")
        identities.append(identity)
        if (event["candidate_hash"]!=event["actual_candidate_hash"] or
            event["group_id"]!=event["actual_group_id"] or
            event["checkpoint_hash"]!=event["actual_checkpoint_hash"] or
            event["subspace_hash"]!=event["actual_subspace_hash"]):
            _no_go(f"event={index} candidate/group/checkpoint/subspace mismatch")
        hash_fields=("candidate_hash","exact_noop_pair_key_hash","checkpoint_hash","subspace_hash")
        if any(not SHA256.fullmatch(str(event[name])) for name in hash_fields):
            _no_go(f"event={index} invalid immutable hash")
        if (event["writer_score_mask_includes_eos_or_stop"] is not True or
            event["writer_score_mask_complete"] is not True or
            "eos_or_stop" not in event["policy_controlled_token_kinds"] or
            len(event["policy_controlled_token_kinds"])!=len(event["writer_token_mask"]) or
            not all(event["writer_token_mask"])):
            _no_go(f"event={index} incomplete writer policy sequence/EOS-stop score mask")
        flattened=[float(x) for row in event["writer_token_score_gradients"] for x in row]
        if event["score_gradient_hash"]!=vector_hash(flattened) or event["actual_gradient_hash"]!=vector_hash(event["actual_grpo_writer_gradient"]):
            _no_go(f"event={index} gradient capture hash mismatch")
        try:tau,g_cf,g_e=witness_vectors(event)
        except (KeyError,TypeError,ValueError,IndexError) as exc:_no_go(f"event={index} malformed gradients: {exc}")
        if not g_cf or len(g_cf)!=len(g_e) or not all(math.isfinite(x) for x in g_cf+g_e):
            _no_go(f"event={index} gradient dimension/nonfinite failure")
        ncf,ne=_norm(g_cf),_norm(g_e);dot=_dot(g_cf,g_e)
        alignment=None if ncf==0 or ne==0 else dot/(ncf*ne)
        captured=None if ncf==0 else dot/(ncf*ncf)
        mass=abs(tau);total_mass+=mass
        if mass>=float(threshold):material+=1
        if ncf>0 and ne<=1e-12:silent_mass+=mass
        if dot<0:opposing_mass+=mass
        metrics.append({"stable_id":event["stable_id"],"tau":tau,"alignment":alignment,
          "captured_signed_ratio":captured,"g_cf_norm":ncf,"g_e_norm":ne})
    if len(events)<4:_no_go(f"pilot plumbing needs at least 4 independent captures, got {len(events)}")
    status="W4_CAPTURE_QUALIFIED_ANALYSIS_ONLY" if material>=20 else "W4_PLUMBING_ONLY"
    return {"status":status,"highest_claim_level":"W3","w4_claim_authorized":False,
      "training_authorized":False,"optimizer_steps":0,"new_rollouts":False,
      "independent_events":len(events),"independent_material_events":material,
      "scientific_audit_minimum":20,"event_metrics":metrics,
      "effect_weighted_silent_mass":silent_mass,"effect_weighted_opposing_mass":opposing_mass,
      "effect_weighted_silent_fraction":None if total_mass==0 else silent_mass/total_mass,
      "effect_weighted_opposing_fraction":None if total_mass==0 else opposing_mass/total_mass,
      "prohibited_inferences":["gradient_difference_norm","gradient_norm_only","scalar_advantage_sign","single_parameter_delta"]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",type=Path,required=True);args=parser.parse_args()
    print(json.dumps(validate(json.loads(args.manifest.read_text())),indent=2,sort_keys=True))
if __name__=="__main__":main()

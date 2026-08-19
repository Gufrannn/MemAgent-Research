#!/usr/bin/env python3
"""Fail-closed W4 v4 capture validator; never runs backward or an optimizer."""
from __future__ import annotations
import argparse,json,math,re,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from recurrent.research.counterfactual_gradient_witness import (FORBIDDEN_EVIDENCE_BASES,vector_hash,
    witness_vectors,writer_score_gradient_sum)

SHA256=re.compile(r"^[0-9a-f]{64}$")
GLOBAL_REQUIRED={
  "schema_version":"counterfactual-gradient-witness-v4","on_policy_same_checkpoint_candidate":True,
  "exact_noop_v2_qualified":True,"noop_baseline_candidate_independent":True,
  "noop_rng_independent":True,"noop_cache_independent":True,"writer_token_mask_exact":True,
  "noop_coupling_frozen_before_candidate":True,"noop_coupling_exogenous_given_state":True,
  "factual_retain_shared_crn":True,"reader_seed_derivation":"pre_candidate_state_coupling_manifest",
  "rng_advance_candidate_length_dependent":False,"coupling_plan_frozen_before_first_tau":True,
  "seeds_added_after_first_tau":False,"seed_selected_by_sign":False,
  "reader_repeats_increase_independent_n":False,"candidate_clustered_inference":True,
  "validity_frozen_before_tau":True,"truncation_frozen_before_tau":True,
  "row_selection_frozen_before_tau":True,"tau_or_outcome_conditioned_selection":False,
  "shared_suffix_endpoint_frozen":True,"actual_group_reconstructable":True,
  "actual_bonus_reconstructable":True,"actual_logprob_reconstructable":True,
  "actual_clip_reconstructable":True,"actual_kl_reconstructable":True,
  "same_loss_graph_decomposition":True,"gradient_components_reconstructable":True,
  "credit_uses_actual_group_advantage":True,"credit_token_broadcast_exact":True,
  "credit_clip_disabled":True,"credit_regularizers_disabled":True,
  "gradient_additivity_atol":1e-8,"optimizer_steps":0,"new_rollouts":False,
}

def _no_go(reason):raise ValueError(f"W4_NO_GO: {reason}; highest_level=W3")
def _dot(a,b):return sum(x*y for x,y in zip(a,b))
def _norm(a):return math.sqrt(_dot(a,a))
def _dispersion(xs):
    if len(xs)<2:return None
    mean=sum(xs)/len(xs);return math.sqrt(sum((x-mean)**2 for x in xs)/(len(xs)-1))

def _validate_event(event,index,atol):
    required=("stable_id","group_id","candidate_hash","exact_noop_pair_key_hash","checkpoint_hash",
      "subspace_hash","reader_coupling_id","y_commit","y_retain","writer_token_score_gradients",
      "writer_token_mask","credit_writer_gradient","task_writer_gradient","regularizer_writer_gradient",
      "total_writer_gradient","loss_graph_hash","actual_candidate_hash","actual_group_id",
      "actual_checkpoint_hash","actual_subspace_hash","score_gradient_hash","credit_gradient_hash",
      "task_gradient_hash","regularizer_gradient_hash","total_gradient_hash","policy_controlled_token_kinds",
      "writer_score_mask_includes_eos_or_stop","writer_score_mask_complete")
    missing=[key for key in required if key not in event]
    if missing:_no_go(f"event={index} missing={missing}")
    if not str(event["reader_coupling_id"]):_no_go(f"event={index} missing reader coupling id")
    if (event["candidate_hash"]!=event["actual_candidate_hash"] or event["group_id"]!=event["actual_group_id"] or
        event["checkpoint_hash"]!=event["actual_checkpoint_hash"] or event["subspace_hash"]!=event["actual_subspace_hash"]):
        _no_go(f"event={index} candidate/group/checkpoint/subspace mismatch")
    hashes=("candidate_hash","exact_noop_pair_key_hash","checkpoint_hash","subspace_hash","loss_graph_hash")
    if any(not SHA256.fullmatch(str(event[name])) for name in hashes):_no_go(f"event={index} invalid immutable hash")
    if (event["writer_score_mask_includes_eos_or_stop"] is not True or event["writer_score_mask_complete"] is not True or
        "eos_or_stop" not in event["policy_controlled_token_kinds"] or
        len(event["policy_controlled_token_kinds"])!=len(event["writer_token_mask"]) or not all(event["writer_token_mask"])):
        _no_go(f"event={index} incomplete writer policy sequence/EOS-stop score mask")
    flattened=[float(x) for row in event["writer_token_score_gradients"] for x in row]
    hash_inputs={"score_gradient_hash":flattened,"credit_gradient_hash":event["credit_writer_gradient"],
      "task_gradient_hash":event["task_writer_gradient"],"regularizer_gradient_hash":event["regularizer_writer_gradient"],
      "total_gradient_hash":event["total_writer_gradient"]}
    if any(event[name]!=vector_hash(vector) for name,vector in hash_inputs.items()):_no_go(f"event={index} gradient capture hash mismatch")
    try:tau,g_cf,g_credit,g_task,g_reg,g_total=witness_vectors(event)
    except (KeyError,TypeError,ValueError,IndexError) as exc:_no_go(f"event={index} malformed gradients: {exc}")
    vectors=(g_cf,g_credit,g_task,g_reg,g_total)
    if not g_cf or len({len(v) for v in vectors})!=1 or not all(math.isfinite(x) for v in vectors for x in v):
        _no_go(f"event={index} gradient dimension/nonfinite failure")
    if any(abs(total-task-reg)>atol for total,task,reg in zip(g_total,g_task,g_reg)):
        _no_go(f"event={index} G_total != G_task + G_reg; test-subspace closure failed")
    return tau,g_credit,g_task,g_reg,g_total

def validate(value:dict)->dict:
    wrong={key:(value.get(key),expected) for key,expected in GLOBAL_REQUIRED.items() if value.get(key)!=expected}
    if wrong:_no_go(f"reference/reconstruction contract failed {wrong}")
    if not SHA256.fullmatch(str(value.get("exact_noop_v2_manifest_hash",""))):_no_go("missing/invalid exact-NOOP v2 manifest hash")
    evidence=set(value.get("evidence_basis",[]));forbidden=sorted(evidence&FORBIDDEN_EVIDENCE_BASES)
    if forbidden:_no_go(f"forbidden evidence basis {forbidden}")
    threshold=value.get("material_effect_threshold")
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or threshold<=0:_no_go("material_effect_threshold must be positive and frozen")
    mode=value.get("coupling_mode");frozen_ids=value.get("frozen_reader_coupling_ids")
    if mode not in {"single_exogenous_crn","prefrozen_multiple_independent_crn"}:_no_go("invalid coupling mode")
    if not isinstance(frozen_ids,list) or not frozen_ids or len(frozen_ids)!=len(set(frozen_ids)):_no_go("invalid frozen reader coupling ids")
    expected=int(value.get("expected_couplings_per_candidate",0))
    if expected!=len(frozen_ids) or (mode=="single_exogenous_crn")!=(expected==1) or (mode=="prefrozen_multiple_independent_crn" and expected<2):
        _no_go("coupling mode/count mismatch")
    if mode=="single_exogenous_crn" and "candidate_stable_help_harm" in evidence:_no_go("single coupling cannot support candidate-stable help/harm")
    events=value.get("events")
    if not isinstance(events,list) or not events:_no_go("missing capture events")
    groups=defaultdict(list);seen=set();atol=float(value["gradient_additivity_atol"])
    for index,event in enumerate(events):
        prepared=_validate_event(event,index,atol)
        identity=(event["stable_id"],event["candidate_hash"],event["checkpoint_hash"],event["reader_coupling_id"])
        if identity in seen:_no_go(f"duplicate candidate-coupling capture={identity}")
        seen.add(identity);groups[identity[:3]].append((event,prepared))
    if len(groups)<4:_no_go(f"pilot plumbing needs at least 4 independent candidates, got {len(groups)}")
    metrics=[];material=0;total_mass=silent_mass=opposing_mass=0.0;adjudications={}
    for key,items in groups.items():
        if {event["reader_coupling_id"] for event,_ in items}!=set(frozen_ids) or len(items)!=expected:
            _no_go(f"candidate={key} does not contain exactly the prefrozen coupling set")
        invariants=("score_gradient_hash","credit_gradient_hash","task_gradient_hash","regularizer_gradient_hash",
          "total_gradient_hash","group_id","subspace_hash","loss_graph_hash")
        if any(len({event[name] for event,_ in items})!=1 for name in invariants):
            _no_go(f"candidate={key} gradient/group/subspace changed across reader repeats")
        taus=[prepared[0] for _,prepared in items];tau=sum(taus)/len(taus)
        event=items[0][0];_,g_credit,g_task,g_reg,g_total=items[0][1]
        g_cf=[tau*x for x in writer_score_gradient_sum(event)]
        ncf,ncredit,ntask=_norm(g_cf),_norm(g_credit),_norm(g_task)
        credit_dot=_dot(g_cf,g_credit);task_dot=_dot(g_cf,g_task);total_dot=_dot(g_cf,g_total)
        alignment=None if ncf==0 or ncredit==0 else credit_dot/(ncf*ncredit)
        captured=None if ncf==0 else credit_dot/(ncf*ncf)
        if credit_dot>0 and (ntask<=1e-12 or task_dot<0):adjudication="CLIP_OR_TRUST_REGION_DELIVERY_BOTTLENECK"
        elif task_dot>0 and total_dot<0:adjudication="REGULARIZATION_TRADEOFF"
        elif ncredit<=1e-12 and ncf>0:adjudication="CREDIT_SILENT"
        elif credit_dot<0:adjudication="CREDIT_OPPOSED"
        else:adjudication="CREDIT_ALIGNMENT_DESCRIPTIVE"
        adjudications[adjudication]=adjudications.get(adjudication,0)+1
        mass=abs(tau);total_mass+=mass;material+=mass>=float(threshold)
        silent_mass+=mass if ncf>0 and ncredit<=1e-12 else 0;opposing_mass+=mass if credit_dot<0 else 0
        sign_stability=None
        if len(taus)>1:
            sign=0 if tau==0 else (1 if tau>0 else -1)
            sign_stability=sum((0 if x==0 else (1 if x>0 else -1))==sign for x in taus)/len(taus)
        metrics.append({"stable_id":key[0],"candidate_hash":key[1],"coupling_count":len(taus),
          "tau_estimate":tau,"tau_scope":"realized_coupling" if len(taus)==1 else "prefrozen_coupling_mean",
          "within_candidate_tau_dispersion":_dispersion(taus),"coupling_sign_stability":sign_stability,
          "credit_alignment":alignment,"credit_captured_signed_ratio":captured,
          "task_alignment_sign":0 if task_dot==0 else (1 if task_dot>0 else -1),
          "total_alignment_sign":0 if total_dot==0 else (1 if total_dot>0 else -1),
          "delivery_adjudication":adjudication})
    status="W4_CAPTURE_QUALIFIED_ANALYSIS_ONLY" if material>=20 else "W4_PLUMBING_ONLY"
    scope="realized_coupling_only" if mode=="single_exogenous_crn" else "prefrozen_candidate_mean"
    prefix="realized_coupling" if mode=="single_exogenous_crn" else "candidate_mean"
    return {"status":status,"highest_claim_level":"W3","w4_claim_authorized":False,"training_authorized":False,
      "optimizer_steps":0,"new_rollouts":False,"independent_candidates":len(groups),
      "reader_coupling_rows":len(events),"reader_repeats_increase_independent_n":False,
      "independent_material_events":material,"scientific_audit_minimum":20,"event_metrics":metrics,
      "aggregate_g_cf_mc_sample_valid":True,"coupling_scope":scope,
      "candidate_stable_help_harm_authorized":mode=="prefrozen_multiple_independent_crn",
      "credit_primary_only":True,"delivery_adjudication_counts":adjudications,
      f"{prefix}_credit_effect_weighted_silent_mass":silent_mass,
      f"{prefix}_credit_effect_weighted_opposing_mass":opposing_mass,
      f"{prefix}_credit_effect_weighted_silent_fraction":None if total_mass==0 else silent_mass/total_mass,
      f"{prefix}_credit_effect_weighted_opposing_fraction":None if total_mass==0 else opposing_mass/total_mass,
      "prohibited_inferences":["gradient_difference_norm","gradient_norm_only","scalar_advantage_sign","single_parameter_delta"]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",type=Path,required=True);args=parser.parse_args()
    print(json.dumps(validate(json.loads(args.manifest.read_text())),indent=2,sort_keys=True))
if __name__=="__main__":main()

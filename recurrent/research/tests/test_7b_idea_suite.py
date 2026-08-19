import json
from pathlib import Path

import pytest
import torch
import importlib.util

from recurrent.research.cerc_native_credit import validate_native_credit
from recurrent.research.idea_admissibility import ELIGIBLE, NO_METHOD, PENDING, adjudicate, require_arm
from recurrent.research.exact_noop_v2 import (E_LABEL,H_LABEL,P_LABEL,ExactNoopV2ReplayBuilder,
    build_arm_record,build_vg_record,classify_estimand,materialize_proposal)
from recurrent.research.counterfactual_gradient_witness import capture_w4_event,vector_hash
from recurrent.research.ncr_certified_routing import generic_auxiliary, route_ncr
from recurrent.research.typed_boundary_prompt import ARMS, build_prompts


def eligible_row():
    gates = {key: True for key in (
        "shape_a_candidate_free_t0", "exact_linked_same_write_key", "exact_tie_coverage",
        "frozen_readout", "writer_only", "non_tie_bitwise_unchanged", "gradient_safety",
        "beats_qa_only", "beats_generic_qa", "beats_generic_judge",
        "beats_information_matched_raw_judge", "beats_generic_tie_rescue")}
    gates.update({"himpo_non_equivalence": True, "himpo_like_baseline_matched": True})
    gates.update({"memory_r2_non_equivalence": True, "memory_r2_like_baseline_matched": True})
    gates["exact_noop_v2_qualified"] = True
    return {"event_id": "e1", "timestamp": "2026-08-19T00:00:00Z", "candidate": "ncr_certified_routing",
            "status": "eligible", "gates": gates,
            "shape_a": {"t0_formula": "P2_raw^T0 vs P2_raw^T0+D_pre^audit", "t1_leaks_into_t0": False},
            "shape_a_contract": {"independent_unit": "stable_example_id", "max_independent_n": 128,
              "primary_representation": "paired_H_H", "stacked_role": "implementation_consistency_audit_only",
              "count_paired_and_stacked_as_one": True, "select_more_significant_representation": False,
              "b_raw": "H_H~P2_raw_T0", "b_struct": "H_H~P2_raw_T0+D_star", "d_star_dimensions": 1,
              "outer_grouped_folds": 4, "model_capacity": "low_capacity_linear", "harm_events": 19,
              "multivariable_logistic_primary": False, "auroc_primary": False,
              "arms_increase_independent_n": False, "turns_increase_independent_n": False,
              "seeds_increase_independent_n": False, "tokens_increase_independent_n": False}}


def add_firewall(row):
    row["shape_a_contract"]["inference_firewall"] = {
      "fold_level_t_or_wilcoxon": False, "repeated_cv_is_independent_replication": False,
      "restricted_d_permutation_label": "artifact_sensitivity_not_exact_crt",
      "central_evidence_rule": "A_AND_B_AND_C", "allow_choose_a_b_or_c": False,
      "arm_x_d_in_maxT": False, "algebra_audit_outputs_p_values": False,
      "algebra_audit_authorizes_claim": False,
      "false_positive_counterexamples": ["pure_difficulty", "p2_redundancy", "role_prevalence", "single_outlier",
        "heteroskedasticity_only", "fold_accident", "regularization_suppression", "coarse_permutation_artifact"]}
    row["shape_a_contract"].update({
      "central_claim": "preregistered relational compression of the same audit transcript predicts commit-vs-discard effect beyond frozen direction-blind marginal summaries and matched pairing shams.",
      "d_input_provenance": {"cells": ["audit"], "roles": ["writer"], "targets": ["same_write"], "masks": ["valid"],
        "normalization": "frozen", "baseline_retained_fields": ["marginal_mean"], "baseline_discarded_fields": ["pairing"],
        "deterministically_reconstructable": True, "d_access_tier": "P2", "baseline_access_tier": "P2",
        "query_budget": 1, "token_budget": 128, "gpu_budget": 0, "baseline_contains_full_transcript_metadata": False,
        "semantic_class": "M_RELATIONAL_COMPRESSION"},
      "outcome_free_representation_audit": {"residual_variance_ratio": "required", "condition_number": "required",
        "vif": "required", "role_overlap": "required", "interpretation": "model_class_representation_audit_not_mutual_information"},
      "cpu_sensitivity": {"full_transcript_frozen_random_projection": "not_available_existing_outputs_insufficient",
        "matched_pairing_sham": "available", "expand_rollout": False},
      "semantic_pairing_null": {"k": 2000, "same_response_tensor": True, "same_pipeline": True,
        "exact_crt_p_value": False, "second_primary": False, "c_layer_falsification_only": True,
        "learned_pairing_or_gnn_rescue_on_same_b128": False, "seed": 2026,
        "allowed_edges_hash": "a"*64, "generator_sha": "b"*64, "folds_hash": "c"*64, "manifest_hash": "d"*64},
      "dstar_measurement": {"status": "MEASUREMENT_RELIABLE", "independent_semantic_replicas": True,
        "isomorphic_replica_contract": True, "deterministic_rerun": False, "audit_hash": "e"*64}})
    return row


def write_ledger(tmp_path: Path, rows):
    path = tmp_path / "evidence.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


# All eligible fixtures must carry the dependence firewall.
_eligible_row_without_firewall = eligible_row
def eligible_row():
    return add_firewall(_eligible_row_without_firewall())


def test_pending_and_missing_manifest_fail_closed(tmp_path):
    pending = eligible_row(); pending["status"] = "pending"
    with pytest.raises(ValueError, match=PENDING): require_arm("ncr_certified_routing", None)
    with pytest.raises(ValueError, match=PENDING): require_arm("ncr_certified_routing", write_ledger(tmp_path, [pending]))


def test_exactly_one_ncr_candidate_and_all_gates_required(tmp_path):
    path = write_ledger(tmp_path, [eligible_row()])
    assert require_arm("ncr_certified_routing", path).status == ELIGIBLE
    bad = eligible_row(); bad["gates"]["frozen_readout"] = False
    with pytest.raises(ValueError, match=NO_METHOD): require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))


def test_shape_a_paired_primary_stacked_not_double_counted(tmp_path):
    bad = eligible_row(); bad["shape_a_contract"]["count_paired_and_stacked_as_one"] = False
    with pytest.raises(ValueError, match="Shape A contract"): require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))
    bad = eligible_row(); bad["shape_a_contract"]["select_more_significant_representation"] = True
    with pytest.raises(ValueError, match="Shape A contract"): require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))


def test_shape_a_independent_n_and_low_dimensional_contract(tmp_path):
    for key, value in (("max_independent_n", 256), ("d_star_dimensions", 2), ("outer_grouped_folds", 5),
                       ("model_capacity", "random_forest"), ("arms_increase_independent_n", True)):
        bad = eligible_row(); bad["shape_a_contract"][key] = value
        with pytest.raises(ValueError, match="Shape A contract|independent n"):
            require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))


def test_shape_a_rare_harm_forbids_logistic_and_auroc_primary(tmp_path):
    for key in ("multivariable_logistic_primary", "auroc_primary"):
        bad = eligible_row(); bad["shape_a_contract"][key] = True
        with pytest.raises(ValueError, match="harm_events=19<20"):
            require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))


def test_shape_a_dependence_firewall_and_eight_counterexamples(tmp_path):
    mutations = (("fold_level_t_or_wilcoxon", True), ("repeated_cv_is_independent_replication", True),
                 ("restricted_d_permutation_label", "exact_CRT"), ("central_evidence_rule", "A_OR_B_OR_C"),
                 ("allow_choose_a_b_or_c", True), ("arm_x_d_in_maxT", True))
    for key, value in mutations:
        bad = eligible_row(); bad["shape_a_contract"]["inference_firewall"][key] = value
        with pytest.raises(ValueError, match="dependence firewall"):
            require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))
    bad = eligible_row(); bad["shape_a_contract"]["inference_firewall"]["false_positive_counterexamples"].pop()
    with pytest.raises(ValueError, match="dependence firewall"):
        require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))


def test_shape_a_data_processing_claim_and_provenance_firewall(tmp_path):
    for claim in ("same-information adds information", "structural information gain", "I(tau;D|R,M)>0"):
        bad = eligible_row(); bad["shape_a_contract"]["central_claim"] = claim
        with pytest.raises(ValueError, match="information-gain claim"):
            require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))
    bad = eligible_row(); del bad["shape_a_contract"]["d_input_provenance"]
    with pytest.raises(ValueError, match="UNKNOWN_PROVENANCE_NO_GO"):
        require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))
    bad = eligible_row(); bad["shape_a_contract"]["d_input_provenance"]["d_access_tier"] = "P3"
    with pytest.raises(ValueError, match="A_ACCESS_MISMATCH_NO_GO"):
        require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))


def test_shape_a_full_transcript_semantic_is_inductive_bias(tmp_path):
    bad = eligible_row(); prov = bad["shape_a_contract"]["d_input_provenance"]
    prov["baseline_contains_full_transcript_metadata"] = True
    with pytest.raises(ValueError, match="F_DETERMINISTIC_INDUCTIVE_BIAS"):
        require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))
    prov["semantic_class"] = "F_DETERMINISTIC_INDUCTIVE_BIAS"
    assert require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad])).status == ELIGIBLE


def test_estimand_equivalence_auditor_and_fail_closed_guards():
    path = Path(__file__).parents[3] / "experiments/7b_ideas/analysis/audit_shapeA_estimand_equivalence_20260819.py"
    spec = importlib.util.spec_from_file_location("shape_a_audit", path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def row(stable,write,d,count,**updates):
        value={"stable_example_id":stable,"checkpoint_hash":"a"*64,"write_id":write,
          "prebranch_eligibility_manifest_hash":"b"*64,"target_write_count":count,
          "measurement_available":True,"measurement_exclusion_reason":None,"eligible_write_count":count,
          "eligible":True,"attempted":True,"factual_commit_complete":True,"noop_retain_complete":True,
          "pair_qualified":True,"failure_reason":None,"shapeA_stage":"audit32","experiment_name":"shape-a-test",
          "turn_type":"update","required_component_pattern":"all_required","required_components_complete":True,
          "joint_null_stratum_yield_sufficient":True,"calibration_stratum_key":"update|all_required",
          "dstar_calibration_manifest_hash":"c"*64,"dstar_raw":d,"joint_null_median":0.0,
          "joint_null_mad":1.0,"d_star":max(-5,min(5,d)),"y_commit":1+d,"y_retain":2+3*d}
        value.update(updates);return value
    rows=[]
    for i,writes in enumerate((1,2,3,1,2,3,1,2)):
        for j in range(writes):
            d=float(i)/2+j/10;rows.append(row(f"e{i}",f"w{j}",d,writes))
    result = module.audit(rows)
    assert result["algebraically_equivalent"] and not result["claim_authorized"] and not result["p_values_emitted"]
    assert result["independent_n"]==8 and result["write_rows"]==15
    assert result["weight"]=="1/m_i^elig" and not result["stacked_is_second_evidence"]
    assert result["row_key"]==["stable_example_id","checkpoint_hash","write_id"]
    assert result["outcome"]=="H_H=y_retain-y_commit" and not result["tau_H_used_as_primary"]
    with pytest.raises(ValueError, match="unique"): module.audit(rows + [rows[0]])
    too_many=[row(f"x{i}","w0",(i%9)/2,1) for i in range(129)]
    with pytest.raises(ValueError, match=r"\[2,128\]"): module.audit(too_many)
    with pytest.raises(ValueError, match="rank deficient"):
        module.audit([{**item,"dstar_raw":1,"d_star":1} for item in rows])
    with pytest.raises(ValueError,match="ledger fields"):module.audit([{k:v for k,v in item.items() if k!="write_id"} for item in rows])
    with pytest.raises(ValueError,match="row-level independence/HC3"):module.audit([dict(item,row_level_hc3=True) for item in rows])
    oof=module.aggregate_oof_loss([dict(item,oof_loss=float(index)) for index,item in enumerate(rows)])
    assert oof["independent_n"]==8 and oof["aggregation"]=="within_example_1_over_m_elig_then_across_examples"

    missing=[dict(item) for item in rows];missing[2].update({"noop_retain_complete":False,"pair_qualified":False,
      "failure_reason":"retain_endpoint_failed"});missing[2].pop("y_retain")
    diagnostic=module.audit(missing)
    assert diagnostic["status"]=="CONSTRUCTION_DIAGNOSTIC_ONLY"
    assert diagnostic["eligible_weight_coverage"]<1 and not diagnostic["missing_weight_reallocated"]
    assert diagnostic["observed_postbranch_missing_weight"]>0
    with pytest.raises(ValueError,match="SHAPEA_PRIMARY_COVERAGE_FAIL"):module.audit(missing,require_primary=True)

    target=[dict(item) for item in rows]
    target[0]["target_write_count"]=2
    target.append(row("e0","w_unmeasured",0.0,2,measurement_available=False,measurement_exclusion_reason="required_component_missing",
      eligible_write_count=1,eligible=False,attempted=False,factual_commit_complete=False,noop_retain_complete=False,
      pair_qualified=False,failure_reason=None,required_components_complete=False,joint_null_stratum_yield_sufficient=False))
    for key in ("dstar_raw","joint_null_median","joint_null_mad","d_star","y_commit","y_retain"):
        target[-1].pop(key,None)
    measured=module.audit(target)
    assert measured["measurement_availability_rate_R1_over_target"]<1
    assert measured["paired_closure_coverage_R1"]==1 and measured["measurement_R0_exclusion_ledger"][0]["arm_failure"] is False
    bad_components=[dict(item) for item in rows];bad_components[0]["required_components_complete"]=False
    with pytest.raises(ValueError,match="R=1 eligible write requires complete components"):
        module.audit(bad_components)
    wrong_d=[dict(item) for item in rows];wrong_d[2]["d_star"]=0.0
    with pytest.raises(ValueError,match="continuous joint-null median/MAD"):
        module.audit(wrong_d)


def test_outcome_blind_provenance_preflight_four_classes_and_mismatch():
    path = Path(__file__).parents[3] / "experiments/7b_ideas/analysis/validate_shapeA_D_input_provenance_20260819.py"
    spec = importlib.util.spec_from_file_location("prov", path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    base = {"d_response_cells": ["r1"], "d_role_metadata": ["writer"], "d_target_metadata": ["same_write"],
      "d_structural_ops": ["pair"], "d_normalization": "frozen", "d_human_oracle_labels": [],
      "baseline_visible_cells": ["r1"], "baseline_visible_metadata": ["writer"],
      "baseline_marginal_summaries": ["mean"], "baseline_discarded_relational_structure": ["pairing"],
      "baseline_contains_full_transcript_metadata": False, "d_deterministically_reconstructable": True,
      "d_object": "audit", "baseline_object": "audit", "d_budget": {"queries": 1, "tokens": 10, "gpu": 0},
      "baseline_budget": {"queries": 1, "tokens": 10, "gpu": 0}}
    assert module.validate({**base, "expected_classification": module.M})["classification"] == module.M
    full = {**base, "baseline_contains_full_transcript_metadata": True, "expected_classification": module.F}
    assert module.validate(full)["classification"] == module.F
    mismatch = {**base, "baseline_object": "other", "expected_classification": module.A}
    assert module.validate(mismatch)["classification"] == module.A
    unknown = dict(base); del unknown["d_normalization"]
    assert module.validate(unknown)["classification"] == module.U
    with pytest.raises(ValueError, match="expected classification mismatch"):
        module.validate({**base, "expected_classification": module.F})
    with pytest.raises(ValueError, match="forbidden harm/outcome"):
        module.validate({**base, "harm": 1})


def test_semantic_pairing_null_generator_manifest_and_spe():
    path = Path(__file__).parents[3] / "experiments/7b_ideas/analysis/semantic_pairing_null_20260819.py"
    spec = importlib.util.spec_from_file_location("pairing", path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    kwargs = {"roles": ["writer", "reader"], "targets": ["fact", "answer"],
      "allowed_edges": [("writer", "answer"), ("reader", "fact")], "seed": 7,
      "folds_hash": "a"*64, "generator_sha": "b"*64}
    first = module.generate(**kwargs); second = module.generate(**kwargs)
    assert first == second and len(first["mappings"]) == 2000
    module.validate(first, expected_hash=first["config_hash"])
    with pytest.raises(ValueError, match="manifest/hash mismatch"):
        module.validate(first, expected_hash="0"*64)
    positive = module.spe_decision(delta_real=2, delta_sham=[0.5]*2000, sesoi=1,
      leave_role_stable=True, checkpoint_stable=True)
    assert positive["classification"] == "obligation_semantic_relational_compression"
    generic = module.spe_decision(delta_real=.2, delta_sham=[.3]*2000, sesoi=.1,
      leave_role_stable=True, checkpoint_stable=True)
    assert "obligation_shapeA_NO_GO" in generic["classification"]
    assert not generic["exact_crt_p_value"] and not generic["second_primary"]


def test_semantic_pairing_null_is_mandatory_for_training_ledger(tmp_path):
    bad = eligible_row(); bad["shape_a_contract"]["semantic_pairing_null"]["k"] = 1999
    with pytest.raises(ValueError, match="semantic pairing null"):
        require_arm("ncr_certified_routing", write_ledger(tmp_path, [bad]))


def test_dstar_measurement_gate_variants_and_ledger(tmp_path):
    path=Path(__file__).parents[3]/"experiments/7b_ideas/analysis/audit_dstar_semantic_reliability_20260819.py"
    spec=importlib.util.spec_from_file_location("dstar",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows=[{"stable_example_id":f"e{i}","D_a":i,"D_b":i+.01*(i%3),"valid_a":True,"valid_b":True,
      "role":"r" if i%2 else "w","checkpoint":"25"} for i in range(32)]
    assert module.audit(rows,independent_semantic_replicas=True,isomorphic_replica_contract=True)["status"]=="MEASUREMENT_RELIABLE"
    assert module.audit(rows[:20],independent_semantic_replicas=True,isomorphic_replica_contract=True)["status"]=="MEASUREMENT_INCOMPLETE"
    assert module.audit(rows,independent_semantic_replicas=True,isomorphic_replica_contract=True,deterministic_rerun=True)["status"]=="MEASUREMENT_NOT_IDENTIFIED"
    assert module.audit(rows,independent_semantic_replicas=True,isomorphic_replica_contract=False)["status"]=="MEASUREMENT_NOT_IDENTIFIED"
    bad=eligible_row(); bad["shape_a_contract"]["dstar_measurement"]["status"]="MEASUREMENT_INCOMPLETE"
    with pytest.raises(ValueError,match="measurement not identified/reliable"):
        require_arm("ncr_certified_routing",write_ledger(tmp_path,[bad]))


def test_cerc_same_variant_native_variance_is_control_only():
    rows = [{"uid": "q", "variant_id": "v", "qa_reward": x} for x in (0, 1)]
    assert validate_native_credit(rows)["adds_reward"] is False


def test_cerc_mixed_group_illusion_and_all_tie_rejected():
    rows = [{"uid": "q", "variant_id": v, "qa_reward": r} for v, r in (("a", 0), ("a", 0), ("b", 1), ("b", 1))]
    with pytest.raises(ValueError, match="MIXED_GROUP_ILLUSION"): validate_native_credit(rows)
    with pytest.raises(ValueError, match="no same-prompt/same-variant"):
        validate_native_credit([{"uid": "q", "variant_id": "a", "qa_reward": 0}] * 2)


def test_ncr_non_tie_bitwise_unchanged_and_writer_only():
    qa = torch.tensor([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    result, meta = route_ncr(qa_advantage=qa, secondary_score=torch.tensor([0., 1., 1., 2.]),
        uid=["tie", "tie", "var", "var"], qa_reward=torch.tensor([0., 0., 0., 1.]),
        writer_mask=torch.tensor([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=torch.bool),
        final_mask=torch.tensor([False, True, False, False]), eligible=torch.ones(4, dtype=torch.bool),
        exact_correct=torch.zeros(4, dtype=torch.bool), lambda_=0.5)
    assert torch.equal(result[2:], qa[2:])
    assert torch.equal(result[1], qa[1])
    assert torch.equal(result[:, 1], qa[:, 1])
    assert meta["writer_only"]


def test_ncr_all_exact_correct_rejected():
    with pytest.raises(ValueError, match="all-exact-correct"):
        route_ncr(qa_advantage=torch.zeros(2, 1), secondary_score=torch.tensor([0., 1.]), uid=["q", "q"],
          qa_reward=torch.ones(2), writer_mask=torch.ones(2, 1, dtype=torch.bool), final_mask=torch.zeros(2, dtype=torch.bool),
          eligible=torch.ones(2, dtype=torch.bool), exact_correct=torch.ones(2, dtype=torch.bool), lambda_=1)


def test_generic_baselines_never_read_bot_noop_labels():
    assert generic_auxiliary([{"generic_qa_score": .2}], "generic_qa_score") == [.2]
    with pytest.raises(ValueError, match="forbidden outcome"):
        generic_auxiliary([{"generic_qa_score": .2, "BOT": 1}], "generic_qa_score")


def test_typed_five_arms_budget_and_permutation_reproducible():
    first = build_prompts("memory", token_budget=64, seed=7)
    second = build_prompts("memory", token_budget=64, seed=7)
    assert tuple(first) == ARMS and first == second
    assert {row["token_budget"] for row in first.values()} == {64}
    assert all(row["diagnostic_only"] for row in first.values())


def test_typed_and_cerc_are_never_training_authorized(tmp_path):
    path = write_ledger(tmp_path, [eligible_row()])
    with pytest.raises(ValueError, match=NO_METHOD): require_arm("cerc_native_credit", path)
    with pytest.raises(ValueError, match=NO_METHOD): require_arm("typed_boundary_prompt_control", path)


def test_budget_metadata_equal_for_information_matched_arms():
    path = Path(__file__).parents[3] / "experiments/7b_ideas/configs/budgets.json"
    budgets = json.loads(path.read_text())
    compare = [budgets[x] for x in ("ncr_certified_routing", "generic_qa_aux", "generic_frozen_judge_tournament", "information_matched_raw_judge", "uniform_tie_rescue")]
    keys = ("questions", "rollout_n", "max_prompt", "max_response", "forward_budget")
    assert all(tuple(row[k] for k in keys) == tuple(compare[0][k] for k in keys) for row in compare)


def test_training_entrypoint_reaches_unified_router():
    trainer = (Path(__file__).parents[3] / "verl/trainer/ppo/ray_trainer.py").read_text()
    assert "from recurrent.research.idea_router import apply_idea_arm" in trainer
    assert "advantages = apply_idea_arm(" in trainer


def test_launcher_fresh_resume_strict_vllm_and_frozen_anchors():
    launcher = (Path(__file__).parents[3] / "experiments/7b_ideas/run_7b_idea.sh").read_text()
    for marker in ("fresh2", "resume3", "global_step_2", "actor_rollout_ref.rollout.name=vllm",
                   "trajectory_seed_mode=independent", "CONFIRM_EXTENDED_RUN", "TERMINAL_RULE_FROZEN",
                   "25|50|100|200", "Refusing to overwrite run directory"):
        assert marker in launcher
    assert "never automatic 400" in launcher


def test_original_requires_no_evidence_and_is_exact_noop():
    assert require_arm("qa_only_original", None).training_authorized


def test_shape_a_schema_v8_mechanical_downgrades():
    path=Path(__file__).parents[3]/"experiments/7b_ideas/analysis/adjudicate_shapeA_structural_claim_20260819.py"
    spec=importlib.util.spec_from_file_location("v8",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.self_test()
    assert all(not module._result("x","y",{})[key] for key in ("method_training_authorized","online_deployment_claim_authorized"))


def test_himpo_t1_columns_hard_fail_in_t0_manifest():
    path=Path(__file__).parents[3]/"experiments/7b_ideas/analysis/validate_himpo_novelty_and_baseline_20260819.py"
    spec=importlib.util.spec_from_file_location("himpo",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    good={"analysis_phase":"T0_SHAPEA","candidate_accessed":False,
      "feature_columns":["pre_action_old_state","direction_blind_raw_marginals","P2_audit"]}
    assert module.validate(good)["himpo_is_t0_baseline"] is False
    for column in ("candidate_text","m_t","new_vs_old_answerability","updated_vs_previous_utility"):
        with pytest.raises(ValueError,match="leaked into T0"):
            module.validate({"analysis_phase":"T0_SHAPEA","candidate_accessed":False,"feature_columns":[column]})


def test_memory_r2_collision_and_dual_launcher_firewall():
    path=Path(__file__).parents[3]/"experiments/7b_ideas/analysis/validate_memory_r2_collision_and_baseline_20260819.py"
    spec=importlib.util.spec_from_file_location("memory_r2",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for claim in ("generic_blocked_within_state_comparison","same_anchor_local_rerollout","CERC_as_method","short_to_long_session_curriculum_as_method"):
        with pytest.raises(ValueError,match="Memory-R2 direct collision"):
            module.validate({"proposed_claims":[claim]})
    assert module.validate({})["status"]=="PENDING_NO_BASELINE_IMPLEMENTATION"
    launcher=(Path(__file__).parents[3]/"experiments/7b_ideas/run_7b_idea.sh").read_text()
    assert "MEMORY_R2_BASELINE_REQUEST" in launcher
    assert "IDEA_EVIDENCE_LEDGER" in launcher and "MECHANISM_EXTENSION_DECISION" in launcher
    assert "LoGo-GRPO implementation and training are not authorized" in launcher


def _e_proposal():
    return materialize_proposal(stable_id="e1",turn=2,upstream_state_hash="a"*64,
      candidate_token_ids=[10,20],writer_seed=7,
      reader_seed_or_coupling_id="coupled-reader-1",endpoint_version="em-v1",
      old_memory_token_hash="c"*64,estimand_mode="EH",
      mode_contract={"exogenous_suffix_contract_hash":"b"*64,"future_policy_hash":"d"*64,"horizon":3})


def _build_e_record(proposal,arm,endpoint):
    return build_arm_record(proposal,arm=arm,output_token_ids=[1 if arm=="commit" else 2],
      endpoint_value=endpoint,realized_trajectory_hash=("e" if arm=="commit" else "f")*64,
      future_hash_chain=[("1" if arm=="commit" else "2")*64])


def _exact_noop_validator():
    path=Path(__file__).parents[3]/"experiments/7b_ideas/analysis/validate_exact_noop_v2_manifest_20260819.py"
    spec=importlib.util.spec_from_file_location("exact_noop_v2_validator",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_p_h_e_estimands_are_mutually_exclusive_and_not_selected_by_outcome():
    assert classify_estimand({"same_anchor_policy_rerollout":True})==P_LABEL
    assert classify_estimand({"teacher_forced_updated_vs_old_target_answerability":True})==H_LABEL
    assert classify_estimand({"same_materialized_candidate":True,"shared_suffix_real_endpoint":True,
      "commit_retain_only_intervention":True})==E_LABEL
    with pytest.raises(ValueError,match="mutually exclusive"):
        classify_estimand({"same_anchor_policy_rerollout":True,
          "teacher_forced_updated_vs_old_target_answerability":True})


def test_exact_noop_v2_materializes_once_and_never_reruns_writer():
    builder=ExactNoopV2ReplayBuilder(_e_proposal())
    commit=builder.record_endpoint(arm="commit",output_token_ids=[1],endpoint_value=1.0,
      realized_trajectory_hash="e"*64,future_hash_chain=["1"*64])
    retain=builder.record_endpoint(arm="retain",output_token_ids=[2],endpoint_value=0.0,
      realized_trajectory_hash="f"*64,future_hash_chain=["2"*64])
    assert commit["candidate_memory_token_hash"]==retain["candidate_memory_token_hash"]
    assert commit["loaded_memory_token_hash"]==commit["candidate_memory_token_hash"]
    assert retain["loaded_memory_token_hash"]!=retain["candidate_memory_token_hash"]
    assert len(builder.complete_pair())==2
    with pytest.raises(ValueError,match="no last-write-wins"):
        builder.record_endpoint(arm="commit",output_token_ids=[3],endpoint_value=.5,
          realized_trajectory_hash="3"*64,future_hash_chain=["4"*64])
    with pytest.raises(ValueError,match="writer ran"):
        ExactNoopV2ReplayBuilder(_e_proposal()).record_endpoint(
          arm="commit",output_token_ids=[1],endpoint_value=1,writer_ran=True)


def test_exact_noop_v2_join_fails_closed_on_duplicates_missing_and_pair_mismatch():
    module=_exact_noop_validator();proposal=_e_proposal()
    rows=[_build_e_record(proposal,"commit",1),_build_e_record(proposal,"retain",0)]
    result=module.validate(rows)
    assert result["status"]=="E_QUALIFIED" and result["pair_count"]==1
    assert not result["state_level_writer_policy_risk_identified"]
    assert not result["training_authorized"] and not result["select_best_estimand"]
    invalid=[rows+[dict(rows[0])],[{k:v for k,v in rows[0].items() if k!="exogenous_suffix_contract_hash"},rows[1]],
      [dict(rows[0],suffix_hash="9"*64),rows[1]]]
    for key in ("exogenous_suffix_contract_hash","future_policy_hash","horizon","endpoint_version","reader_seed_or_coupling_id"):
        mismatch=[dict(rows[0]),dict(rows[1])]
        mismatch[1][key]=4 if key=="horizon" else ("9"*64 if key in {"exogenous_suffix_contract_hash","future_policy_hash"} else "different")
        invalid.append(mismatch)
    for bad in invalid:
        with pytest.raises(ValueError,match="E_QUALIFICATION_FAIL"):module.validate(bad)


def test_exact_noop_v2_is_mandatory_in_evidence_and_launcher(tmp_path):
    bad=eligible_row();bad["gates"].pop("exact_noop_v2_qualified")
    with pytest.raises(ValueError,match="exact_noop_v2_qualified"):
        require_arm("ncr_certified_routing",write_ledger(tmp_path,[bad]))
    launcher=(Path(__file__).parents[3]/"experiments/7b_ideas/run_7b_idea.sh").read_text()
    assert "EXACT_NOOP_V2_MANIFEST" in launcher
    assert "validate_exact_noop_v2_manifest_20260819.py" in launcher
    assert "legacy replay is ineligible" in launcher
    assert "--require-shape-a-e" in launcher


def test_horizon_estimands_eh_allows_endogenous_divergence_but_not_ambiguous_suffix():
    module=_exact_noop_validator();rows=[_build_e_record(_e_proposal(),"commit",.2),
      _build_e_record(_e_proposal(),"retain",.7)]
    result=module.validate(rows)
    assert result["estimand_mode"]=="EH"
    assert result["estimand"]=="regime-conditional total execution effect of one update"
    assert not result["realized_trajectory_is_pair_key"]
    assert rows[0]["realized_trajectory_hash"]!=rows[1]["realized_trajectory_hash"]
    bad=[dict(rows[0],suffix_hash="8"*64),rows[1]]
    with pytest.raises(ValueError,match="LEGACY_SUFFIX_SEMANTICS_AMBIGUOUS"):module.validate(bad)


def test_vg_is_closed_loop_value_not_shape_a_execution_effect():
    module=_exact_noop_validator()
    rows=[build_vg_record(stable_id=f"e{i}",common_initial_state_hash="a"*64,
      fixed_policy_hash="b"*64,horizon=3,endpoint_version="f1-v1",endpoint_value=.5,
      realized_trajectory_hash=f"{i+10:064x}",future_hash_chain=["c"*64]) for i in range(4)]
    result=module.validate(rows)
    assert result["status"]=="VG_QUALIFIED_CLOSED_LOOP_VALUE" and result["closed_loop_policy_value"]
    assert not result["shape_a_execution_effect_qualified"]


def test_shape_a_horizon_primary_freeze_and_selection_failures():
    path=Path(__file__).parents[3]/"experiments/7b_ideas/analysis/validate_shapeA_horizon_primary_20260819.py"
    spec=importlib.util.spec_from_file_location("shape_horizon",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((Path(__file__).parents[3]/"experiments/7b_ideas/configs/shapeA_horizon_primary_freeze.json").read_text())
    result=module.validate(config)
    assert result["status"]=="SHAPEA_EH_PRIMARY_FROZEN" and not result["outcomes_read"]
    for key,value in (("estimand_mode","E0"),("endpoint","terminal_EM"),
      ("b0_candidate_descendants",True),("horizon_rows_increase_n",True),
      ("legacy_valid_n_96_rescue",True),("dstar_truncate_at_zero",True),
      ("legacy_max0_w_minus_q95",True),("mixed_tau_H_and_H_H_primary",True),
      ("endpoint_estimand_subset_selection_after_results",True),
      ("delete_negative_or_invalid_rows_after_results",True)):
        bad=dict(config);bad[key]=value
        with pytest.raises(ValueError,match="HORIZON_ENDPOINT_SELECTION_INVALID"):module.validate(bad)
    bad=dict(config);bad["f1_commit"]=.8
    with pytest.raises(ValueError,match="HORIZON_ENDPOINT_SELECTION_INVALID"):module.validate(bad)
    bad=dict(config);bad["b0_covariates"]=list(config["b0_covariates"])+["candidate_length"]
    with pytest.raises(ValueError,match="b0_postbranch"):module.validate(bad)
    launcher=(Path(__file__).parents[3]/"experiments/7b_ideas/run_7b_idea.sh").read_text()
    assert "SHAPEA_HORIZON_PRIMARY_MANIFEST" in launcher


def _w4_validator():
    path=Path(__file__).parents[3]/"experiments/7b_ideas/analysis/validate_counterfactual_gradient_witness_20260819.py"
    spec=importlib.util.spec_from_file_location("w4_validator",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _w4_manifest(n=20,*,coupling_ids=("u0",),credit=(.5,0.),task=(.5,0.),reg=(0.,0.),total=(.5,0.)):
    events=[]
    for i in range(n):
        candidate=f"{i+1000:064x}";pair=f"{i+2000:064x}"
        for coupling_index,coupling_id in enumerate(coupling_ids):
          events.append(capture_w4_event(stable_id=f"e{i}",group_id=f"g{i}",candidate_hash=candidate,
            pair_key_hash=pair,checkpoint_hash="a"*64,subspace_hash="b"*64,reader_coupling_id=coupling_id,
            y_commit=1.0+.1*coupling_index,y_retain=0.0,
            writer_token_score_gradients=[[1.0,0.0],[0.0,1.0]],writer_token_mask=[True,True],
            policy_controlled_token_kinds=["token","eos_or_stop"],credit_writer_gradient=list(credit),
            task_writer_gradient=list(task),regularizer_writer_gradient=list(reg),total_writer_gradient=list(total),
            loss_graph_hash="d"*64,actual_candidate_hash=candidate,actual_group_id=f"g{i}",
            actual_checkpoint_hash="a"*64,actual_subspace_hash="b"*64))
    multiple=len(coupling_ids)>1
    return {"schema_version":"counterfactual-gradient-witness-v4",
      "on_policy_same_checkpoint_candidate":True,"exact_noop_v2_qualified":True,
      "exact_noop_v2_manifest_hash":"c"*64,"noop_baseline_candidate_independent":True,
      "noop_rng_independent":True,"noop_cache_independent":True,"writer_token_mask_exact":True,
      "noop_coupling_frozen_before_candidate":True,"noop_coupling_exogenous_given_state":True,
      "factual_retain_shared_crn":True,
      "reader_seed_derivation":"pre_candidate_state_coupling_manifest",
      "rng_advance_candidate_length_dependent":False,"coupling_plan_frozen_before_first_tau":True,
      "seeds_added_after_first_tau":False,"seed_selected_by_sign":False,
      "reader_repeats_increase_independent_n":False,"candidate_clustered_inference":True,
      "coupling_mode":"prefrozen_multiple_independent_crn" if multiple else "single_exogenous_crn",
      "frozen_reader_coupling_ids":list(coupling_ids),"expected_couplings_per_candidate":len(coupling_ids),
      "validity_frozen_before_tau":True,
      "truncation_frozen_before_tau":True,"row_selection_frozen_before_tau":True,
      "tau_or_outcome_conditioned_selection":False,
      "shared_suffix_endpoint_frozen":True,"actual_group_reconstructable":True,
      "actual_bonus_reconstructable":True,"actual_logprob_reconstructable":True,
      "actual_clip_reconstructable":True,"actual_kl_reconstructable":True,
      "same_loss_graph_decomposition":True,"gradient_components_reconstructable":True,
      "credit_uses_actual_group_advantage":True,"credit_token_broadcast_exact":True,
      "credit_clip_disabled":True,"credit_regularizers_disabled":True,"gradient_additivity_atol":1e-8,
      "optimizer_steps":0,"new_rollouts":False,"material_effect_threshold":.1,
      "evidence_basis":["alignment","captured_signed_ratio","effect_weighted_mass"],"events":events}


def test_w4_capture_metrics_pilot_and_scientific_minimum():
    module=_w4_validator()
    result=module.validate(_w4_manifest(20))
    assert result["status"]=="W4_CAPTURE_QUALIFIED_ANALYSIS_ONLY"
    assert result["independent_material_events"]==20 and result["highest_claim_level"]=="W3"
    assert result["coupling_scope"]=="realized_coupling_only"
    assert "realized_coupling_credit_effect_weighted_opposing_mass" in result
    assert not result["candidate_stable_help_harm_authorized"]
    assert not result["w4_claim_authorized"] and not result["training_authorized"]
    assert module.validate(_w4_manifest(4))["status"]=="W4_PLUMBING_ONLY"


def test_w4_fail_closed_reference_reconstruction_and_forbidden_shortcuts():
    module=_w4_validator()
    for key,value in (("on_policy_same_checkpoint_candidate",False),("noop_baseline_candidate_independent",False),
      ("noop_rng_independent",False),("noop_cache_independent",False),("writer_token_mask_exact",False),
      ("noop_coupling_frozen_before_candidate",False),("noop_coupling_exogenous_given_state",False),
      ("rng_advance_candidate_length_dependent",True),("validity_frozen_before_tau",False),
      ("truncation_frozen_before_tau",False),("row_selection_frozen_before_tau",False),
      ("tau_or_outcome_conditioned_selection",True),
      ("actual_group_reconstructable",False),("actual_bonus_reconstructable",False),
      ("actual_logprob_reconstructable",False),("actual_clip_reconstructable",False),
      ("actual_kl_reconstructable",False),("gradient_components_reconstructable",False),
      ("same_loss_graph_decomposition",False),("seeds_added_after_first_tau",True),
      ("seed_selected_by_sign",True),("reader_repeats_increase_independent_n",True),
      ("optimizer_steps",1),("new_rollouts",True)):
        bad=_w4_manifest();bad[key]=value
        with pytest.raises(ValueError,match="W4_NO_GO.*highest_level=W3"):module.validate(bad)
    bad=_w4_manifest();bad["evidence_basis"]=["gradient_difference_norm"]
    with pytest.raises(ValueError,match="forbidden evidence basis"):module.validate(bad)
    bad=_w4_manifest();bad["events"][0]["actual_group_id"]="wrong"
    with pytest.raises(ValueError,match="candidate/group/checkpoint/subspace mismatch"):module.validate(bad)


def test_w4_v3_graph_decomposition_and_delivery_attribution():
    module=_w4_validator()
    regularized=_w4_manifest(4,credit=(1.,0.),task=(1.,0.),reg=(-2.,0.),total=(-1.,0.))
    result=module.validate(regularized)
    assert result["delivery_adjudication_counts"]=={"REGULARIZATION_TRADEOFF":4}
    assert all(row["credit_alignment"]>0 and row["total_alignment_sign"]<0 for row in result["event_metrics"])
    bottleneck=module.validate(_w4_manifest(4,credit=(1.,0.),task=(-1.,0.),reg=(0.,0.),total=(-1.,0.)))
    assert bottleneck["delivery_adjudication_counts"]=={"CLIP_OR_TRUST_REGION_DELIVERY_BOTTLENECK":4}
    bad=_w4_manifest(4);bad["events"][0]["total_writer_gradient"]=[2.,0.]
    bad["events"][0]["total_gradient_hash"]=vector_hash([2.,0.])
    with pytest.raises(ValueError,match=r"G_total != G_task \+ G_reg"):module.validate(bad)


def test_w4_v4_single_vs_prefrozen_multiple_coupling_reporting():
    module=_w4_validator()
    single=module.validate(_w4_manifest(4))
    assert single["aggregate_g_cf_mc_sample_valid"] and single["independent_candidates"]==4
    assert single["reader_coupling_rows"]==4 and not single["reader_repeats_increase_independent_n"]
    multiple=module.validate(_w4_manifest(4,coupling_ids=("u0","u1","u2")))
    assert multiple["independent_candidates"]==4 and multiple["reader_coupling_rows"]==12
    assert multiple["coupling_scope"]=="prefrozen_candidate_mean"
    assert all(row["within_candidate_tau_dispersion"] is not None and row["coupling_sign_stability"]==1 for row in multiple["event_metrics"])
    bad=_w4_manifest(4);bad["evidence_basis"].append("candidate_stable_help_harm")
    with pytest.raises(ValueError,match="single coupling cannot support"):module.validate(bad)


def test_csfgw_v2_score_function_identity_positive_and_three_negative_controls():
    path=Path(__file__).parents[3]/"experiments/7b_ideas/analysis/audit_counterfactual_score_function_identity_20260819.py"
    spec=importlib.util.spec_from_file_location("csfgw_identity",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.self_test()
    rows=[{"probability":.5,"commit_return":1.0,"noop_baseline":.25,"score_gradient":[.5],
      "selected":True,"policy_controlled_token_kinds":["token","eos_or_stop"],
      "score_mask_token_kinds":["token","eos_or_stop"]},
      {"probability":.5,"commit_return":0.0,"noop_baseline":.25,"score_gradient":[-.5],
      "selected":True,"policy_controlled_token_kinds":["token","eos_or_stop"],
      "score_mask_token_kinds":["token","eos_or_stop"]}]
    manifest={**module.EXPECTED,"direct_commit_return_gradient":[.25],"candidates":rows}
    assert module.audit(manifest)["status"]=="CSFGW_IDENTITY_V2_PASS"
    bad=json.loads(json.dumps(manifest));bad["reader_seed_derivation"]="hash(candidate)"
    with pytest.raises(ValueError,match="W4_NO_GO"):module.audit(bad)


def test_w4_launcher_is_dual_gated_and_never_runs_pilot():
    launcher=(Path(__file__).parents[3]/"experiments/7b_ideas/run_7b_idea.sh").read_text()
    for marker in ("W4_GRADIENT_PILOT_REQUEST","W4_OPTIMIZER_STEPS","IDEA_EVIDENCE_LEDGER",
                   "MECHANISM_EXTENSION_DECISION","W4_NO_GO","highest_level=W3"):
        assert marker in launcher


def test_adaptive_stop_rule_v4_and_launcher_guard():
    path=Path(__file__).parents[3]/"experiments/7b_ideas/analysis/validate_adaptive_stop_rule_v4_20260819.py"
    spec=importlib.util.spec_from_file_location("stop_v4",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((Path(__file__).parents[3]/"experiments/7b_ideas/configs/adaptive_stop_rule_v4.json").read_text())
    result=module.validate(config)
    assert result["terminal_step"]==200 and not result["controls_continuation"]
    assert not result["step_400_authorized"] and not result["best_checkpoint_selection"]
    for key,value in (("controls_continuation",True),("terminal_step",400),
      ("select_best_checkpoint_for_confirmatory",True),("step_400_automatic",True)):
        bad=dict(config);bad[key]=value
        with pytest.raises(ValueError,match="STOP_RULE_V4_FAIL_CLOSED"):module.validate(bad)
    launcher=(Path(__file__).parents[3]/"experiments/7b_ideas/run_7b_idea.sh").read_text()
    assert "STOP_RULE_MANIFEST" in launcher and "validate_adaptive_stop_rule_v4_20260819.py" in launcher

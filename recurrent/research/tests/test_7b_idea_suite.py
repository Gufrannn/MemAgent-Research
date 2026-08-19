import json
from pathlib import Path

import pytest
import torch
import importlib.util

from recurrent.research.cerc_native_credit import validate_native_credit
from recurrent.research.idea_admissibility import ELIGIBLE, NO_METHOD, PENDING, adjudicate, require_arm
from recurrent.research.ncr_certified_routing import generic_auxiliary, route_ncr
from recurrent.research.typed_boundary_prompt import ARMS, build_prompts


def eligible_row():
    gates = {key: True for key in (
        "shape_a_candidate_free_t0", "exact_linked_same_write_key", "exact_tie_coverage",
        "frozen_readout", "writer_only", "non_tie_bitwise_unchanged", "gradient_safety",
        "beats_qa_only", "beats_generic_qa", "beats_generic_judge",
        "beats_information_matched_raw_judge", "beats_generic_tie_rescue")}
    return {"event_id": "e1", "timestamp": "2026-08-19T00:00:00Z", "candidate": "ncr_certified_routing",
            "status": "eligible", "gates": gates,
            "shape_a": {"t0_formula": "P2_raw^T0 vs P2_raw^T0+D_pre^audit", "t1_leaks_into_t0": False},
            "shape_a_contract": {"independent_unit": "stable_example_id", "max_independent_n": 128,
              "primary_representation": "paired_tau", "stacked_role": "implementation_consistency_audit_only",
              "count_paired_and_stacked_as_one": True, "select_more_significant_representation": False,
              "b_raw": "tau~P2_raw_T0", "b_struct": "tau~P2_raw_T0+D_star", "d_star_dimensions": 1,
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
    rows = [{"stable_example_id": f"e{i}", "d_star": float(i), "y_noop": 1 + i, "y_factual": 2 + 3*i} for i in range(8)]
    result = module.audit(rows)
    assert result["algebraically_equivalent"] and not result["claim_authorized"] and not result["p_values_emitted"]
    with pytest.raises(ValueError, match="unique"): module.audit(rows + [rows[0]])
    with pytest.raises(ValueError, match=r"\[2,128\]"): module.audit(rows * 17)
    with pytest.raises(ValueError, match="rank deficient"):
        module.audit([{**row, "d_star": 1} for row in rows])


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

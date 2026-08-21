from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
import uuid
from pathlib import Path

import pytest

from recurrent.research.commit_retain_capture import (
    build_pair_record,
    canonical_sha256,
)
from recurrent.research.paired_effect_credit import (
    CAPTURE32_COUNT,
    CAPTURE32_SELECTED_POSITIONS,
    CAPTURE4_PILOT_POSITIONS,
    FEATURE_SCHEMA,
    _compute_centered_trajectory_bonuses,
    audit_writer_credit_routing,
    build_prebranch_candidate_payload,
    build_crossfit_bundle_from_observations,
    extract_outcome_hidden_features,
    extract_prebranch_features,
    order_and_validate_capture32_pairs,
    paired_outcome,
    validate_capture32_authority_binding,
    validate_capture32_preregistration,
    validate_s128_authority,
    validate_prebranch_candidate_payload,
    validate_crossfit_bundle,
)
from tools.h20.audit_qwen25_7b_paired_effect_candidate import (
    CAPTURE32_EXECUTION_CODE_OBJECTS,
    CAPTURE32_PREREG_REL,
    Capture32AttritionError,
    MANIFEST_REL,
    LOADED_PIPELINE_PYTHON_OBJECTS,
    PIPELINE_CODE_OBJECTS,
    _authenticate_capture_git,
    _assert_loaded_repo_module_origins,
    _capture32_artifact_state,
    _validate_capture32_runtime_bindings,
    _validate_capture32_structural_completeness,
    _readiness,
    _write_outputs,
    build_report,
    crossfit_diagnostics,
    load_capture32_preregistration,
    load_s128_authority,
    load_manifest,
    validate_report_shape,
)
from tools.h20.preflight_qwen25_7b_commit_retain import (
    _code_objects,
    load_manifest as load_capture_manifest,
)


REPO = Path(__file__).resolve().parents[2]
CAPTURE_COMMIT = "85ba6d3b03874978ed5d9713d8a628ce37f0c478"


def _observations(count: int = 8) -> list[dict]:
    result = []
    for index in range(count):
        vector = [
            (index + 1) * (column + 1) / 100.0
            + (0.01 if column == (index % len(FEATURE_SCHEMA)) else 0.0)
            for column in range(len(FEATURE_SCHEMA))
        ]
        target = ((index % 4) - 1.5) / 4.0 + vector[index % len(vector)]
        result.append({
            "stable_example_id": f"stable-example-{index:03d}",
            "stable_write_id": canonical_sha256(["write", index]),
            "pair_id": canonical_sha256(["pair", index]),
            "feature_schema": list(FEATURE_SCHEMA),
            "feature_schema_sha256": canonical_sha256(list(FEATURE_SCHEMA)),
            "feature_input_sha256": canonical_sha256(["input", index]),
            "feature_vector": vector,
            "feature_vector_sha256": canonical_sha256(vector),
            "outcome_hidden_for_scored_row": True,
            "forbidden_input_fields_absent": [
                "arms", "ground_truth", "terminal_answer", "reward", "outcome"
            ],
            "target_name": "token_f1_commit_minus_retain",
            "commit_token_f1": 0.5 + target / 2,
            "retain_token_f1": 0.5 - target / 2,
            "paired_effect_target": target,
            "commit_exact_match": 0.0,
            "retain_exact_match": 0.0,
            "paired_exact_match_difference": 0.0,
        })
    return result


def _candidate_environment(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMAGENT_PAIRED_EFFECT_WORK_ROOT": str(tmp_path.resolve()),
        "MEMAGENT_PAIRED_EFFECT_REPO_DIR": str(REPO.resolve()),
        "MEMAGENT_PAIRED_EFFECT_EXPECTED_COMMIT": CAPTURE_COMMIT,
        "MEMAGENT_PAIRED_EFFECT_RUN_ID": "paired_canary",
        "MEMAGENT_PAIRED_EFFECT_CAPTURE_RUN_ID": "capture_gpu45",
    }


def _candidate_manifest(tmp_path: Path) -> dict:
    return load_manifest(REPO / MANIFEST_REL, _candidate_environment(tmp_path))


def _capture32_preregistration() -> dict:
    return json.loads((REPO / CAPTURE32_PREREG_REL).read_text(encoding="utf-8"))


def _capture32_authority() -> dict:
    return json.loads((
        REPO / "manifests/h20/qwen25_7b_paired_effect_s128_authority.json"
    ).read_text(encoding="utf-8"))


def _rehash_preregistration(value: dict) -> dict:
    for row in value.get("selected_inventory", []):
        unsigned = {key: child for key, child in row.items() if key != "row_sha256"}
        row["row_sha256"] = canonical_sha256(unsigned)
    ranking = value.get("selection", {}).get("full_population_ranking")
    if isinstance(ranking, dict):
        value["selection"]["full_population_ranking_sha256"] = canonical_sha256(ranking)
    membership = value.get("folds", {}).get("membership")
    if isinstance(membership, dict):
        value["folds"]["membership_sha256"] = canonical_sha256(membership)
    unsigned = {key: child for key, child in value.items()
                if key != "preregistration_sha256"}
    value["preregistration_sha256"] = canonical_sha256(unsigned)
    return value


def _passing_diagnostics() -> dict:
    return {
        "stable_example_count": 32,
        "nontrivial_effect_epsilon": 0.01,
        "nontrivial_effect_count": 12,
        "effect_bin_precision": 0.000001,
        "distinct_effect_bin_count": 6,
        "mean_absolute_effect": 0.08,
        "target_variance": 0.02,
        "crossfit_mse_improvement_fraction": 0.10,
        "crossfit_pearson_correlation": 0.25,
        "folds_with_positive_mse_improvement": 4,
        "per_fold": {
            str(fold): {"heldout_count": 8, "fit_count": 24}
            for fold in range(4)
        },
        "nonfinite_score_count": 0,
    }


def _capture_manifest(candidate: dict) -> dict:
    return load_capture_manifest(
        REPO / candidate["source_capture"]["manifest"],
        {
            "MEMAGENT_COMMIT_RETAIN_WORK_ROOT": candidate["work_root"],
            "MEMAGENT_COMMIT_RETAIN_REPO_DIR": candidate["repo_dir"],
            "MEMAGENT_COMMIT_RETAIN_EXPECTED_COMMIT": CAPTURE_COMMIT,
            "MEMAGENT_COMMIT_RETAIN_RUN_ID": candidate["capture_run_id"],
        },
    )


def _bind_test_pipeline(report: dict) -> None:
    hashes = {
        relative: hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
        for relative in PIPELINE_CODE_OBJECTS
    }
    report["pipeline"]["code_sha256"] = hashes
    report["pipeline"]["code_combined_sha256"] = canonical_sha256(hashes)


def _assert_schema_valid(report: dict) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((REPO / "paired_effect_admissibility_report.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(report)


def test_capture32_literal_preregistration_is_exact_disjoint_and_four_by_eight() -> None:
    prereg = validate_capture32_preregistration(_capture32_preregistration())
    assert prereg["selection"]["selected_sorted_positions"] == list(
        CAPTURE32_SELECTED_POSITIONS
    )
    assert set(CAPTURE32_SELECTED_POSITIONS).isdisjoint(CAPTURE4_PILOT_POSITIONS)
    assert len(prereg["selected_inventory"]) == CAPTURE32_COUNT
    assert [len(prereg["folds"]["membership"][str(fold)]) for fold in range(4)] \
        == [8, 8, 8, 8]
    assert len({row["stable_example_id"] for row in prereg["selected_inventory"]}) == 32
    assert prereg["attrition"]["capture4_may_fill_missing"] is False
    assert prereg["claim_boundary"]["training_authorized"] is False


def test_every_loaded_repo_helper_is_in_authenticated_code_closure() -> None:
    assert set(LOADED_PIPELINE_PYTHON_OBJECTS).issubset(PIPELINE_CODE_OBJECTS)
    required_helpers = {
        "recurrent/research/gate_a_execution.py",
        "recurrent/research/s128_hotpot_metrics.py",
        "tools/h20/preflight_qwen25_7b_commit_retain.py",
        "tools/h20/preflight_qwen25_7b_serialization_credit.py",
        "tools/h20/preflight_qwen25_7b_s128_it.py",
        "tools/h20/preflight_qwen25_7b_stable_i4x2.py",
    }
    assert required_helpers.issubset(PIPELINE_CODE_OBJECTS)
    assert {
        "tools/h20/preflight_qwen25_7b_serialization_credit.py",
        "tools/h20/preflight_qwen25_7b_s128_it.py",
        "tools/h20/preflight_qwen25_7b_stable_i4x2.py",
    }.issubset(CAPTURE32_EXECUTION_CODE_OBJECTS)


def test_capture32_stable_identity_tamper_fails_after_all_local_hashes_are_rebuilt() -> None:
    prereg = _capture32_preregistration()
    row = prereg["selected_inventory"][0]
    position = row["prompt_length_sorted_position"]
    row["stable_example_id"] = "f" * 64
    prereg["selection"]["full_population_ranking"]["stable_example_ids"][
        position
    ] = "f" * 64
    _rehash_preregistration(prereg)
    with pytest.raises(ValueError, match="ranking digest|stable IDs"):
        validate_capture32_preregistration(prereg)


def test_capture32_fold_imbalance_fails_even_when_membership_is_rehashed() -> None:
    prereg = _capture32_preregistration()
    moved = prereg["folds"]["membership"]["0"].pop()
    prereg["folds"]["membership"]["1"].append(moved)
    prereg["folds"]["membership"]["1"].sort()
    _rehash_preregistration(prereg)
    with pytest.raises(ValueError, match="4x8 fold contract"):
        validate_capture32_preregistration(prereg)


def test_capture32_missing_or_replacement_row_cannot_be_rehashed_into_validity() -> None:
    missing = _capture32_preregistration()
    missing["selected_inventory"].pop()
    _rehash_preregistration(missing)
    with pytest.raises(ValueError, match="exactly 32"):
        validate_capture32_preregistration(missing)

    replacement = _capture32_preregistration()
    replacement["selected_inventory"][0] = copy.deepcopy(
        replacement["selected_inventory"][1]
    )
    _rehash_preregistration(replacement)
    with pytest.raises(ValueError, match="stratum drifted"):
        validate_capture32_preregistration(replacement)


def test_capture32_pair_inventory_rejects_31_before_pair_validation() -> None:
    with pytest.raises(ValueError, match="attrition 31 != 32"):
        order_and_validate_capture32_pairs(
            [{} for _ in range(31)], _capture32_preregistration()
        )


def test_capture32_full_population_ranking_tamper_fails_closed() -> None:
    prereg = _capture32_preregistration()
    lengths = prereg["selection"]["full_population_ranking"][
        "writer_turn0_prompt_token_lengths"
    ]
    lengths[0], lengths[-1] = lengths[-1], lengths[0]
    _rehash_preregistration(prereg)
    with pytest.raises(ValueError, match="ranking digest|ranking order"):
        validate_capture32_preregistration(prereg)


def test_capture32_authority_binds_all_128_real_stable_i_rows(tmp_path: Path) -> None:
    manifest = _candidate_manifest(tmp_path)
    prereg = load_capture32_preregistration(manifest)
    authority = load_s128_authority(manifest)
    assert len(validate_s128_authority(authority)["identity_payload"]["rows"]) == 128
    assert validate_capture32_authority_binding(prereg, authority) == prereg


def test_capture32_authority_cannot_be_rewritten_and_self_rehashed() -> None:
    authority = _capture32_authority()
    authority["identity_payload"]["rows"][0]["source_question_hash"] = "f" * 64
    authority["eval_manifest_hash"] = canonical_sha256(authority["identity_payload"])
    unsigned = {key: child for key, child in authority.items()
                if key != "authority_sha256"}
    authority["authority_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="authority binding"):
        validate_s128_authority(authority)


def test_capture32_trajectory_seed_formula_rejects_full_local_rehash() -> None:
    prereg = _capture32_preregistration()
    row = prereg["selected_inventory"][0]
    row["trajectory_seed"] += 1
    from recurrent.research.commit_retain_capture import stable_capture_ids
    from recurrent.research.trajectory_seeding import derive_turn_request_seeds

    identity = {
        field: prereg["source"]["eval_manifest_hash"] if field == "eval_manifest_hash"
        else row[field]
        for field in (
            "example_id", "semantic_dataset_index", "source_order_index",
            "raw_row_position", "production_effective_position", "eval_manifest_hash",
            "source_question_hash", "source_context_hash", "ground_truth_hash",
        )
    }
    row.update(stable_capture_ids(
        identity,
        trajectory_seed=row["trajectory_seed"],
        writer_turn=row["intervention_writer_turn"],
    ))
    row["writer_turn0_request_seed"] = derive_turn_request_seeds(
        [row["trajectory_seed"]], [0], 0
    )[0]
    position = row["prompt_length_sorted_position"]
    prereg["selection"]["full_population_ranking"]["stable_example_ids"][
        position
    ] = row["stable_example_id"]
    _rehash_preregistration(prereg)
    with pytest.raises(ValueError, match="trajectory seed drifted"):
        validate_capture32_preregistration(prereg)


def test_capture32_writer_schedule_formula_rejects_full_local_rehash() -> None:
    prereg = _capture32_preregistration()
    prereg["selected_inventory"][0]["total_writer_turns"] += 1
    _rehash_preregistration(prereg)
    with pytest.raises(ValueError, match="lacks prefix/future"):
        validate_capture32_preregistration(prereg)


def test_capture32_selection_and_folds_are_outcome_blind_by_construction() -> None:
    prereg = validate_capture32_preregistration(_capture32_preregistration())
    forbidden = set(prereg["selection"]["forbidden_selection_inputs"])
    assert {
        "arm_outcome", "reader_answer", "reward", "token_f1", "exact_match",
        "candidate_output", "actual_cost", "existing_score", "runtime_uuid",
        "pair_id", "ground_truth", "ground_truth_hash",
    }.issubset(forbidden)
    assert prereg["selection"]["allowed_selection_inputs"] == [
        "writer_turn0_prompt_token_length", "source_order_index"
    ]
    assert prereg["folds"]["frozen_before_first_generate"] is True


def test_capture32_scorer_cannot_be_retuned_after_outcomes_and_rehashed() -> None:
    prereg = _capture32_preregistration()
    prereg["scorer"]["ridge"] = 0.01
    _rehash_preregistration(prereg)
    with pytest.raises(ValueError, match="scorer preregistration drifted"):
        validate_capture32_preregistration(prereg)


def test_stable_example_grouped_crossfit_excludes_every_scored_example() -> None:
    observations = _observations()
    bundle = build_crossfit_bundle_from_observations(
        observations, fold_count=4, ridge=1.0
    )
    validate_crossfit_bundle(bundle, observations)
    assert len(bundle["scores"]) == len(observations)
    for score in bundle["scores"]:
        model = bundle["models"][str(score["score_fold"])]
        assert score["stable_example_id"] not in model["fit_stable_example_ids"]
        assert score["fit_membership_sha256"] == model["fit_membership_sha256"]
        assert score["outcome_hidden_for_scored_row"] is True
    assert bundle["deployment_model"]["deployment_use_authorized"] is False
    assert bundle["training_authorized"] is False
    assert bundle["method_selected"] is False


def test_crossfit_is_invariant_to_observation_serialization_order() -> None:
    observations = _observations(32)
    forward = build_crossfit_bundle_from_observations(
        observations, fold_count=4, ridge=1.0
    )
    reverse = build_crossfit_bundle_from_observations(
        list(reversed(observations)), fold_count=4, ridge=1.0
    )
    assert forward == reverse


def test_heldout_outcome_cannot_change_its_model_or_score() -> None:
    observations = _observations(32)
    baseline = build_crossfit_bundle_from_observations(
        observations, fold_count=4, ridge=1.0
    )
    stable_id = observations[0]["stable_example_id"]
    baseline_score = next(
        row for row in baseline["scores"] if row["stable_example_id"] == stable_id
    )
    score_fold = str(baseline_score["score_fold"])
    mutated = copy.deepcopy(observations)
    mutated[0]["paired_effect_target"] += 1000.0
    mutated_bundle = build_crossfit_bundle_from_observations(
        mutated, fold_count=4, ridge=1.0
    )
    mutated_score = next(
        row for row in mutated_bundle["scores"] if row["stable_example_id"] == stable_id
    )
    assert baseline["models"][score_fold] == mutated_bundle["models"][score_fold]
    assert baseline_score["paired_effect_score"] == mutated_score["paired_effect_score"]


def test_heldout_features_change_only_prediction_not_fit_standardization() -> None:
    observations = _observations(32)
    baseline = build_crossfit_bundle_from_observations(
        observations, fold_count=4, ridge=1.0
    )
    stable_id = observations[0]["stable_example_id"]
    baseline_score = next(
        row for row in baseline["scores"] if row["stable_example_id"] == stable_id
    )
    score_fold = str(baseline_score["score_fold"])
    mutated = copy.deepcopy(observations)
    mutated[0]["feature_vector"][0] += 1000.0
    mutated[0]["feature_vector_sha256"] = canonical_sha256(
        mutated[0]["feature_vector"]
    )
    mutated_bundle = build_crossfit_bundle_from_observations(
        mutated, fold_count=4, ridge=1.0
    )
    mutated_score = next(
        row for row in mutated_bundle["scores"] if row["stable_example_id"] == stable_id
    )
    assert baseline["models"][score_fold] == mutated_bundle["models"][score_fold]
    assert baseline_score["paired_effect_score"] != mutated_score["paired_effect_score"]


def test_handfilled_or_rehashed_score_cannot_pass_recomputation() -> None:
    observations = _observations()
    bundle = build_crossfit_bundle_from_observations(
        observations, fold_count=4, ridge=1.0
    )
    forged = copy.deepcopy(bundle)
    forged["scores"][0]["paired_effect_score"] += 9.0
    unsigned = {key: value for key, value in forged.items() if key != "bundle_sha256"}
    forged["bundle_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="does not reproduce"):
        validate_crossfit_bundle(forged, observations)


def test_crossfit_rejects_duplicate_stable_example_membership() -> None:
    observations = _observations()
    observations[-1]["stable_example_id"] = observations[0]["stable_example_id"]
    with pytest.raises(ValueError, match="unique examples"):
        build_crossfit_bundle_from_observations(
            observations, fold_count=4, ridge=1.0
        )


def test_crossfit_rejects_reassigned_preregistered_fold() -> None:
    observations = _observations(32)
    baseline = build_crossfit_bundle_from_observations(
        observations, fold_count=4, ridge=1.0
    )
    assignments = copy.deepcopy(baseline["fold_assignments"])
    stable_id = sorted(assignments)[0]
    assignments[stable_id] = (assignments[stable_id] + 1) % 4
    with pytest.raises(ValueError, match="differs from preregistration"):
        build_crossfit_bundle_from_observations(
            observations, fold_count=4, ridge=1.0,
            expected_fold_assignments=assignments,
        )


def test_feature_extractor_has_no_arm_outcome_or_runtime_uuid_input() -> None:
    from tests.h20.test_commit_retain_capture import pair_payload

    pair = build_pair_record(pair_payload(index=7))
    before = extract_outcome_hidden_features(pair, validate=False)
    mutated = copy.deepcopy(pair)
    mutated["arms"]["COMMIT"]["final_reader"]["outcome"]["token_f1"] = -999.0
    mutated["arms"]["RETAIN"]["final_reader"]["output_text"] = "future leakage"
    mutated["execution"]["process_instance_uuid"] = str(uuid.uuid4())
    after = extract_outcome_hidden_features(mutated, validate=False)
    assert before == after
    assert "arms" not in before
    assert "process_instance_uuid" not in before


def test_offline_prebranch_feature_payload_is_outcome_free_and_recomputes_stable_identity() -> None:
    from tests.h20.test_commit_retain_capture import pair_payload

    pair = build_pair_record(pair_payload(index=7))
    payload = build_prebranch_candidate_payload(pair)
    assert "arms" not in payload
    assert "ground_truth" not in payload
    assert "pair_id" not in payload
    features = extract_prebranch_features(payload)
    assert features["outcome_hidden_for_scored_row"] is True
    forged_payload = copy.deepcopy(payload)
    forged_payload["outcome"] = {"token_f1": 1.0}
    with pytest.raises(ValueError, match="extra/missing fields"):
        extract_prebranch_features(forged_payload)
    forged_identity = copy.deepcopy(payload)
    forged_identity["stable_example_id"] = "f" * 64
    unsigned = {key: value for key, value in forged_identity.items()
                if key != "prebranch_payload_sha256"}
    forged_identity["prebranch_payload_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="stable identity does not reproduce"):
        validate_prebranch_candidate_payload(forged_identity)


def test_paired_target_is_recomputed_from_validated_arm_outcomes() -> None:
    from tests.h20.test_commit_retain_capture import pair_payload

    pair = build_pair_record(pair_payload(index=7))
    outcome = paired_outcome(pair)
    assert outcome["paired_effect_target"] == pytest.approx(
        outcome["commit_token_f1"] - outcome["retain_token_f1"]
    )
    forged = copy.deepcopy(pair)
    forged["arms"]["COMMIT"]["final_reader"]["outcome"]["token_f1"] = 123.0
    with pytest.raises(ValueError, match="stale evidence|non-canonical"):
        paired_outcome(forged)


def test_centered_credit_protects_nonties_ineligible_and_all_correct_groups() -> None:
    tie = canonical_sha256(["group", "tie"])
    nontie = canonical_sha256(["group", "nontie"])
    solved = canonical_sha256(["group", "solved"])
    bonuses, routed = _compute_centered_trajectory_bonuses(
        scores=[1.0, 3.0, 5.0, 7.0, 10.0, 12.0],
        qa_rewards=[0.5, 0.5, 0.0, 1.0, 1.0, 1.0],
        stable_group_ids=[tie, tie, nontie, nontie, solved, solved],
        eligible=[True, True, True, True, True, True],
        exact_correct=[False, False, False, False, True, True],
        lambda_=0.2,
    )
    assert bonuses == pytest.approx([-0.2, 0.2, 0.0, 0.0, 0.0, 0.0])
    assert routed == [True, True, False, False, False, False]
    group = canonical_sha256(["group", "q"])
    ineligible, routed = _compute_centered_trajectory_bonuses(
        scores=[1.0, 2.0], qa_rewards=[0.0, 0.0], stable_group_ids=[group, group],
        eligible=[True, False], exact_correct=[False, False], lambda_=1.0,
    )
    assert ineligible == [0.0, 0.0]
    assert routed == [False, False]


def test_writer_token_trajectory_total_and_bitwise_protection() -> None:
    torch = pytest.importorskip("torch")
    trajectory_base = torch.tensor([1.0, 2.0, -0.0, 4.0], dtype=torch.float32)
    sample_index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    final_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    response_mask = torch.tensor([
        [1, 1, 0, 0], [1, 0, 0, 0],
        [1, 1, 1, 0], [1, 1, 0, 0],
        [1, 0, 0, 0], [1, 0, 0, 0],
        [1, 1, 0, 0], [1, 0, 0, 0],
    ], dtype=torch.float32)
    baseline = trajectory_base[sample_index].unsqueeze(-1).expand_as(response_mask) * response_mask
    tie = canonical_sha256(["group", "tie"])
    nontie = canonical_sha256(["group", "nontie"])
    result, audit = audit_writer_credit_routing(
        trajectory_qa_advantage=trajectory_base,
        qa_reward=torch.tensor([0.5, 0.5, 0.0, 1.0]),
        stable_group_ids=[tie, tie, nontie, nontie],
        diagnostic_scores=torch.tensor([1.0, 3.0, 5.0, 7.0]),
        sample_index=sample_index,
        response_mask=response_mask,
        final_mask=final_mask,
        eligible=torch.tensor([True, True, True, True]),
        exact_correct=torch.tensor([False, False, False, False]),
        lambda_=0.2,
    )
    assert audit[0]["delivered_writer_token_bonus"] == pytest.approx(-0.2)
    assert audit[1]["delivered_writer_token_bonus"] == pytest.approx(0.2)
    assert audit[0]["writer_token_count"] == 2
    assert audit[1]["writer_token_count"] == 3
    assert torch.equal(result[final_mask].view(torch.int32), baseline[final_mask].view(torch.int32))
    protected = (sample_index == 2) | (sample_index == 3)
    assert torch.equal(result[protected].view(torch.int32), baseline[protected].view(torch.int32))


def test_routed_trajectory_without_writer_tokens_fails_clearly() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="no writer token"):
        audit_writer_credit_routing(
            trajectory_qa_advantage=torch.tensor([0.0, 0.0]),
            qa_reward=torch.tensor([0.0, 0.0]),
            stable_group_ids=[canonical_sha256(["q"]), canonical_sha256(["q"])],
            diagnostic_scores=torch.tensor([0.0, 1.0]),
            sample_index=torch.tensor([0, 1]),
            response_mask=torch.tensor([[1.0], [1.0]]),
            final_mask=torch.tensor([True, True]),
            eligible=torch.tensor([True, True]),
            exact_correct=torch.tensor([False, False]),
            lambda_=1.0,
        )


def test_writer_credit_rejects_malformed_trajectory_metadata_before_routing() -> None:
    torch = pytest.importorskip("torch")
    common = {
        "trajectory_qa_advantage": torch.tensor([0.0, 0.0]),
        "qa_reward": torch.tensor([0.0, 0.0]),
        "stable_group_ids": [canonical_sha256(["q"]), canonical_sha256(["q"])],
        "diagnostic_scores": torch.tensor([0.0, 1.0]),
        "sample_index": torch.tensor([0, 1]),
        "response_mask": torch.tensor([[1.0], [1.0]]),
        "final_mask": torch.tensor([True, True]),
        "eligible": torch.tensor([True, True]),
        "exact_correct": torch.tensor([False, False]),
        "lambda_": 1.0,
    }
    wrong_length = dict(common)
    wrong_length["qa_reward"] = torch.tensor([0.0])
    with pytest.raises(ValueError, match="trajectory metadata length mismatch"):
        audit_writer_credit_routing(**wrong_length)
    wrong_mask_dtype = dict(common)
    wrong_mask_dtype["eligible"] = torch.tensor([1.0, 1.0])
    with pytest.raises(ValueError, match="eligibility masks must be bool"):
        audit_writer_credit_routing(**wrong_mask_dtype)
    wrong_score_dtype = dict(common)
    wrong_score_dtype["diagnostic_scores"] = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="values/scores must be floating point"):
        audit_writer_credit_routing(**wrong_score_dtype)


def test_missing_capture_is_pending_not_pass(tmp_path: Path) -> None:
    manifest = _candidate_manifest(tmp_path)
    report, bundle = build_report(manifest, verify_pipeline_git=False)
    _bind_test_pipeline(report)
    validate_report_shape(report)
    _assert_schema_valid(report)
    assert report["status"] == "PENDING"
    assert report["decision"] == "PAIRED_EFFECT_CAPTURE_PENDING"
    assert report["training_authorized"] is False
    assert report["method_selected"] is False
    assert bundle is None


def test_capture32_attrition_state_is_pending_only_before_p0(tmp_path: Path) -> None:
    manifest = _candidate_manifest(tmp_path)
    state, paths = _capture32_artifact_state(manifest)
    assert state == "NOT_STARTED"

    paths["capture_ledger"].parent.mkdir(parents=True, exist_ok=True)
    paths["capture_ledger"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(Capture32AttritionError, match="without.*P0"):
        _capture32_artifact_state(manifest)


def test_capture32_p0_commitment_makes_every_missing_artifact_a_failure(
    tmp_path: Path,
) -> None:
    manifest = _candidate_manifest(tmp_path)
    _, paths = _capture32_artifact_state(manifest)
    paths["p0_certificate"].parent.mkdir(parents=True, exist_ok=True)
    paths["p0_certificate"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(Capture32AttritionError, match="attrition after P0"):
        _capture32_artifact_state(manifest)
    report, bundle = build_report(manifest, verify_pipeline_git=False)
    _bind_test_pipeline(report)
    validate_report_shape(report)
    _assert_schema_valid(report)
    assert report["status"] == "FAIL"
    assert report["decision"] == "PAIRED_EFFECT_NO_GO:CAPTURE32_ATTRITION"
    assert report["gates"]["capture32_inventory"]["status"] == "FAIL"
    assert bundle is None


def _capture32_binding_fixture(*, pair_gpus: list[int] | None = None) -> dict:
    prereg = validate_capture32_preregistration(_capture32_preregistration())
    pair_gpus = [4, 5] if pair_gpus is None else pair_gpus
    visible = ",".join(str(item) for item in pair_gpus)
    locks = [
        {
            "path": f"/work/locks/memagent_h20_gpu_{item}.lock",
            "fd": 180 + offset,
            "device": 1,
            "inode": 100 + offset,
            "owner_uid": 1000,
        }
        for offset, item in enumerate(pair_gpus)
    ]
    lock_sha = canonical_sha256({
        "physical_gpu_indices": pair_gpus,
        "locks": [
            {key: item[key] for key in ("path", "device", "inode", "owner_uid")}
            for item in locks
        ],
    })
    pair = {
        "writer_checkpoint_sha256": "1" * 64,
        "reader_checkpoint_sha256": "1" * 64,
        "writer_prompt_template_sha256": "2" * 64,
        "reader_prompt_template_sha256": "3" * 64,
        "writer_decode": {"temperature": 1.0},
        "reader_decode": {"temperature": 0.0},
        "physical_gpu_whitelist": pair_gpus,
        "visible_devices": visible,
        "physical_gpu_identity": [
            {
                "physical_index": item,
                "uuid": f"GPU-{item}",
                "pci_bus_id": f"00000000:{item:02X}:00.0",
                "name": "NVIDIA H20",
                "compute_mode": "Default",
                "mig_mode": "Disabled",
            }
            for item in pair_gpus
        ],
        "engine_config_sha256": "4" * 64,
        "worker_multiproc_method": "spawn",
        "vllm_observed_worker_multiproc_method": "spawn",
        "multiprocessing_context_method": "spawn",
        "parent_cuda_initialization_policy": "record_observed_spawn_required",
        "global_generate_call_count": 353,
        "eos_token_id": 151645,
        "gpu_lock_binding_sha256": lock_sha,
    }
    resolved = {
        "run_id": "capture32_test",
        "git_commit": "5" * 40,
        "physical_gpu_whitelist": pair_gpus,
        "visible_devices": visible,
        "execution_code_combined_sha256": "6" * 64,
        "gpu_lock_binding": {
            "schema": "memagent.capture32.lock-holder-receipt.v1",
            "physical_gpu_indices": pair_gpus,
            "locks": locks,
            "gpu_lock_binding_sha256": lock_sha,
        },
        "gpu_lock_binding_sha256": lock_sha,
        "expected_pair_binding": pair,
    }
    pair_sha = canonical_sha256(pair)
    execution = {
        "schema": "memagent.commit-retain.capture32-execution-binding.v1",
        "run_id": resolved["run_id"],
        "git_commit": resolved["git_commit"],
        "preregistration_sha256": prereg["preregistration_sha256"],
        "eval_manifest_hash": prereg["source"]["eval_manifest_hash"],
        "selected_inventory_sha256": prereg["inventory"]["selected_inventory_sha256"],
        "fold_membership_sha256": prereg["folds"]["membership_sha256"],
        "execution_code_combined_sha256": resolved["execution_code_combined_sha256"],
        "expected_pair_binding_sha256": pair_sha,
        "physical_gpu_whitelist": pair_gpus,
        "visible_devices": visible,
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
        "rollout_backend": "strict_vllm_0.8.2",
    }
    runtime = {
        "schema": "memagent.commit-retain.capture32-runtime-binding.v1",
        "run_id": resolved["run_id"],
        "git_commit": resolved["git_commit"],
        "expected_pair_binding_sha256": pair_sha,
        "physical_gpu_whitelist": pair_gpus,
        "visible_devices": visible,
        "physical_gpu_identity": pair["physical_gpu_identity"],
        "rollout_backend": "strict_vllm_0.8.2",
        "engine_config_sha256": pair["engine_config_sha256"],
        "worker_multiproc_method": "spawn",
        "vllm_observed_worker_multiproc_method": "spawn",
        "multiprocessing_context_method": "spawn",
        "parent_cuda_initialization_policy": "record_observed_spawn_required",
        "writer_checkpoint_sha256": pair["writer_checkpoint_sha256"],
        "reader_checkpoint_sha256": pair["reader_checkpoint_sha256"],
        "model_file_manifest_sha256": prereg["source"][
            "model_file_manifest_sha256"
        ],
        "tokenizer_manifest_sha256": prereg["source"]["tokenizer_manifest_sha256"],
        "gpu_lock_binding_sha256": resolved["gpu_lock_binding_sha256"],
    }
    resolved["execution_binding"] = execution
    resolved["runtime_binding"] = runtime
    resolved["execution_binding_sha256"] = canonical_sha256(execution)
    resolved["runtime_binding_sha256"] = canonical_sha256(runtime)
    current = {
        "schema": "memagent.commit-retain.capture32-current-binding.v1",
        "run_id": resolved["run_id"],
        "git_commit": resolved["git_commit"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "expected_pair_binding_sha256": pair_sha,
        "physical_gpu_whitelist": pair_gpus,
        "visible_devices": visible,
    }
    resolved["current_binding"] = current
    resolved["current_binding_sha256"] = canonical_sha256(current)
    return resolved


def test_capture32_runtime_binding_recomputes_all_three_digests() -> None:
    prereg = validate_capture32_preregistration(_capture32_preregistration())
    resolved = _capture32_binding_fixture()
    digests = _validate_capture32_runtime_bindings(
        resolved, preregistration=prereg
    )
    assert digests["current_binding_sha256"] == resolved["current_binding_sha256"]
    forged = copy.deepcopy(resolved)
    forged["execution_binding_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        _validate_capture32_runtime_bindings(forged, preregistration=prereg)


def test_capture32_dynamic_gpu67_pair_binding_is_accepted() -> None:
    prereg = validate_capture32_preregistration(_capture32_preregistration())
    resolved = _capture32_binding_fixture(pair_gpus=[6, 7])
    digests = _validate_capture32_runtime_bindings(resolved, preregistration=prereg)
    assert digests["runtime_binding_sha256"] == resolved["runtime_binding_sha256"]


def test_capture32_split_gpu_pair_binding_is_rejected() -> None:
    prereg = validate_capture32_preregistration(_capture32_preregistration())
    resolved = _capture32_binding_fixture(pair_gpus=[6, 7])
    resolved["runtime_binding"]["visible_devices"] = "4,5"
    resolved["runtime_binding_sha256"] = canonical_sha256(
        resolved["runtime_binding"]
    )
    resolved["current_binding"]["runtime_binding_sha256"] = resolved[
        "runtime_binding_sha256"
    ]
    resolved["current_binding_sha256"] = canonical_sha256(
        resolved["current_binding"]
    )
    with pytest.raises(ValueError, match="runtime binding values|split binding"):
        _validate_capture32_runtime_bindings(resolved, preregistration=prereg)


def test_capture32_missing_nested_arm_turn_or_outcome_is_attrition() -> None:
    prereg = validate_capture32_preregistration(_capture32_preregistration())
    records = []
    for row in prereg["selected_inventory"]:
        future_count = row["total_writer_turns"] - row["intervention_writer_turn"] - 1
        records.append({
            "stable_write_id": row["stable_write_id"],
            "pair": {
                "stable_write_id": row["stable_write_id"],
                "prefix_turns": [{} for _ in range(row["intervention_writer_turn"])],
                "candidate": {},
                "arms": {
                    arm: {
                        "future_turns": [{} for _ in range(future_count)],
                        "final_reader": {"outcome": {
                            "prediction": "",
                            "extraction_route": "empty",
                            "format_success": 0.0,
                            "exact_match": 0.0,
                            "token_f1": 0.0,
                            "sub_exact_match": 0.0,
                        }},
                    }
                    for arm in ("COMMIT", "RETAIN")
                },
            },
        })
    _validate_capture32_structural_completeness(records, prereg)
    del records[0]["pair"]["arms"]["RETAIN"]["final_reader"]["outcome"]
    with pytest.raises(Capture32AttritionError, match="reader outcome"):
        _validate_capture32_structural_completeness(records, prereg)


def test_valid_capture4_pipeline_still_reports_more_capture_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.h20.test_commit_retain_capture import pair_payload
    import tools.h20.audit_qwen25_7b_paired_effect_candidate as audit

    manifest = _candidate_manifest(tmp_path)
    pairs = [build_pair_record(pair_payload(index=7 + index, call_offset=0))
             for index in range(4)]
    monkeypatch.setattr(audit, "authenticate_capture", lambda *args, **kwargs: {
        "pairs": pairs,
        "contract_evidence": [],
        "capture_report": {"status": "PASS"},
        "git_evidence": {"git_commit": CAPTURE_COMMIT},
        "artifacts": {},
        "resolved_binding": {},
        "credential": {},
    })
    monkeypatch.setattr(audit, "external_capture_anchor_state", lambda *args: {
        "status": "PASS", "decision": "TEST_EXTERNAL_ANCHOR_PASS"
    })
    monkeypatch.setattr(audit, "_readiness", lambda *args: (
        "PASS",
        "PAIRED_EFFECT_EVIDENCE_READY_FOR_EXTERNAL_REVIEW",
        {
            "minimum_unique_stable_examples": {
                "status": "PASS", "observed": 32, "required": 32
            },
            "predictive_signal": {"status": "PASS", "failures": []},
        },
    ))
    report, bundle = build_report(manifest, verify_pipeline_git=False)
    _bind_test_pipeline(report)
    validate_report_shape(report)
    _assert_schema_valid(report)
    assert report["status"] == "PENDING"
    assert report["decision"] == "PAIRED_EFFECT_CAPTURE4_PILOT_ONLY"
    assert report["source_capture"]["capture_role"] == "pipeline_pilot4_only"
    assert report["gates"]["capture32_inventory"]["status"] == "PENDING"
    assert report["gates"]["capture32_inventory"][
        "capture4_observed_but_excluded"
    ] == 4
    assert report["paired_outcomes"]["observation_count"] == 4
    assert report["gates"]["training_evidence"][
        "minimum_unique_stable_examples"
    ]["observed"] == 0
    assert report["gates"]["training_evidence"][
        "minimum_unique_stable_examples"
    ]["pilot4_observed_but_excluded"] == 4
    assert report["training_authorized"] is False
    assert report["method_selected"] is False
    assert bundle is not None


def test_internal_capture_consistency_without_external_anchor_stays_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.h20.test_commit_retain_capture import pair_payload
    import tools.h20.audit_qwen25_7b_paired_effect_candidate as audit

    manifest = _candidate_manifest(tmp_path)
    pairs = [build_pair_record(pair_payload(index=17 + index, call_offset=0))
             for index in range(4)]
    monkeypatch.setattr(audit, "authenticate_capture", lambda *args, **kwargs: {
        "pairs": pairs,
        "contract_evidence": [],
        "capture_report": {"status": "PASS"},
        "git_evidence": {"git_commit": CAPTURE_COMMIT},
        "artifacts": {"final_report": {"path": "/frozen/final.json", "sha256": "a" * 64}},
        "resolved_binding": {},
        "credential": {},
    })
    report, bundle = build_report(manifest, verify_pipeline_git=False)
    _bind_test_pipeline(report)
    validate_report_shape(report)
    _assert_schema_valid(report)
    assert report["status"] == "PENDING"
    assert report["decision"] == "PAIRED_EFFECT_CAPTURE4_PILOT_ONLY"
    assert report["gates"]["capture_internal_consistency"]["status"] == "PASS"
    assert report["gates"]["capture_external_provenance"]["status"] == "NOT_APPLICABLE"
    assert report["gates"]["capture_authentication"]["status"] == "PASS"
    assert report["training_authorized"] is False
    assert bundle is not None


def test_complete_capture32_reaches_scientific_review_but_not_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.h20.audit_qwen25_7b_paired_effect_candidate as audit

    manifest = _candidate_manifest(tmp_path)
    observations = _observations(32)
    bundle_fixture = build_crossfit_bundle_from_observations(
        observations, fold_count=4, ridge=1.0
    )
    monkeypatch.setattr(audit, "authenticate_capture32", lambda *args, **kwargs: {
        "capture_role": "preregistered_capture32",
        "physical_gpu_whitelist": [6, 7],
        "visible_devices": "6,7",
        "pairs": [{} for _ in range(32)],
        "contract_evidence": [{} for _ in range(32)],
        "capture_report": {"status": "PASS", "pair_count": 32},
        "git_evidence": {"git_commit": CAPTURE_COMMIT},
        "artifacts": {"final_report": {
            "path": "/frozen/capture32-final.json", "sha256": "a" * 64
        }},
        "resolved_binding": {},
        "credential": {},
    })
    monkeypatch.setattr(
        audit, "build_crossfit_bundle",
        lambda *args, **kwargs: (bundle_fixture, observations),
    )
    monkeypatch.setattr(audit, "validate_crossfit_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(audit, "_readiness", lambda *args: (
        "PASS", "PAIRED_EFFECT_EVIDENCE_READY_FOR_EXTERNAL_REVIEW",
        {
            "minimum_unique_stable_examples": {
                "status": "PASS", "observed": 32, "required": 32
            },
            "predictive_signal": {"status": "PASS", "failures": []},
        },
    ))
    report, bundle = build_report(manifest, verify_pipeline_git=False)
    _bind_test_pipeline(report)
    validate_report_shape(report)
    _assert_schema_valid(report)
    assert report["status"] == "PENDING"
    assert report["decision"] == "PAIRED_EFFECT_CAPTURE_PROVENANCE_PENDING"
    assert report["source_capture"]["capture_role"] == "preregistered_capture32"
    assert report["source_capture"]["expected_profile"] \
        == "dynamic_explicit_two_h20"
    assert report["source_capture"]["physical_gpu_whitelist"] == [6, 7]
    assert report["source_capture"]["visible_devices"] == "6,7"
    assert report["gates"]["capture32_inventory"]["status"] == "PASS"
    assert report["gates"]["training_evidence"]["predictive_signal"]["status"] == "PASS"
    assert report["training_authorized"] is False
    assert report["method_selected"] is False
    assert bundle is bundle_fixture


def test_scorer_failure_is_not_hidden_by_pending_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.h20.audit_qwen25_7b_paired_effect_candidate as audit

    manifest = _candidate_manifest(tmp_path)
    observations = _observations(32)
    bundle_fixture = build_crossfit_bundle_from_observations(
        observations, fold_count=4, ridge=1.0
    )
    monkeypatch.setattr(audit, "authenticate_capture32", lambda *args, **kwargs: {
        "capture_role": "preregistered_capture32",
        "physical_gpu_whitelist": [4, 5],
        "visible_devices": "4,5",
        "pairs": [{} for _ in range(32)],
        "contract_evidence": [],
        "capture_report": {"status": "PASS"},
        "git_evidence": {"git_commit": CAPTURE_COMMIT},
        "artifacts": {"final_report": {
            "path": "/frozen/final.json", "sha256": "a" * 64
        }},
        "resolved_binding": {},
        "credential": {},
    })
    monkeypatch.setattr(
        audit, "build_crossfit_bundle",
        lambda *args, **kwargs: (bundle_fixture, observations),
    )
    monkeypatch.setattr(audit, "validate_crossfit_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(audit, "_readiness", lambda *args: (
        "FAIL",
        "PAIRED_EFFECT_NO_GO:SCORER_ADMISSIBILITY",
        {
            "minimum_unique_stable_examples": {
                "status": "PASS", "observed": 32, "required": 32
            },
            "predictive_signal": {
                "status": "FAIL", "failures": ["crossfit signal failed"]
            },
        },
    ))
    report, bundle = build_report(manifest, verify_pipeline_git=False)
    _bind_test_pipeline(report)
    validate_report_shape(report)
    _assert_schema_valid(report)
    assert report["status"] == "FAIL"
    assert report["decision"] == "PAIRED_EFFECT_NO_GO:SCORER_ADMISSIBILITY"
    assert report["gates"]["capture_external_provenance"]["status"] == "PENDING"
    assert report["failures"] == ["crossfit signal failed"]
    assert bundle is not None


def test_training_grade_signal_only_unlocks_external_review_not_training(
    tmp_path: Path,
) -> None:
    manifest = _candidate_manifest(tmp_path)
    status, decision, gates = _readiness(manifest, _passing_diagnostics())
    assert status == "PASS"
    assert decision == "PAIRED_EFFECT_EVIDENCE_READY_FOR_EXTERNAL_REVIEW"
    assert gates["predictive_signal"]["status"] == "PASS"
    assert manifest["claim_boundary"]["training_authorized"] is False
    assert manifest["claim_boundary"]["method_selected"] is False


@pytest.mark.parametrize(
    ("updates", "failure"),
    [
        ({
            "nontrivial_effect_count": 0,
            "distinct_effect_bin_count": 1,
            "mean_absolute_effect": 1e-9,
            "target_variance": 1e-18,
        }, "too few nontrivial paired effects"),
        ({
            "crossfit_pearson_correlation": None,
            "crossfit_mse_improvement_fraction": 0.0,
        }, "does not improve"),
        ({"folds_with_positive_mse_improvement": 2}, "too few folds"),
        ({"nonfinite_score_count": 1}, "non-finite"),
    ],
)
def test_capture32_conservative_signal_gates_reject_dust_constant_or_weak_folds(
    tmp_path: Path, updates: dict, failure: str,
) -> None:
    manifest = _candidate_manifest(tmp_path)
    diagnostics = _passing_diagnostics()
    diagnostics.update(updates)
    status, decision, gates = _readiness(manifest, diagnostics)
    assert status == "FAIL"
    assert decision == "PAIRED_EFFECT_NO_GO:SCORER_ADMISSIBILITY"
    assert any(failure in item for item in gates["predictive_signal"]["failures"])


def test_all_zero_observed_effects_produce_no_admissibility_signal(
    tmp_path: Path,
) -> None:
    manifest = _candidate_manifest(tmp_path)
    observations = _observations(32)
    for row in observations:
        row["paired_effect_target"] = 0.0
        row["commit_token_f1"] = 0.5
        row["retain_token_f1"] = 0.5
    bundle = build_crossfit_bundle_from_observations(
        observations, fold_count=4, ridge=1.0
    )
    diagnostics = crossfit_diagnostics(bundle, observations)
    status, decision, gates = _readiness(manifest, diagnostics)
    assert status == "FAIL"
    assert decision == "PAIRED_EFFECT_NO_GO:SCORER_ADMISSIBILITY"
    assert diagnostics["nontrivial_effect_count"] == 0
    assert diagnostics["crossfit_pearson_correlation"] is None
    assert gates["predictive_signal"]["status"] == "FAIL"


def test_present_but_invalid_capture_is_fail(tmp_path: Path) -> None:
    manifest = _candidate_manifest(tmp_path)
    for path in (
        manifest["source_capture"]["p0_certificate"],
        manifest["source_capture"]["resolved_manifest"],
        manifest["source_capture"]["supervisor_ledger"],
        manifest["source_capture"]["capture_credential"],
        manifest["source_capture"]["capture_ledger"],
        manifest["source_capture"]["capture_run_receipt"],
        manifest["source_capture"]["final_report"],
    ):
        artifact = Path(path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n", encoding="utf-8")
    report, bundle = build_report(manifest, verify_pipeline_git=False)
    assert report["status"] == "FAIL"
    assert report["decision"] == "PAIRED_EFFECT_NO_GO:INVALID_CAPTURE_OR_PIPELINE"
    assert report["training_authorized"] is False
    assert bundle is None


def test_present_but_forged_capture32_cannot_fall_back_to_pilot4(tmp_path: Path) -> None:
    manifest = _candidate_manifest(tmp_path)
    _, paths = _capture32_artifact_state(manifest)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    report, bundle = build_report(manifest, verify_pipeline_git=False)
    _bind_test_pipeline(report)
    validate_report_shape(report)
    _assert_schema_valid(report)
    assert report["status"] == "FAIL"
    assert report["decision"] == "PAIRED_EFFECT_NO_GO:INVALID_CAPTURE_OR_PIPELINE"
    assert "P0 commitment" in report["failures"][0]
    assert bundle is None


def test_symlinked_capture_artifact_fails_closed(tmp_path: Path) -> None:
    manifest = _candidate_manifest(tmp_path)
    paths = [Path(manifest["source_capture"][name]) for name in (
        "p0_certificate", "resolved_manifest", "supervisor_ledger",
        "capture_credential", "capture_ledger", "capture_run_receipt", "final_report",
    )]
    for path in paths[:-1]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    external = tmp_path / "external-final.json"
    external.write_text("{}\n", encoding="utf-8")
    paths[-1].parent.mkdir(parents=True, exist_ok=True)
    paths[-1].symlink_to(external)
    report, _ = build_report(manifest, verify_pipeline_git=False)
    assert report["status"] == "FAIL"
    assert "symlinked capture evidence" in report["failures"][0]


def test_output_writer_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    manifest = _candidate_manifest(tmp_path)
    external = tmp_path / "real-logs"
    external.mkdir()
    (tmp_path / "logs").symlink_to(external, target_is_directory=True)
    report, _ = build_report(manifest, verify_pipeline_git=False)
    with pytest.raises(ValueError, match="traverses a symlink"):
        _write_outputs(manifest, report, None)


def test_capture_code_hashes_must_be_real_git_blobs(tmp_path: Path) -> None:
    candidate = _candidate_manifest(tmp_path)
    capture = _capture_manifest(candidate)
    keys = _code_objects(capture)
    forged = {
        "execution_binding": {
            "execution_code_sha256": {key: "a" * 64 for key in keys},
            "execution_code_combined_sha256": canonical_sha256(
                {key: "a" * 64 for key in keys}
            ),
        }
    }
    with pytest.raises(ValueError, match="not its Git blob"):
        _authenticate_capture_git(candidate, forged, capture)
    missing = copy.deepcopy(forged)
    missing["execution_binding"]["execution_code_sha256"].pop(keys[0])
    with pytest.raises(ValueError, match="incomplete or has extras"):
        _authenticate_capture_git(candidate, missing, capture)


def test_shadowed_recurrent_module_origin_is_rejected(tmp_path: Path) -> None:
    shadow = types.ModuleType("recurrent.shadowed_fixture")
    shadow.__file__ = str(tmp_path / "shadow.py")
    sys.modules[shadow.__name__] = shadow
    try:
        with pytest.raises(ImportError, match="outside REPO_ROOT"):
            _assert_loaded_repo_module_origins()
    finally:
        sys.modules.pop(shadow.__name__, None)


def test_report_shape_rejects_forged_training_authorization(tmp_path: Path) -> None:
    manifest = _candidate_manifest(tmp_path)
    report, _ = build_report(manifest, verify_pipeline_git=False)
    _bind_test_pipeline(report)
    report["training_authorized"] = True
    with pytest.raises(ValueError, match="claim firewall"):
        validate_report_shape(report)


def test_shape_and_json_schema_reject_empty_unbound_false_pass(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    manifest = _candidate_manifest(tmp_path)
    report, _ = build_report(manifest, verify_pipeline_git=False)
    _bind_test_pipeline(report)
    report.update({
        "status": "PASS",
        "decision": "PAIRED_EFFECT_EVIDENCE_READY_FOR_EXTERNAL_REVIEW",
        "paired_outcomes": {},
        "crossfit_bundle": {},
        "gates": {
            "capture_internal_consistency": {"status": "PASS"},
            "capture_external_provenance": {"status": "FAIL"},
            "capture_authentication": {"status": "FAIL"},
            "paired_outcome_recomputation": {"status": "PASS"},
            "outcome_hidden_crossfit": {"status": "PASS"},
            "training_evidence": {
                "minimum_unique_stable_examples": {"status": "FAIL"},
                "predictive_signal": {"status": "FAIL"},
            },
        },
        "failures": [],
    })
    with pytest.raises(ValueError, match="report v1 forbids PASS"):
        validate_report_shape(report)
    schema = json.loads((REPO / "paired_effect_admissibility_report.schema.json").read_text())
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    assert errors, "schema accepted an empty/unbound false PASS"


def test_shape_and_schema_reject_fully_populated_forged_pass(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    manifest = _candidate_manifest(tmp_path)
    report, _ = build_report(manifest, verify_pipeline_git=False)
    _bind_test_pipeline(report)
    observations = _observations(32)
    report.update({
        "status": "PASS",
        "decision": "PAIRED_EFFECT_EVIDENCE_READY_FOR_EXTERNAL_REVIEW",
        "source_capture": {
            "artifacts": {
                name: {"path": f"/frozen/{name}.json", "sha256": "a" * 64}
                for name in (
                    "p0_certificate", "resolved_manifest", "supervisor_ledger",
                    "capture_credential", "capture_ledger", "capture_run_receipt",
                    "final_report",
                )
            },
            "contract_evidence": [{} for _ in range(32)],
            "capture_report": {"status": "PASS"},
            "git_evidence": {"git_commit": CAPTURE_COMMIT},
        },
        "paired_outcomes": {
            "target_name": "token_f1_commit_minus_retain",
            "observation_count": 32,
            "observations_sha256": canonical_sha256(observations),
            "observations": observations,
        },
        "crossfit_bundle": {
            "path": "/frozen/bundle.json",
            "bundle_sha256": "b" * 64,
            "file_sha256": "c" * 64,
            "written": True,
            "diagnostics": {"stable_example_count": 32},
        },
        "gates": {
            "capture_internal_consistency": {"status": "PASS"},
            "capture_external_provenance": {"status": "PASS"},
            "capture_authentication": {"status": "PASS"},
            "paired_outcome_recomputation": {"status": "PASS"},
            "outcome_hidden_crossfit": {"status": "PASS"},
            "training_evidence": {
                "minimum_unique_stable_examples": {"status": "PASS"},
                "predictive_signal": {"status": "PASS"},
            },
        },
        "failures": [],
    })
    with pytest.raises(ValueError, match="report v1 forbids PASS"):
        validate_report_shape(report)
    schema = json.loads((REPO / "paired_effect_admissibility_report.schema.json").read_text())
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    assert any(list(error.path) == ["status"] and error.validator == "enum"
               for error in errors), "schema accepted a fully populated forged PASS"

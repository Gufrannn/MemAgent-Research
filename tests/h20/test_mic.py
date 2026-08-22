import json
from pathlib import Path

import pytest

from recurrent.research.mic import (
    CriticCheckpoint, append_jsonl_new, cross_fitted_values, innovation_ledger,
    route_role_advantages, select_trajectory_ledger, sha256_json,
    stable_fold_assignments, stable_source_identities,
    validate_admissible_state,
)


def state(root, trajectory, turn, memory, *, pre=False):
    return {
        "stable_example_id": root + "-example", "stable_root_id": root,
        "trajectory_id": trajectory, "turn_index": turn, "question": "q",
        "visible_chunks": [] if pre else [""] * (turn - 1) + [memory],
        "materialized_memory": memory,
        "materialized_memory_history": [] if pre else [memory] * turn,
        "is_prewrite": pre,
    }


def fixture_rows(count=8):
    rows, outcomes = [], {}
    for index in range(count):
        root, trajectory = f"root-{index}", f"trajectory-{index}"
        outcome = 1.0 if index % 2 else -1.0
        outcomes[trajectory] = outcome
        rows += [state(root, trajectory, 0, "", pre=True),
                 state(root, trajectory, 1, "good" if outcome > 0 else "bad")]
    return rows, outcomes


@pytest.mark.parametrize("key", [
    "gold_answer", "future_chunk", "outcome", "reward", "generated_answer",
    "token_f1", "final_answer",
])
def test_firewall_rejects_taints(key):
    row = state("root", "trajectory", 0, "", pre=True)
    row[key] = "leak"
    with pytest.raises(ValueError, match="forbidden"):
        validate_admissible_state(row)


def test_state_hash_tamper_rejected():
    row = state("root", "trajectory", 0, "", pre=True)
    row["state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_admissible_state(row)


def test_stable_root_folds_are_deterministic_and_grouped():
    ids = ["b", "a", "b", "c", "d"]
    one = stable_fold_assignments(ids, 2)
    two = stable_fold_assignments(reversed(ids), 2)
    assert one == two
    assert one["b"] in (0, 1)


def test_real_recurrent_source_schema_builds_stable_ids_without_prompt_ids_on_original_batch():
    dataset_indices = [11, 11, 19, 19]
    source_snapshot = [[101, 7], [101, 7], [101, 8], [101, 8]]
    roots, examples = stable_source_identities(dataset_indices, source_snapshot, rollout_n=2)
    assert roots[0] == roots[1] and roots[2] == roots[3] and roots[0] != roots[2]
    assert len(set(examples)) == 4
    with pytest.raises(ValueError, match="coverage mismatch"):
        stable_source_identities(dataset_indices, source_snapshot[:-1], rollout_n=2)


def test_oof_receipts_exclude_held_roots_and_close():
    rows, outcomes = fixture_rows()
    oof = cross_fitted_values(rows, outcomes, fold_count=4, dimension=8)
    expected_occupied = len(set(stable_fold_assignments(
        [row["stable_root_id"] for row in rows], 4
    ).values()))
    assert len(oof["receipts"]) == expected_occupied
    ledger = innovation_ledger(oof, outcomes)
    assert ledger["maximum_closure_error"] <= 1e-12
    assert len(ledger["trajectories"]) == 8


def test_cumulative_critic_ledger_selects_only_current_on_policy_step():
    first_rows, first_outcomes = fixture_rows(8)
    second_rows, second_outcomes = fixture_rows(8)
    for row in second_rows:
        suffix = row["trajectory_id"].split("-")[-1]
        row["trajectory_id"] = f"step2-trajectory-{suffix}"
        row["stable_root_id"] = f"step2-root-{suffix}"
        row["stable_example_id"] = f"step2-root-{suffix}-example"
    second_outcomes = {
        f"step2-trajectory-{key.split('-')[-1]}": value
        for key, value in second_outcomes.items()
    }
    combined_rows = [*first_rows, *second_rows]
    combined_outcomes = {**first_outcomes, **second_outcomes}
    cumulative = innovation_ledger(
        cross_fitted_values(combined_rows, combined_outcomes, fold_count=4, dimension=8),
        combined_outcomes,
    )
    current_ids = list(second_outcomes)
    current = select_trajectory_ledger(cumulative, current_ids)
    assert [row["trajectory_id"] for row in current["trajectories"]] == current_ids
    assert len(current["trajectories"]) == 8
    assert current["cumulative_ledger_sha256"] == cumulative["ledger_sha256"]


def test_outcome_coverage_and_duplicate_root_errors():
    rows, outcomes = fixture_rows()
    outcomes.pop("trajectory-0")
    with pytest.raises(ValueError, match="coverage"):
        cross_fitted_values(rows, outcomes)
    rows, outcomes = fixture_rows()
    rows[0]["stable_root_id"] = "different"
    with pytest.raises(ValueError, match="spans stable roots"):
        cross_fitted_values(rows, outcomes)


def test_role_specific_routing_has_no_overlap():
    torch = pytest.importorskip("torch")
    ledger = [{"trajectory_id": "t0", "writer_innovations": [0.25, -0.5],
               "answer_residual": 0.75}]
    output, receipt = route_role_advantages(
        sample_index=torch.tensor([0, 0, 0]),
        final_mask=torch.tensor([False, False, True]),
        turn_index=torch.tensor([1, 2, 3]),
        response_mask=torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.float32),
        ledger_rows=ledger, trajectory_ids=["t0"],
    )
    assert output.tolist() == [[0.25, 0.25], [-0.5, 0.0], [0.75, 0.75]]
    assert receipt["writer_active_tokens"] == 3
    assert receipt["answer_active_tokens"] == 2


def test_role_routing_rejects_missing_turn_and_bad_identity():
    torch = pytest.importorskip("torch")
    kwargs = dict(sample_index=torch.tensor([0]), final_mask=torch.tensor([False]),
                  turn_index=torch.tensor([2]), response_mask=torch.ones((1, 1)),
                  ledger_rows=[{"trajectory_id": "t", "writer_innovations": [1.0],
                                "answer_residual": 0.0}], trajectory_ids=["t"])
    with pytest.raises(ValueError, match="no innovation"):
        route_role_advantages(**kwargs)
    kwargs["trajectory_ids"] = ["other"]
    with pytest.raises(ValueError, match="missing or duplicated"):
        route_role_advantages(**kwargs)


def test_critic_checkpoint_is_separate_immutable_and_tamper_evident(tmp_path):
    path = tmp_path / "critic.json"
    checkpoint = CriticCheckpoint("a" * 40, "b" * 64, "c" * 64, {"folds": 4})
    checkpoint.write_new(path)
    assert CriticCheckpoint.read(path, expected_actor_commit="a" * 40)["critic_payload"] == {"folds": 4}
    with pytest.raises(FileExistsError):
        checkpoint.write_new(path)
    payload = json.loads(path.read_text())
    payload["critic_payload"]["folds"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        CriticCheckpoint.read(path, expected_actor_commit="a" * 40)


def test_append_only_ledger_detects_tampered_tail(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_jsonl_new(path, {"schema": "memagent.mic.training-ledger.v1", "record_type": "x"})
    append_jsonl_new(path, {"schema": "memagent.mic.training-ledger.v1", "record_type": "y"})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[1]["previous_entry_sha256"] == rows[0]["entry_sha256"]
    rows[-1]["record_type"] = "tampered"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="corrupted"):
        append_jsonl_new(path, {"schema": "memagent.mic.training-ledger.v1", "record_type": "z"})


def test_training_entry_guards_are_present():
    repo = Path(__file__).resolve().parents[2]
    common = (repo / "scripts/h20/mic_common.sh").read_text()
    fresh = (repo / "scripts/h20/run_qwen25_7b_mic_t5.sh").read_text()
    continuation = (repo / "scripts/h20/continue_qwen25_7b_mic.sh").read_text()
    assert "memagent_h20_gpu_${MIC_GPU_A}.lock" in common
    assert "git status --porcelain" in common
    assert "nvidia-smi" in common and "no process changed" in common
    assert "mic_require_training_gates" in fresh
    assert "PHASE=fresh" in fresh and "FRESH_TOTAL_STEPS=5" in fresh
    assert "5:10|10:15|15:20|20:25" in continuation
    assert "MIC_T5_TRAINING_HEALTH_PASS" in continuation
    assert "Original_global_step_3" not in fresh + continuation


def test_manifest_freezes_update1_and_no_warmstart():
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads((repo / "manifests/h20/qwen25_7b_mic_seed2026.json").read_text())
    assert manifest["method"]["enabled_from_update"] == 1
    assert manifest["training"]["fresh_base_only"] is True
    assert manifest["training"]["forbidden_warm_start"] == "Original_global_step_3"
    assert manifest["training"]["anchors"] == [5, 10, 15, 20, 25]

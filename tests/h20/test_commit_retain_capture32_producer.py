from __future__ import annotations

import copy
import fcntl
import json
import os
import subprocess
from pathlib import Path

import pytest

from recurrent.research.commit_retain_capture import (
    build_capture_envelope,
    build_pair_record,
    canonical_sha256,
    validate_capture_ledger,
)
from recurrent.research.gate_a_execution import append_jsonl
from tests.h20.test_commit_retain_capture import pair_payload
from tools.h20.preflight_qwen25_7b_commit_retain_capture32 import (
    CODE_OBJECTS,
    MANIFEST_REL,
    _parse_gpu_pair,
    _validate_anchor_schema,
    _validate_gpu_identity,
    _validate_manifest,
    _validate_schema_instance,
    capture_lock_holder_receipt,
    load_manifest,
    project_frozen_pair_eval_identity,
)
from tools.h20.run_qwen25_7b_commit_retain_capture32 import (
    _pair_identity_from_frozen,
)


REPO = Path(__file__).resolve().parents[2]


def _environment(gpus: str = "5,6") -> dict[str, str]:
    return {
        "MEMAGENT_CAPTURE32_WORK_ROOT": "/data/cw/memagent_work",
        "MEMAGENT_CAPTURE32_REPO_DIR": str(REPO),
        "MEMAGENT_CAPTURE32_EXPECTED_COMMIT": "f" * 40,
        "MEMAGENT_CAPTURE32_RUN_ID": "capture32_test",
        "MEMAGENT_CAPTURE32_PHYSICAL_GPUS": gpus,
    }


def _frozen(pair: dict) -> dict:
    return {
        **{field: pair[field] for field in (
            "example_id", "semantic_dataset_index", "source_order_index",
            "raw_row_position", "production_effective_position", "eval_manifest_hash",
            "source_question_hash", "source_context_hash", "ground_truth_hash",
        )},
        "trajectory_seed": pair["trajectory_seed"],
        "intervention_writer_turn": pair["intervention_writer_turn"],
        "total_writer_turns": pair["total_writer_turns"],
        "question_token_ids_sha256": pair["question_token_ids_sha256"],
        "no_memory_token_ids_sha256": pair["no_memory_state"]["token_ids_sha256"],
        "chunk_token_ids_sha256": [
            canonical_sha256([300]), canonical_sha256([301]), canonical_sha256([302])
        ],
        "future_chunk_token_ids_sha256": [canonical_sha256([302])],
        "context_token_ids_sha256": canonical_sha256([300, 301, 302]),
        "writer_turn0_prompt_token_sha256": pair["prefix_turns"][0]["prompt"][
            "token_ids_sha256"
        ],
        "expected_pair_generate_calls": pair["pair_generate_call_count"],
    }


def _capture32_records(tmp_path: Path) -> tuple[list[dict], list[dict]]:
    path = tmp_path / "capture32.jsonl"
    pairs = []
    for offset in range(32):
        payload = pair_payload(index=1000 + offset, call_offset=offset * 6)
        payload["execution"]["global_generate_call_count"] = 32 * 6
        pair = build_pair_record(payload)
        pairs.append(pair)
        append_jsonl(path, build_capture_envelope(
            pair, experiment_name="capture32-fixture", git_commit="f" * 40,
            run_id="capture32_fixture", execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64, current_binding_sha256="8" * 64,
        ))
    return [json.loads(line) for line in path.read_text().splitlines()], [
        _frozen(pair) for pair in pairs
    ]


def test_dynamic_gpu_pair_is_explicit_distinct_and_ascending() -> None:
    assert _parse_gpu_pair("5,6") == [5, 6]
    for invalid in ("", "4", "4,4", "6,5", "4,5,6", "x,5", "04,5"):
        with pytest.raises(ValueError):
            _parse_gpu_pair(invalid)


def test_runtime_projects_authenticated_top_level_eval_hash_into_real_row_shape() -> None:
    pair = build_pair_record(pair_payload())
    frozen = _frozen(pair)
    manifest_hash = frozen.pop("eval_manifest_hash")
    projected = _pair_identity_from_frozen(
        frozen, eval_manifest_hash=manifest_hash
    )
    assert projected["eval_manifest_hash"] == manifest_hash
    assert all(
        projected[field] == frozen[field]
        for field in projected if field != "eval_manifest_hash"
    )


def test_runtime_rejects_conflicting_or_invalid_eval_hash_projection() -> None:
    pair = build_pair_record(pair_payload())
    frozen = _frozen(pair)
    manifest_hash = frozen["eval_manifest_hash"]
    frozen["eval_manifest_hash"] = "f" * 64
    with pytest.raises(ValueError, match="conflicts with P0"):
        _pair_identity_from_frozen(frozen, eval_manifest_hash=manifest_hash)
    frozen.pop("eval_manifest_hash")
    with pytest.raises(ValueError, match="canonical SHA-256"):
        _pair_identity_from_frozen(frozen, eval_manifest_hash="not-a-sha")


def test_post_generation_validator_projects_real_preregistered_inventory_shape() -> None:
    prereg = json.loads((
        REPO / "manifests/h20/qwen25_7b_paired_effect_capture32_preregistration.json"
    ).read_text())
    frozen = prereg["selected_inventory"]
    manifest_hash = prereg["source"]["eval_manifest_hash"]
    assert len(frozen) == 32
    assert all("eval_manifest_hash" not in row for row in frozen)

    projected = project_frozen_pair_eval_identity(frozen, manifest_hash)
    assert len(projected) == 32
    assert all(row["eval_manifest_hash"] == manifest_hash for row in projected)
    assert all("eval_manifest_hash" not in row for row in frozen)


def test_post_generation_validator_rejects_missing_invalid_or_conflicting_hash() -> None:
    prereg = json.loads((
        REPO / "manifests/h20/qwen25_7b_paired_effect_capture32_preregistration.json"
    ).read_text())
    frozen = prereg["selected_inventory"]
    manifest_hash = prereg["source"]["eval_manifest_hash"]
    conflicting = copy.deepcopy(frozen)
    conflicting[7]["eval_manifest_hash"] = "f" * 64
    with pytest.raises(ValueError, match="conflicts with P0"):
        project_frozen_pair_eval_identity(conflicting, manifest_hash)
    for invalid in ("", "f" * 63, "F" * 64, None):
        with pytest.raises(ValueError, match="canonical SHA-256"):
            project_frozen_pair_eval_identity(frozen, invalid)  # type: ignore[arg-type]


def test_post_generation_real_shape_roundtrips_through_generic_ledger_validator(
    tmp_path: Path,
) -> None:
    records, frozen_with_hash = _capture32_records(tmp_path)
    manifest_hash = frozen_with_hash[0]["eval_manifest_hash"]
    real_preregistered_shape = copy.deepcopy(frozen_with_hash)
    for row in real_preregistered_shape:
        assert row.pop("eval_manifest_hash") == manifest_hash

    with pytest.raises(ValueError, match="stable identity is missing"):
        validate_capture_ledger(
            records, frozen_pairs=real_preregistered_shape,
            experiment_name="capture32-fixture", git_commit="f" * 40,
            run_id="capture32_fixture", execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64,
            current_binding_sha256="8" * 64, expected_pair_count=32,
        )

    projected = project_frozen_pair_eval_identity(
        real_preregistered_shape, manifest_hash
    )
    report = validate_capture_ledger(
        records, frozen_pairs=projected,
        experiment_name="capture32-fixture", git_commit="f" * 40,
        run_id="capture32_fixture", execution_binding_sha256="6" * 64,
        runtime_binding_sha256="7" * 64,
        current_binding_sha256="8" * 64, expected_pair_count=32,
    )
    assert report["pair_count"] == 32


def test_manifest_resolves_dynamic_pair_and_per_gpu_locks() -> None:
    manifest = load_manifest(REPO / MANIFEST_REL, _environment("5,6"))
    _validate_manifest(manifest)
    assert manifest["gpu"]["physical_whitelist"] == [5, 6]
    assert manifest["gpu"]["visible_devices"] == "5,6"
    assert manifest["gpu"]["per_gpu_lock_paths"] == [
        "/data/cw/memagent_work/locks/memagent_h20_gpu_5.lock",
        "/data/cw/memagent_work/locks/memagent_h20_gpu_6.lock",
    ]
    assert manifest["scope"]["examples"] == 32
    assert manifest["scope"]["capture4_may_fill_missing"] is False


def test_capture32_literal_inventory_is_disjoint_4x8_and_exact_353_calls() -> None:
    prereg = json.loads((
        REPO / "manifests/h20/qwen25_7b_paired_effect_capture32_preregistration.json"
    ).read_text())
    rows = prereg["selected_inventory"]
    assert len(rows) == 32
    assert len({row["stable_example_id"] for row in rows}) == 32
    assert len({row["stable_root_id"] for row in rows}) == 32
    assert len({row["stable_write_id"] for row in rows}) == 32
    assert set(prereg["selection"]["selected_sorted_positions"]).isdisjoint(
        prereg["selection"]["prior_observed_pilot_positions"]
    )
    assert [sum(row["crossfit_fold"] == fold for row in rows) for fold in range(4)] \
        == [8, 8, 8, 8]
    calls = [
        row["intervention_writer_turn"] + 1
        + 2 * (row["total_writer_turns"] - row["intervention_writer_turn"])
        for row in rows
    ]
    assert calls.count(11) == 31
    assert calls.count(12) == 1
    assert sum(calls) == 353


def test_generic_ledger_default_stays_four_but_explicit_32_passes(tmp_path: Path) -> None:
    records, frozen = _capture32_records(tmp_path)
    with pytest.raises(ValueError, match="exactly 4"):
        validate_capture_ledger(
            records, frozen_pairs=frozen, experiment_name="capture32-fixture",
            git_commit="f" * 40, run_id="capture32_fixture",
            execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64,
            current_binding_sha256="8" * 64,
        )
    report = validate_capture_ledger(
        records, frozen_pairs=frozen, experiment_name="capture32-fixture",
        git_commit="f" * 40, run_id="capture32_fixture",
        execution_binding_sha256="6" * 64,
        runtime_binding_sha256="7" * 64,
        current_binding_sha256="8" * 64,
        expected_pair_count=32,
    )
    assert report["pair_count"] == 32


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "replacement"])
def test_capture32_rejects_attrition_extra_duplicate_and_replacement(
    tmp_path: Path, kind: str
) -> None:
    records, frozen = _capture32_records(tmp_path)
    if kind == "missing":
        records = records[:-1]
    elif kind == "extra":
        records = [*records, copy.deepcopy(records[-1])]
    elif kind == "duplicate":
        frozen[-1] = copy.deepcopy(frozen[0])
    else:
        frozen[-1]["example_id"] = "capture4-substitute"
    with pytest.raises(ValueError, match="attrition|unique|unexpected|differs|hash chain"):
        validate_capture_ledger(
            records, frozen_pairs=frozen, experiment_name="capture32-fixture",
            git_commit="f" * 40, run_id="capture32_fixture",
            execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64,
            current_binding_sha256="8" * 64,
            expected_pair_count=32,
        )


def test_capture32_rejects_two_run_stitching_and_call_index_reset(
    tmp_path: Path,
) -> None:
    records, frozen = _capture32_records(tmp_path)
    stitched_path = tmp_path / "stitched.jsonl"
    for index, record in enumerate(records):
        pair = copy.deepcopy(record["pair"])
        if index >= 16:
            pair["execution"]["engine_id"] = "second-run-engine"
            pair["execution"]["process_instance_uuid"] = "00000000-0000-0000-0000-000000000002"
        rebuilt = build_pair_record(pair)
        append_jsonl(stitched_path, build_capture_envelope(
            rebuilt, experiment_name="capture32-fixture", git_commit="f" * 40,
            run_id="capture32_fixture", execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64, current_binding_sha256="8" * 64,
        ))
    stitched = [json.loads(line) for line in stitched_path.read_text().splitlines()]
    with pytest.raises(ValueError, match="one process/engine"):
        validate_capture_ledger(
            stitched, frozen_pairs=frozen, experiment_name="capture32-fixture",
            git_commit="f" * 40, run_id="capture32_fixture",
            execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64,
            current_binding_sha256="8" * 64,
            expected_pair_count=32,
        )

    reset_path = tmp_path / "reset.jsonl"
    reset_pairs = []
    for offset in range(32):
        call_offset = offset * 6 if offset < 16 else (offset - 16) * 6
        payload = pair_payload(index=2000 + offset, call_offset=call_offset)
        payload["execution"]["global_generate_call_count"] = 32 * 6
        pair = build_pair_record(payload)
        reset_pairs.append(pair)
        append_jsonl(reset_path, build_capture_envelope(
            pair, experiment_name="capture32-fixture", git_commit="f" * 40,
            run_id="capture32_fixture", execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64, current_binding_sha256="8" * 64,
        ))
    reset = [json.loads(line) for line in reset_path.read_text().splitlines()]
    with pytest.raises(ValueError, match="contiguous"):
        validate_capture_ledger(
            reset, frozen_pairs=[_frozen(pair) for pair in reset_pairs],
            experiment_name="capture32-fixture", git_commit="f" * 40,
            run_id="capture32_fixture", execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64,
            current_binding_sha256="8" * 64, expected_pair_count=32,
        )


def test_dynamic_pair_validator_accepts_nonlegacy_explicit_pair() -> None:
    payload = pair_payload(index=77)
    payload["execution"].update(
        physical_gpu_whitelist=[0, 3],
        visible_devices="0,3",
        physical_gpu_identity=[
            {"physical_index": 0, "uuid": "GPU-0", "pci_bus_id": "0000:01:00.0",
             "name": "NVIDIA H20", "compute_mode": "Default", "mig_mode": "Disabled"},
            {"physical_index": 3, "uuid": "GPU-3", "pci_bus_id": "0000:04:00.0",
             "name": "NVIDIA H20", "compute_mode": "Default", "mig_mode": "Disabled"},
        ],
    )
    assert build_pair_record(payload)["execution"]["visible_devices"] == "0,3"


def test_gpu_identity_rejects_nondefault_compute_or_enabled_mig() -> None:
    devices = [
        {
            "physical_index": index,
            "uuid": f"GPU-{index}",
            "pci_bus_id": f"0000:{index:02X}:00.0",
            "name": "NVIDIA H20",
            "compute_mode": "Default",
            "mig_mode": "Disabled",
        }
        for index in (6, 7)
    ]
    assert _validate_gpu_identity(devices, [6, 7]) == devices
    for field, value in (("compute_mode", "Exclusive Process"), ("mig_mode", "Enabled")):
        forged = copy.deepcopy(devices)
        forged[0][field] = value
        with pytest.raises(ValueError, match="exact two H20s"):
            _validate_gpu_identity(forged, [6, 7])


def test_lock_receipt_rejects_inode_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    paths = [lock_root / "memagent_h20_gpu_8.lock", lock_root / "memagent_h20_gpu_9.lock"]
    handles = [open(path, "w") for path in paths]
    try:
        for handle in handles:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv(
            "MEMAGENT_CAPTURE32_LOCK_FDS",
            ",".join(str(handle.fileno()) for handle in handles),
        )
        monkeypatch.setattr(
            "tools.h20.preflight_qwen25_7b_commit_retain_capture32._process_identity",
            lambda pid: {"pid": pid, "boot_id": "fixture", "process_start_ticks": 1},
        )
        manifest = {
            "work_root": str(tmp_path),
            "gpu": {
                "physical_whitelist": [8, 9],
                "per_gpu_lock_paths": [str(path) for path in paths],
            },
        }
        receipt = capture_lock_holder_receipt(manifest)
        assert len(receipt["locks"]) == 2
        replacement = lock_root / "replacement.lock"
        replacement.write_text("")
        os.replace(replacement, paths[0])
        with pytest.raises(ValueError, match="inode drifted|single-link"):
            capture_lock_holder_receipt(manifest)
    finally:
        for handle in handles:
            handle.close()


def test_shell_and_code_closure_forbid_subset_resume_and_capture4_fill() -> None:
    runner = (REPO / "tools/h20/run_qwen25_7b_commit_retain_capture32.py").read_text()
    common = (REPO / "scripts/h20/commit_retain_capture32_common.sh").read_text()
    wrapper = (REPO / "scripts/h20/run_qwen25_7b_commit_retain_capture32.sh").read_text()
    assert "expected_calls != 353" in runner
    assert "expected_pair_count=32" in runner
    assert "--subset" not in runner and "--resume" not in runner
    assert "memagent_h20_gpu_$CAPTURE32_GPU0.lock" in common
    assert "memagent_h20_gpu_$CAPTURE32_GPU1.lock" in common
    assert common.index("memagent_h20_gpu_$CAPTURE32_GPU0.lock") < common.index(
        "memagent_h20_gpu_$CAPTURE32_GPU1.lock"
    )
    assert wrapper.index("capture32_issue_capture_credential") < wrapper.index(
        "run_qwen25_7b_commit_retain_capture32.py"
    )
    assert "capture4 fill" in wrapper
    for relative in CODE_OBJECTS:
        assert (REPO / relative).is_file(), relative


def test_static_checks_for_capture32_scripts_and_json() -> None:
    for script in (
        "scripts/h20/commit_retain_capture32_common.sh",
        "scripts/h20/preflight_qwen25_7b_commit_retain_capture32.sh",
        "scripts/h20/run_qwen25_7b_commit_retain_capture32.sh",
    ):
        subprocess.run(["bash", "-n", str(REPO / script)], check=True)
    for artifact in (
        "manifests/h20/qwen25_7b_commit_retain_capture32_seed2026.json",
        "manifests/h20/qwen25_7b_commit_retain_capture32_commands.json",
        "commit_retain_capture32_execution_ledger.schema.json",
        "commit_retain_capture32_provenance_anchor.schema.json",
    ):
        json.loads((REPO / artifact).read_text())


def _audit_schema_record() -> dict:
    return {
        "record_type": "audit_result",
        "experiment_name": "qwen25_7b_commit_retain_capture32_seed2026",
        "git_commit": "f" * 40,
        "run_id": "capture32_test",
        "recorded_at": "2026-08-21T00:00:00+00:00",
        "eval_manifest_hash": "1" * 64,
        "execution_binding_sha256": "2" * 64,
        "runtime_binding_sha256": "3" * 64,
        "current_binding_sha256": "4" * 64,
        "gpu_lock_binding_sha256": "5" * 64,
        "artifact": "/work/final.json",
        "artifact_sha256": "6" * 64,
        "status": "PASS",
        "decision": "COMMIT_RETAIN_CAPTURE32_LOCAL_AUDIT_COMPLETE_PROVENANCE_PENDING",
        "record_index": 4,
        "previous_record_sha256": "7" * 64,
        "record_sha256": "8" * 64,
        "training_authorized": False,
        "method_selected": False,
        "pair_count": 32,
        "pair_ids": [f"{index:064x}" for index in range(1, 33)],
        "stable_write_ids": [f"{index:064x}" for index in range(33, 65)],
        "generate_call_count": 353,
    }


@pytest.mark.parametrize(
    ("record_type", "decision", "extra"),
    [
        (
            "s0_preflight",
            "COMMIT_RETAIN_CAPTURE32_P0_PASS",
            {
                "resolved_manifest": "/work/resolved.json",
                "resolved_manifest_sha256": "9" * 64,
                "external_preregistration_anchor": "/work/prereg-anchor.json",
                "external_preregistration_anchor_sha256": "a" * 64,
            },
        ),
        (
            "capture_authorization",
            "COMMIT_RETAIN_CAPTURE32_CHILD_AUTHORIZED",
            {
                "lock_holder_receipt_sha256": "9" * 64,
                "parent_credential_id": "a" * 64,
                "parent_issuer_pid": 123,
            },
        ),
        (
            "capture_started",
            "COMMIT_RETAIN_CAPTURE32_STARTED",
            {
                "lock_holder_receipt_sha256": "9" * 64,
                "parent_credential_id": "a" * 64,
                "credential_consumption_sha256": "b" * 64,
            },
        ),
        (
            "capture_complete",
            "COMMIT_RETAIN_CAPTURE32_COMPLETE",
            {
                "pair_count": 32,
                "pair_ids": [f"{index:064x}" for index in range(1, 33)],
                "stable_write_ids": [f"{index:064x}" for index in range(33, 65)],
                "generate_call_count": 353,
                "run_receipt": "/work/run-receipt.json",
                "run_receipt_sha256": "9" * 64,
            },
        ),
    ],
)
def test_executed_ledger_schema_accepts_each_nonterminal_state(
    record_type: str, decision: str, extra: dict
) -> None:
    record = _audit_schema_record()
    for field in (
        "pair_count", "pair_ids", "stable_write_ids", "generate_call_count"
    ):
        record.pop(field)
    record.update(record_type=record_type, decision=decision, **extra)
    _validate_schema_instance(
        "commit_retain_capture32_execution_ledger.schema.json", record
    )


def test_executed_ledger_schema_rejects_unknown_field_and_non353() -> None:
    valid = _audit_schema_record()
    _validate_schema_instance(
        "commit_retain_capture32_execution_ledger.schema.json", valid
    )
    extra = {**valid, "forged_training_claim": True}
    with pytest.raises(ValueError, match="validation failed"):
        _validate_schema_instance(
            "commit_retain_capture32_execution_ledger.schema.json", extra
        )
    wrong_calls = {**valid, "generate_call_count": 352}
    with pytest.raises(ValueError, match="validation failed"):
        _validate_schema_instance(
            "commit_retain_capture32_execution_ledger.schema.json", wrong_calls
        )


def test_executed_anchor_schema_rejects_claim_escalation_and_unknown_field() -> None:
    anchor = {
        "schema": "memagent.commit-retain.capture32-provenance-anchor.v1",
        "anchor_kind": "PRE_GENERATION_LOCAL_EXPORT_CANDIDATE",
        "trust_status": "PENDING_EXTERNAL_SIGNATURE",
        "run_id": "capture32_test",
        "git_commit": "f" * 40,
        "recorded_at": "2026-08-21T00:00:00+00:00",
        "preregistration": "/repo/prereg.json",
        "preregistration_sha256": "1" * 64,
        "p0_certificate": "/work/p0.json",
        "p0_certificate_sha256": "2" * 64,
        "resolved_manifest": "/work/resolved.json",
        "resolved_manifest_sha256": "3" * 64,
        "training_authorized": False,
        "method_selected": False,
        "anchor_payload_sha256": "4" * 64,
    }
    _validate_anchor_schema(anchor)
    for forged in (
        {**anchor, "training_authorized": True},
        {**anchor, "forged_signature": "self-signed"},
    ):
        with pytest.raises(ValueError, match="validation failed"):
            _validate_anchor_schema(forged)


def test_failure_wrapper_keeps_locks_until_gpu_idle_check() -> None:
    wrapper = (REPO / "scripts/h20/run_qwen25_7b_commit_retain_capture32.sh").read_text()
    assert "trap capture32_cleanup_on_exit EXIT" in wrapper
    assert wrapper.index("capture32_acquire_locks") < wrapper.index(
        "trap capture32_cleanup_on_exit EXIT"
    )
    assert "capture32_wait_idle || status=81" in wrapper

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from recurrent.research.commit_retain_capture import (
    ARMS,
    build_capture_envelope,
    build_pair_record,
    build_state_blob,
    canonical_sha256,
    stable_capture_ids,
    stable_turn_id,
    validate_capture_ledger,
    validate_pair_record,
    validate_state_blob,
)
from recurrent.research.gate_a_execution import append_jsonl
from recurrent.research.trajectory_seeding import derive_turn_request_seeds
from tools.h20.preflight_qwen25_7b_commit_retain import (
    _validate_manifest,
    validate_capture_credential,
)


REPO = Path(__file__).resolve().parents[2]


HEX = "a" * 64
GPU_IDENTITIES = [
    "6, GPU-deadbeef-0006, NVIDIA H20",
    "7, GPU-deadbeef-0007, NVIDIA H20",
]


def decode(ids: list[int]) -> str:
    return " ".join(str(item) for item in ids)


def identity(index: int = 7) -> dict:
    ground_truth = ["50"]
    return {
        "example_id": str(index),
        "semantic_dataset_index": index,
        "source_order_index": 1,
        "raw_row_position": 9,
        "production_effective_position": 1,
        "eval_manifest_hash": "1" * 64,
        "source_question_hash": "2" * 64,
        "source_context_hash": "3" * 64,
        "ground_truth_hash": canonical_sha256(ground_truth),
    }


def loaded(role: str, turn_id: str | None, tokens: list[int]) -> dict:
    return {
        "source_role": role,
        "source_turn_id": turn_id,
        "state": build_state_blob(tokens),
    }


def prompt(
    ids: list[int], state: dict, *, chunk: list[int] | None = None
) -> dict:
    value = {
        "text": decode(ids),
        "token_ids": ids,
        "template_sha256": "4" * 64 if chunk is not None else "5" * 64,
        "checkpoint_sha256": HEX,
        "loaded_state_receipt": state,
    }
    if chunk is not None:
        value["chunk_token_ids"] = chunk
    return value


def writer(
    *,
    prompt_ids: list[int],
    state: dict,
    chunk: list[int],
    output: list[int],
    seed: int,
    call: int,
) -> dict:
    return {
        "prompt": prompt(prompt_ids, state, chunk=chunk),
        "sampling_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "n": 1,
            "best_of": 1,
            "max_tokens": 8,
        },
        "request_seed": seed,
        "configured_request_seed": seed,
        "actual_request_seed": seed,
        "generate_call_index": call,
        "raw_completion_token_ids": output,
        "eos_token_id": 999,
        "eos_token_positions_removed": [],
        "eos_removal_semantics": "remove_all_eos_matching_native_unpad",
        "state_after": build_state_blob(output),
        "output_text": decode(output),
    }


def final(
    *, prompt_ids: list[int], state: dict, output: list[int], seed: int, call: int
) -> dict:
    return {
        "prompt": prompt(prompt_ids, state),
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "n": 1,
            "best_of": 1,
            "max_tokens": 7,
        },
        "request_seed": seed,
        "configured_request_seed": seed,
        "actual_request_seed": seed,
        "generate_call_index": call,
        "output_token_ids": output,
        "output_text": decode(output),
    }


def pair_payload(index: int = 7, call_offset: int = 0) -> dict:
    stable = identity(index)
    trajectory_seed = 12345 + index
    intervention, total = 1, 3
    stable_ids = stable_capture_ids(
        stable, trajectory_seed=trajectory_seed, writer_turn=intervention
    )
    prefix_id = stable_turn_id(
        stable_write_id=stable_ids["stable_write_id"],
        phase="prefix_writer",
        arm="SHARED",
        writer_turn=0,
    )
    candidate_id = stable_turn_id(
        stable_write_id=stable_ids["stable_write_id"],
        phase="candidate_writer",
        arm="SHARED",
        writer_turn=1,
    )
    seeds = [
        derive_turn_request_seeds([trajectory_seed], [0], turn)[0]
        for turn in range(total + 1)
    ]
    no_memory = build_state_blob([1])
    old = build_state_blob([10])
    candidate = build_state_blob([20])
    commit_future = build_state_blob([30])
    retain_future = build_state_blob([40])
    writer_decode = {
        "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
        "n": 1, "best_of": 1, "max_tokens": 8,
    }
    reader_decode = {
        "temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
        "n": 1, "best_of": 1, "max_tokens": 7,
    }
    future_chunks = [{"writer_turn": 2, "token_ids": [302]}]
    horizon = {
        "future_writer_turns": [2],
        "future_writer_calls_per_arm": 1,
        "final_reader_calls_per_arm": 1,
        "terminal_writer_turn": 2,
    }
    cost = {
        "shared_candidate_generation_calls": 1,
        "per_arm_writer_generation_calls": 1,
        "per_arm_reader_generation_calls": 1,
        "per_arm_total_generation_calls": 2,
        "per_arm_writer_max_tokens": 8,
        "per_arm_reader_max_tokens": 7,
        "budgets_identical_by_design": True,
        "realized_token_counts_are_measured_not_forced_equal": True,
    }
    commit_future_id = stable_turn_id(
        stable_write_id=stable_ids["stable_write_id"],
        phase="future_writer",
        arm="COMMIT",
        writer_turn=2,
    )
    retain_future_id = stable_turn_id(
        stable_write_id=stable_ids["stable_write_id"],
        phase="future_writer",
        arm="RETAIN",
        writer_turn=2,
    )
    return {
        **stable,
        "trajectory_seed": trajectory_seed,
        "intervention_writer_turn": intervention,
        "total_writer_turns": total,
        "question_token_ids": [99],
        "ground_truth": ["50"],
        "no_memory_state": no_memory,
        "prefix_turns": [
            writer(
                prompt_ids=[400],
                state=loaded("no_memory", None, [1]),
                chunk=[300],
                output=[10],
                seed=seeds[0],
                call=call_offset + 1,
            )
        ],
        "old_state": old,
        "candidate": writer(
            prompt_ids=[410],
            state=loaded("old_state", prefix_id, [10]),
            chunk=[301],
            output=[20],
            seed=seeds[1],
            call=call_offset + 2,
        ),
        "shared_contract": {
            "intervention_writer_turn": intervention,
            "total_writer_turns": total,
            "trajectory_seed": trajectory_seed,
            "future_chunks": future_chunks,
            "horizon": horizon,
            "writer_checkpoint_sha256": HEX,
            "reader_checkpoint_sha256": HEX,
            "writer_prompt_template_sha256": "4" * 64,
            "reader_prompt_template_sha256": "5" * 64,
            "writer_decode": writer_decode,
            "reader_decode": reader_decode,
            "cache_contract": {
                "enable_prefix_caching": False,
                "max_num_seqs": 1,
                "one_prompt_per_generate_call": True,
                "kv_state_reuse_across_generate_calls": False,
                "same_engine_for_both_arms": True,
            },
            "cost_contract": cost,
        },
        "arms": {
            "COMMIT": {
                "initial_loaded_state_receipt": loaded("candidate", candidate_id, [20]),
                "future_turns": [
                    writer(
                        prompt_ids=[421],
                        state=loaded("candidate", candidate_id, [20]),
                        chunk=[302],
                        output=[30],
                        seed=seeds[2],
                        call=call_offset + 3,
                    )
                ],
                "final_reader": final(
                    prompt_ids=[129],
                    state=loaded("previous_future_output", commit_future_id, [30]),
                    output=[50],
                    seed=seeds[3],
                    call=call_offset + 4,
                ),
            },
            "RETAIN": {
                "initial_loaded_state_receipt": loaded("old_state", prefix_id, [10]),
                "future_turns": [
                    writer(
                        prompt_ids=[411],
                        state=loaded("old_state", prefix_id, [10]),
                        chunk=[302],
                        output=[40],
                        seed=seeds[2],
                        call=call_offset + 5,
                    )
                ],
                "final_reader": final(
                    prompt_ids=[139],
                    state=loaded("previous_future_output", retain_future_id, [40]),
                    output=[60],
                    seed=seeds[3],
                    call=call_offset + 6,
                ),
            },
        },
        "execution": {
            "backend": "vllm",
            "vllm_version": "0.8.2",
            "strict_vllm": True,
            "tensor_parallel_size": 2,
            "physical_gpu_whitelist": [6, 7],
            "physical_gpu_identity": GPU_IDENTITIES,
            "visible_devices": "6,7",
            "cuda_device_order": "PCI_BUS_ID",
            "prefix_cache_enabled": False,
            "max_num_seqs": 1,
            "one_prompt_per_generate_call": True,
            "engine_construction_count": 1,
            "full_model_sha_verified_at_capture_start": True,
            "trainer_attached": False,
            "actor_training_calls": 0,
            "engine_id": "engine-one",
            "cache_namespace": "cache-none",
            "process_instance_uuid": str(uuid.UUID(int=1)),
            "process_pid": 123,
            "global_generate_call_count": 24,
            "engine_config_sha256": "e" * 64,
            "parent_credential_id": "b" * 64,
            "parent_credential_sha256": "c" * 64,
            "parent_credential_path": "/tmp/fixture-credential.json",
            "parent_issuer_pid": 122,
            "observed_parent_pid": 122,
            "parent_authorization_record_sha256": "d" * 64,
        },
    }


def canonical_pair(index: int = 7, call_offset: int = 0) -> dict:
    return build_pair_record(pair_payload(index, call_offset))


def test_state_blob_is_exact_u32le_and_fail_closed() -> None:
    state = build_state_blob([1, 256, 65537])
    assert state["bytes_b64"] == "AQAAAAABAAABAAEA"
    assert validate_state_blob(state) == state
    corrupt = copy.deepcopy(state)
    corrupt["bytes_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="non-canonical"):
        validate_state_blob(corrupt)


def test_pair_rebuilds_all_evidence_and_outcomes() -> None:
    item = canonical_pair()
    assert validate_pair_record(item, decoder=decode) == item
    assert item["candidate_generation_count"] == 1
    assert item["arms"]["COMMIT"]["initial_loaded_state_receipt"]["state"] == item["candidate"]["state_after"]
    assert item["arms"]["RETAIN"]["initial_loaded_state_receipt"]["state"] == item["old_state"]
    assert item["arms"]["COMMIT"]["final_reader"]["outcome"]["exact_match"] == 1.0


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda row: row["arms"].update(
                COMMIT=row["arms"]["RETAIN"], RETAIN=row["arms"]["COMMIT"]
            ),
            "arm swap|must be an object|initial source",
        ),
        (
            lambda row: row["arms"]["COMMIT"].update(
                initial_loaded_state_receipt=loaded(
                    "candidate", row["candidate"]["stable_turn_id"], [21]
                )
            ),
            "initial load",
        ),
        (
            lambda row: row["arms"]["COMMIT"]["final_reader"]["prompt"].update(
                loaded_state_receipt=loaded(
                    "previous_future_output",
                    row["arms"]["COMMIT"]["future_turns"][0]["stable_turn_id"],
                    [31],
                )
            ),
            "final reader state bytes",
        ),
        (
            lambda row: row["arms"]["COMMIT"]["future_turns"][0]["prompt"].update(
                chunk_token_ids=[999]
            ),
            "future chunk drift",
        ),
        (
            lambda row: row["arms"]["RETAIN"]["future_turns"][0].update(
                actual_request_seed=row["arms"]["RETAIN"]["future_turns"][0]["actual_request_seed"] + 1
            ),
            "RNG seed mismatch",
        ),
        (
            lambda row: row["shared_contract"]["cost_contract"].update(
                per_arm_writer_max_tokens=9
            ),
            "cost contract drifted",
        ),
        (
            lambda row: row["arms"].pop("RETAIN"),
            "exactly COMMIT and RETAIN",
        ),
    ],
)
def test_adversarial_pair_drift_is_rejected(mutator, match: str) -> None:
    item = canonical_pair()
    mutator(item)
    with pytest.raises(ValueError, match=match):
        validate_pair_record(item)


def test_candidate_count_and_handwritten_pass_are_not_trusted() -> None:
    item = canonical_pair()
    item["candidate_generation_count"] = 2
    with pytest.raises(ValueError, match="non-canonical"):
        validate_pair_record(item)
    with pytest.raises(ValueError):
        validate_pair_record({"status": "PASS", "decision": "looks-good"})


def test_four_pair_capture_chain_rejects_attrition_and_handcrafted_fields(tmp_path) -> None:
    path = tmp_path / "captures.jsonl"
    pairs = [canonical_pair(index, offset * 6) for offset, index in enumerate((7, 8, 9, 10))]
    for pair in pairs:
        append_jsonl(
            path,
            build_capture_envelope(
                pair,
                experiment_name="fixture",
                git_commit="f" * 40,
                run_id="fixture1",
                execution_binding_sha256="6" * 64,
                runtime_binding_sha256="7" * 64,
                current_binding_sha256="8" * 64,
            ),
        )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    frozen = [
        {
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
        for pair in pairs
    ]
    report = validate_capture_ledger(
        records,
        frozen_pairs=frozen,
        experiment_name="fixture",
        git_commit="f" * 40,
        run_id="fixture1",
        execution_binding_sha256="6" * 64,
        runtime_binding_sha256="7" * 64,
        current_binding_sha256="8" * 64,
        decoder=decode,
        writer_prompt_builder=lambda question, memory, chunk: [
            sum(question) + sum(memory) + sum(chunk)
        ],
        reader_prompt_builder=lambda question, memory: [sum(question) + sum(memory)],
        expected_pair_binding={
            "writer_checkpoint_sha256": HEX,
            "reader_checkpoint_sha256": HEX,
            "writer_prompt_template_sha256": "4" * 64,
            "reader_prompt_template_sha256": "5" * 64,
            "writer_decode": pairs[0]["shared_contract"]["writer_decode"],
            "reader_decode": pairs[0]["shared_contract"]["reader_decode"],
            "physical_gpu_identity": GPU_IDENTITIES,
            "engine_config_sha256": "e" * 64,
            "global_generate_call_count": 24,
            "eos_token_id": 999,
        },
    )
    assert report["decision"] == "COMMIT_RETAIN_CAPTURE_AUDIT_COMPLETE"
    with pytest.raises(ValueError, match="prompt was not reconstructed from exact loaded bytes"):
        validate_capture_ledger(
            records,
            frozen_pairs=frozen,
            experiment_name="fixture",
            git_commit="f" * 40,
            run_id="fixture1",
            execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64,
            current_binding_sha256="8" * 64,
            writer_prompt_builder=lambda question, memory, chunk: [999],
            reader_prompt_builder=lambda question, memory: [999],
        )
    with pytest.raises(ValueError, match="attrition"):
        validate_capture_ledger(
            records[:-1],
            frozen_pairs=frozen,
            experiment_name="fixture",
            git_commit="f" * 40,
            run_id="fixture1",
            execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64,
            current_binding_sha256="8" * 64,
        )
    forged_path = tmp_path / "forged.jsonl"
    for index, pair in enumerate(pairs):
        envelope = build_capture_envelope(
            pair,
            experiment_name="fixture",
            git_commit="f" * 40,
            run_id="fixture1",
            execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64,
            current_binding_sha256="8" * 64,
        )
        if index == 0:
            envelope["status"] = "PASS"
        append_jsonl(forged_path, envelope)
    forged = [json.loads(line) for line in forged_path.read_text().splitlines()]
    with pytest.raises(ValueError, match="handcrafted fields"):
        validate_capture_ledger(
            forged,
            frozen_pairs=frozen,
            experiment_name="fixture",
            git_commit="f" * 40,
            run_id="fixture1",
            execution_binding_sha256="6" * 64,
            runtime_binding_sha256="7" * 64,
            current_binding_sha256="8" * 64,
        )


def test_frozen_manifest_and_shell_wire_parent_authorization_before_capture() -> None:
    manifest = json.loads(
        (REPO / "manifests/h20/qwen25_7b_commit_retain_capture_seed2026.json").read_text()
    )
    commands = json.loads(
        (REPO / "manifests/h20/qwen25_7b_commit_retain_capture_commands.json").read_text()
    )
    _validate_manifest(manifest)
    assert commands["required_sequence"] == [
        "p0", "capture_authorization", "single_engine_four_pair_capture", "readonly_audit"
    ]
    assert commands["execution"]["physical_gpus"] == [6, 7]
    assert commands["execution"]["backend"] == "strict_vllm_0.8.2"
    assert commands["execution"]["training_updates"] == 0
    shell = (REPO / "scripts/h20/run_qwen25_7b_commit_retain.sh").read_text()
    assert shell.index("commit_retain_issue_capture_credential") < shell.index(
        "tools/h20/run_qwen25_7b_commit_retain.py"
    )
    assert '--credential "$COMMIT_RETAIN_CREDENTIAL"' in shell


def test_capture_credential_is_single_use_direct_parent_and_supervisor_bound(
    tmp_path: Path,
) -> None:
    credential_path = tmp_path / "credential.json"
    ledger_path = tmp_path / "supervisor.jsonl"
    parent_pid = 4242
    manifest = {
        "run_id": "fixture1",
        "paths": {
            "capture_credential": str(credential_path),
            "execution_ledger": str(ledger_path),
        },
    }
    resolved = {
        "eval_manifest_hash": "9" * 64,
        "runtime_binding_sha256": "5" * 64,
        "execution_binding_sha256": "6" * 64,
    }
    current_sha = "7" * 64
    credential = {
        "schema": "memagent.commit-retain.parent-capture-credential.v1",
        "run_id": "fixture1",
        "git_commit": "f" * 40,
        "child_kind": "single_engine_four_pair_capture",
        "child_identity": "fixture1:four-frozen-stable-writes",
        "parent_issuer_pid": parent_pid,
        "issued_at": "2026-08-21T00:00:00+00:00",
        "nonce": "8" * 64,
        "current_binding_sha256": current_sha,
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "child_full_model_sha_required": True,
        "single_use": True,
        "training_authorized": False,
        "method_selected": False,
    }
    credential["parent_credential_id"] = canonical_sha256(credential)
    credential_path.write_text(json.dumps(credential), encoding="utf-8")
    append_jsonl(ledger_path, {"record_type": "s0_preflight"})
    append_jsonl(
        ledger_path,
        {
            "record_type": "capture_authorization",
            "experiment_name": "qwen25_7b_commit_retain_capture_seed2026",
            "git_commit": "f" * 40,
            "run_id": "fixture1",
            "recorded_at": "2026-08-21T00:00:00+00:00",
            "eval_manifest_hash": "9" * 64,
            "execution_binding_sha256": resolved["execution_binding_sha256"],
            "runtime_binding_sha256": resolved["runtime_binding_sha256"],
            "artifact": str(credential_path.resolve()),
            "artifact_sha256": hashlib.sha256(credential_path.read_bytes()).hexdigest(),
            "parent_credential_id": credential["parent_credential_id"],
            "parent_issuer_pid": parent_pid,
            "current_binding_sha256": current_sha,
            "status": "PASS",
            "decision": "COMMIT_RETAIN_CAPTURE_CHILD_AUTHORIZED",
            "training_authorized": False,
            "method_selected": False,
        },
    )
    with patch.dict(
        os.environ,
        {"MEMAGENT_COMMIT_RETAIN_EXPECTED_COMMIT": "f" * 40},
        clear=False,
    ), patch(
        "tools.h20.preflight_qwen25_7b_commit_retain.os.getppid",
        return_value=parent_pid,
    ):
        evidence = validate_capture_credential(
            credential_path,
            manifest=manifest,
            resolved=resolved,
            current_binding_sha256=current_sha,
            require_live_parent=True,
        )
    assert evidence["parent_credential_id"] == credential["parent_credential_id"]
    assert evidence["observed_parent_pid"] == parent_pid

    forged = dict(credential)
    forged["handwritten_pass"] = True
    forged.pop("parent_credential_id")
    forged["parent_credential_id"] = canonical_sha256(forged)
    credential_path.write_text(json.dumps(forged), encoding="utf-8")
    with patch.dict(
        os.environ,
        {"MEMAGENT_COMMIT_RETAIN_EXPECTED_COMMIT": "f" * 40},
        clear=False,
    ), patch(
        "tools.h20.preflight_qwen25_7b_commit_retain.os.getppid",
        return_value=parent_pid,
    ), pytest.raises(ValueError, match="handcrafted fields"):
        validate_capture_credential(
            credential_path,
            manifest=manifest,
            resolved=resolved,
            current_binding_sha256=current_sha,
            require_live_parent=True,
        )

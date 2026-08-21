from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import types
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from recurrent.research.serialization_credit_pilots import (
    SMSB_REGIMES,
    TETRAD_ROLES,
    adjudicate_tetrad_pilot,
    best_length_derangement,
    build_parent_launch_receipt,
    build_capture_record,
    build_replay_request,
    build_tetrad_requests,
    canonical_sha256,
    center_truncate_token_ids,
    summarize_smsb_pilot,
    parent_authority_mac,
    validate_capture_record,
    validate_replay,
    validate_single_request_token_budget,
    validate_tetrad_manifest,
)
from recurrent.research.s128_hotpot_metrics import score_terminal_output
from recurrent.research.stable_eval_identity import validate_resolved_manifest
from recurrent.research.trajectory_seeding import derive_turn_request_seeds
from tools.h20.preflight_qwen25_7b_serialization_credit import (
    _validate_numeric_contract,
    build_generation_capacity_audit,
    select_pilot_rows,
    validate_child_credential,
    verify_current_binding,
    load_manifest,
    parse_gpu_pair,
)
from tools.h20.audit_qwen25_7b_serialization_credit import (
    _authenticate_authoring_from_s128,
    _authenticate_capture_chain_from_s128,
    _rebuild_authoring_from_s128,
    _schema_failures,
)
from tools.h20 import audit_qwen25_7b_serialization_credit as audit_module


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "manifests/h20/qwen25_7b_serialization_credit_pilots_seed2026.json"
COMMANDS = REPO / "manifests/h20/qwen25_7b_serialization_credit_pilots_commands.json"
HEX = "a" * 64
CAPTURE_PROCESS_UUID = str(uuid.UUID(int=1))
AUTHORITY_SECRET = b"serialization-credit-test-authority"
GPU_IDENTITIES = [
    "6, GPU-00000000-0000-0000-0000-000000000006, NVIDIA H20",
    "7, GPU-00000000-0000-0000-0000-000000000007, NVIDIA H20",
]


def decode_fixture(token_ids: list[int]) -> str:
    return "".join(chr(value) for value in token_ids)


def identity(index: int) -> dict:
    return {
        "example_id": str(index),
        "semantic_dataset_index": index,
        "source_order_index": index,
        "raw_row_position": index,
        "production_effective_position": index,
        "eval_manifest_hash": "1" * 64,
        "source_question_hash": "2" * 64,
        "source_context_hash": "3" * 64,
        "ground_truth_hash": canonical_sha256([f"answer-{index}"]),
    }


def final_prompt(index: int, memory: list[int]) -> list[int]:
    return [7, 20 + index, 8, *memory, 9]


def capture(index: int, call_start: int, *, process_generate_count: int = 12) -> dict:
    trajectory_seed = 1000 + index
    writer_seed = derive_turn_request_seeds([trajectory_seed], [0], 0)[0]
    final_seed = derive_turn_request_seeds([trajectory_seed], [0], 1)[0]
    memory = [40 + index]
    input_memory = [30 + index]
    chunk = [35 + index]
    writer_prompt = [10, 20 + index, 11, *input_memory, 12, *chunk, 13]
    return build_capture_record(
        {
            **identity(index),
            "experiment_id": "fixture:SMSB4",
            "engine_id": "capture-engine",
            "cache_namespace": "capture-cache",
            "memory_ledger": [
                {
                    "turn": 0,
                    "writer_prompt_token_ids": writer_prompt,
                    "writer_prompt_token_ids_sha256": canonical_sha256(
                        writer_prompt
                    ),
                    "writer_prompt_token_length": len(writer_prompt),
                    "chunk_start": 0,
                    "chunk_end": len(chunk),
                    "chunk_token_ids": chunk,
                    "chunk_token_ids_sha256": canonical_sha256(chunk),
                    "chunk_token_length": len(chunk),
                    "input_memory_token_ids": input_memory,
                    "input_memory_token_ids_sha256": canonical_sha256(
                        input_memory
                    ),
                    "input_memory_token_length": len(input_memory),
                    "text": f"memory-{index}",
                    "token_ids": memory,
                    "request_seed": writer_seed,
                    "configured_request_seed": writer_seed,
                    "actual_request_seed": writer_seed,
                    "generate_call_index": call_start,
                }
            ],
            "question_token_ids": [20 + index],
            "final_memory_token_ids": memory,
            "final_prompt_token_ids": final_prompt(index, memory),
            "answer_token_ids": [50 + index],
            "temperature_zero_control_answer_token_ids": [60 + index],
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "n": 1,
                "best_of": 1,
                "max_tokens": 1024,
                "do_sample": True,
            },
            "trajectory_seed": trajectory_seed,
            "request_seed": final_seed,
            "final_stochastic_request_seed": final_seed,
            "final_control_request_seed": final_seed,
            "final_stochastic_configured_request_seed": final_seed,
            "final_stochastic_actual_request_seed": final_seed,
            "final_control_configured_request_seed": final_seed,
            "final_control_actual_request_seed": final_seed,
            "final_stochastic_generate_call_index": call_start + 1,
            "final_control_generate_call_index": call_start + 2,
            "hashes": {name: HEX for name in ("model", "tokenizer", "config", "code")},
            "vllm_version": "0.8.2",
            "updater_calls": 1,
            "prompt_template_sha256": "5" * 64,
            "runtime_binding_sha256": "6" * 64,
            "engine_config_sha256": "7" * 64,
            "current_binding_sha256": "8" * 64,
            "execution": {
                "strict_vllm": True,
                "tensor_parallel_size": 2,
                "physical_gpu_whitelist": [2, 3],
                "physical_gpu_identity": GPU_IDENTITIES,
                "cuda_device_order": "PCI_BUS_ID",
                "prefix_cache_enabled": False,
                "process_instance_uuid": CAPTURE_PROCESS_UUID,
                "process_pid": 101,
                "observed_parent_pid": 900,
                "engine_construction_count": 1,
                "generate_call_count": process_generate_count,
                "full_model_sha_verified_at_capture_start": True,
                "full_model_sha_verified_at_child_start": True,
                "model_manifest_sha256": HEX,
                "parent_credential_id": "a" * 64,
                "parent_credential_mac": "b" * 64,
                "parent_credential_sha256": "c" * 64,
                "parent_credential_path": "/fixture/capture.credential.json",
                "parent_issuer_pid": 900,
                "runtime_binding_sha256": "6" * 64,
                "execution_binding_sha256": "d" * 64,
                "current_binding_sha256": "8" * 64,
            },
            "ground_truth": [f"answer-{index}"],
        }
    )


def replay_payload(item: dict, regime: str, ordinal: int) -> dict:
    request = build_replay_request(
        item,
        regime,
        replay_engine_id=f"replay-engine-{ordinal}",
        replay_cache_namespace=f"replay-cache-{ordinal}",
        independent_seed=item["request_seed"] + 33 if regime == "independent_seed" else None,
    )
    answer = (
        item["temperature_zero_control_answer_token_ids"]
        if regime == "temperature_zero"
        else item["answer_token_ids"]
    )
    result = {
        "request_id": request["request_id"],
        "engine_id": request["engine_id"],
        "cache_namespace": request["cache_namespace"],
        "fresh_engine_verified": True,
        "cache_isolation_verified": True,
        "single_request_execution_verified": True,
        "max_num_seqs": 1,
        "prefix_cache_enabled": False,
        "observed_updater_calls": 0,
        "context_or_chunks_visible": False,
        "prompt_reconstructed_from_serialized_state": True,
        "hashes": item["hashes"],
        "vllm_version": "0.8.2",
        "runtime_binding_sha256": item["runtime_binding_sha256"],
        "engine_config_sha256": item["engine_config_sha256"],
        "current_binding_sha256": item["current_binding_sha256"],
        "execution_binding_sha256": "d" * 64,
        "model_manifest_sha256": item["hashes"]["model"],
        "prompt_token_ids": item["final_prompt_token_ids"],
        "prompt_token_ids_sha256": canonical_sha256(item["final_prompt_token_ids"]),
        "answer_token_ids": answer,
        "answer_token_ids_sha256": canonical_sha256(answer),
        "tensor_parallel_size": 2,
        "physical_gpu_whitelist": [2, 3],
        "physical_gpu_identity": GPU_IDENTITIES,
        "cuda_device_order": "PCI_BUS_ID",
        "process_instance_uuid": str(uuid.UUID(int=100 + ordinal)),
        "process_pid": 1000 + ordinal,
        "engine_construction_count": 1,
        "generate_call_count": 1,
        "full_model_sha_verified_at_child_start": True,
        "parent_credential_id": f"{ordinal:064x}",
        "parent_credential_mac": f"{ordinal + 200:064x}",
        "parent_credential_sha256": f"{ordinal + 100:064x}",
        "parent_issuer_pid": 9000 + ordinal,
        "observed_parent_pid": 9000 + ordinal,
        "configured_request_seed": request["request_seed"],
        "actual_request_seed": request["request_seed"],
    }
    validation = validate_replay(item, request, result)
    return {
        "capture_id": item["capture_id"],
        "request": request,
        "result": result,
        "validation": validation,
    }


def four_captures() -> list[dict]:
    return [capture(index, index * 3 + 1) for index in range(4)]


def parent_receipt(
    *,
    child_kind: str,
    child_identity: str,
    artifact_payload: object,
    evidence: dict,
    ordinal: int,
) -> dict:
    return build_parent_launch_receipt(
        {
            "schema": "memagent.serialization-credit.parent-launch-receipt.v2",
            "record_type": "parent_launch_receipt",
            "recorded_at": "2026-08-21T00:00:00+00:00",
            "child_kind": child_kind,
            "child_identity": child_identity,
            "credential_id": evidence["parent_credential_id"],
            "credential_mac": evidence["parent_credential_mac"],
            "credential_sha256": evidence["parent_credential_sha256"],
            "parent_launcher_pid": evidence["observed_parent_pid"],
            "child_pid": evidence["process_pid"],
            "observed_child_ppid": evidence["observed_parent_pid"],
            "child_exit_code": 0,
            "parent_observed_launch": True,
            "process_instance_uuid": evidence["process_instance_uuid"],
            "artifact": f"/fixture/artifact-{ordinal}.json",
            "artifact_sha256": f"{ordinal + 4000:064x}",
            "artifact_canonical_sha256": canonical_sha256(artifact_payload),
            "stdout_artifact": f"/fixture/stdout-{ordinal}.log",
            "stdout_artifact_sha256": f"{ordinal + 5000:064x}",
            "runner_argv_sha256": f"{ordinal + 6000:064x}",
            "runner_code_sha256": "e" * 64,
            "current_binding_sha256": evidence["current_binding_sha256"],
            "runtime_binding_sha256": evidence["runtime_binding_sha256"],
            "execution_binding_sha256": evidence["execution_binding_sha256"],
            "pre_child_full_model_sha_verified": evidence[
                "full_model_sha_verified_at_child_start"
            ],
            "pre_child_model_manifest_sha256": evidence[
                "model_manifest_sha256"
            ],
            "pre_child_physical_gpu_identity": evidence[
                "physical_gpu_identity"
            ],
            "pre_child_physical_gpu_identity_sha256": canonical_sha256(
                evidence["physical_gpu_identity"]
            ),
            "post_child_full_model_sha_verified": True,
            "post_child_model_manifest_sha256": evidence[
                "model_manifest_sha256"
            ],
            "post_child_current_binding_sha256": evidence[
                "current_binding_sha256"
            ],
            "post_child_physical_gpu_identity": evidence[
                "physical_gpu_identity"
            ],
            "post_child_physical_gpu_identity_sha256": canonical_sha256(
                evidence["physical_gpu_identity"]
            ),
            "post_child_cuda_device_order": "PCI_BUS_ID",
            "authority_secret_sha256": hashlib.sha256(
                AUTHORITY_SECRET
            ).hexdigest(),
            "training_authorized": False,
        },
        AUTHORITY_SECRET,
    )


def smsb_parent_receipts(
    captures: list[dict], replays: list[dict]
) -> tuple[dict, list[dict]]:
    capture_receipt = parent_receipt(
        child_kind="smsb_capture",
        child_identity="capture4",
        artifact_payload=captures,
        evidence=captures[0]["execution"],
        ordinal=1,
    )
    replay_receipts = [
        parent_receipt(
            child_kind="smsb_replay",
            child_identity=(
                f"{payload['validation']['example_id']}::"
                f"{payload['validation']['regime']}"
            ),
            artifact_payload=payload,
            evidence=payload["result"],
            ordinal=ordinal + 2,
        )
        for ordinal, payload in enumerate(replays)
    ]
    return capture_receipt, replay_receipts


def tetrad_parent_receipts(results: list[dict]) -> list[dict]:
    return [
        parent_receipt(
            child_kind="tetrad_replay",
            child_identity=result["request_id"],
            artifact_payload=result,
            evidence=result,
            ordinal=ordinal + 100,
        )
        for ordinal, result in enumerate(results)
    ]


def authoring_rows(captures: list[dict]) -> list[dict]:
    rows = []
    for item in captures:
        index = int(item["example_id"])
        common = {
            **{field: item[field] for field in identity(index)},
            "question_token_ids": item["question_token_ids"],
            "ground_truth": item["ground_truth"],
            "question_type": "hotpot_multihop",
            "answer_type": "hotpot_short_span",
            "checkpoint_hash": HEX,
            "model_hash": HEX,
            "tokenizer_hash": HEX,
            "hashes": item["hashes"],
            "vllm_version": "0.8.2",
            "runtime_binding_sha256": item["runtime_binding_sha256"],
            "engine_config_sha256": item["engine_config_sha256"],
        "current_binding_sha256": item["current_binding_sha256"],
        "execution_binding_sha256": "d" * 64,
            "prompt_protocol_hash": "5" * 64,
            "prompt_outside_memory_span_hash": "9" * 64,
            "physical_gpu_identity": GPU_IDENTITIES,
            "cuda_device_order": "PCI_BUS_ID",
            "generated": {
                "state_id": f"{index}:generated",
                "memory_token_ids": item["final_memory_token_ids"],
                "validity_status": "pass",
                "smsb_status": "pass",
            },
            "empty": {
                "state_id": f"{index}:empty",
                "memory_token_ids": [],
                "validity_status": "pass",
            },
            "irrelevant": {
                "state_id": f"{index}:irrelevant",
                "memory_token_ids": [70 + index],
                "validity_status": "pass",
                "support_answer_bridge_leakage_audit": "pass",
                "length_match_audit": "pass",
            },
            "gold": {
                "state_id": f"{index}:gold",
                "memory_token_ids": [80 + index],
                "validity_status": "pass",
                "canonical_authoring_audit": "pass",
            },
            "shuffle_approved_donor_ids": [str((index + 1) % 4)],
            "shuffle_memory_token_delta": 0,
            "generated_memory_token_length": 1,
            "gold_memory_token_length": 1,
            "irrelevant_memory_token_length": 1,
            "full_model_sha_verified_at_tetrad_start": True,
        }
        rows.append(common)
    return rows


def tetrad_fixture() -> tuple[list[dict], list[dict], list[dict]]:
    captures = four_captures()
    authoring = authoring_rows(captures)
    matching = {str(index): str((index + 1) % 4) for index in range(4)}
    requests = build_tetrad_requests(
        authoring,
        matching=matching,
        base_seed=2026,
        prompt_builder=lambda question, memory: [7, *question, 8, *memory, 9],
        prompt_template_sha256="5" * 64,
        capture_prompt_ids={row["example_id"]: row["final_prompt_token_ids"] for row in captures},
    )
    results = []
    for ordinal, request in enumerate(requests):
        prompt = request["expected_prompt_token_ids"]
        answer_text = (
            f"\\boxed{{answer-{request['example_id']}}}"
            if request["state_role"] == "gold"
            else "\\boxed{wrong}"
        )
        answer = [ord(character) for character in answer_text]
        metrics = score_terminal_output(answer_text, request["ground_truth"])
        results.append(
            {
                "request_id": request["request_id"],
                "example_id": request["example_id"],
                "state_role": request["state_role"],
                "engine_id": f"engine-{ordinal}",
                "cache_namespace": request["cache_namespace"],
                "fresh_engine_verified": True,
                "single_request_execution_verified": True,
                "tensor_parallel_size": 2,
                "physical_gpu_whitelist": [2, 3],
                "physical_gpu_identity": GPU_IDENTITIES,
                "cuda_device_order": "PCI_BUS_ID",
                "max_num_seqs": 1,
                "prefix_cache_enabled": False,
                "observed_updater_calls": 0,
                "context_or_chunks_visible": False,
                "prompt_reconstructed_from_question_and_serialized_memory": True,
                "prompt_token_ids": prompt,
                "prompt_token_sha256": canonical_sha256(prompt),
                "prompt_token_ids_sha256": canonical_sha256(prompt),
                "answer_token_ids": answer,
                "answer_token_ids_sha256": canonical_sha256(answer),
                "answer_text": answer_text,
                "score": float(metrics["token_f1"]),
                "exact_match": float(metrics["exact_match"]),
                "format_success": float(metrics["format_success"]),
                "extraction_route": metrics["extraction_route"],
                "construction_only_pilot": True,
                "effects_reportable": False,
                "process_instance_uuid": str(uuid.UUID(int=1000 + ordinal)),
                "process_pid": 2000 + ordinal,
                "engine_construction_count": 1,
                "generate_call_count": 1,
                "full_model_sha_verified_at_child_start": True,
                "parent_credential_id": f"{ordinal + 1000:064x}",
                "parent_credential_mac": f"{ordinal + 3000:064x}",
                "parent_credential_sha256": f"{ordinal + 2000:064x}",
                "parent_issuer_pid": 10000 + ordinal,
                "observed_parent_pid": 10000 + ordinal,
                "configured_request_seed": request["request_seed"],
                "actual_request_seed": request["request_seed"],
                "vllm_version": request["vllm_version"],
                "hashes": request["hashes"],
                "runtime_binding_sha256": request["runtime_binding_sha256"],
                "engine_config_sha256": request["engine_config_sha256"],
                "current_binding_sha256": request["current_binding_sha256"],
                "execution_binding_sha256": "d" * 64,
                "model_manifest_sha256": request["hashes"]["model"],
            }
        )
    return authoring, requests, results


def test_capture_seed_and_generate_schedule_round_trip() -> None:
    captures = four_captures()
    for item in captures:
        assert validate_capture_record(item) == item
    replays = [
        replay_payload(item, regime, ordinal + 1)
        for ordinal, (item, regime) in enumerate(
            (item, regime) for item in captures for regime in SMSB_REGIMES
        )
    ]
    capture_receipt, replay_receipts = smsb_parent_receipts(captures, replays)
    report = summarize_smsb_pilot(
        captures,
        replays,
        capture_receipt=capture_receipt,
        replay_receipts=replay_receipts,
        authority_secret=AUTHORITY_SECRET,
    )
    assert report["status"] == "PASS"
    assert report["E_det_pass"] is True
    assert report["L2_report_only"] is True


def test_handcrafted_results_without_parent_observation_fail_closed() -> None:
    """A complete-looking result fixture is not execution evidence by itself."""
    captures = four_captures()
    replays = [
        replay_payload(item, regime, ordinal + 1)
        for ordinal, (item, regime) in enumerate(
            (item, regime) for item in captures for regime in SMSB_REGIMES
        )
    ]
    report = summarize_smsb_pilot(captures, replays)
    assert report["status"] == "FAIL"
    assert "parent_receipt_capture_missing" in report["errors"]
    assert "parent_receipt_replays_missing" in report["errors"]

    authoring, requests, results = tetrad_fixture()
    with pytest.raises(ValueError, match="authenticated parent launch receipts"):
        adjudicate_tetrad_pilot(
            requests,
            authoring,
            results,
            answer_decoder=decode_fixture,
        )


def test_parent_receipt_mac_and_unique_supervisor_pid_are_gate_critical() -> None:
    captures = four_captures()
    replays = [
        replay_payload(item, regime, ordinal + 1)
        for ordinal, (item, regime) in enumerate(
            (item, regime) for item in captures for regime in SMSB_REGIMES
        )
    ]
    capture_receipt, replay_receipts = smsb_parent_receipts(captures, replays)
    tampered_receipts = copy.deepcopy(replay_receipts)
    tampered_receipts[0]["receipt_mac"] = "0" * 64
    report = summarize_smsb_pilot(
        captures,
        replays,
        capture_receipt=capture_receipt,
        replay_receipts=tampered_receipts,
        authority_secret=AUTHORITY_SECRET,
    )
    assert report["status"] == "FAIL"
    assert any(
        error.startswith("parent_receipt_replay_invalid:")
        for error in report["errors"]
    )

    duplicate_parent = copy.deepcopy(replay_receipts)
    unsigned = {
        key: value
        for key, value in duplicate_parent[1].items()
        if key not in {"receipt_id", "receipt_mac"}
    }
    unsigned["parent_launcher_pid"] = duplicate_parent[0]["parent_launcher_pid"]
    unsigned["observed_child_ppid"] = duplicate_parent[0]["parent_launcher_pid"]
    duplicate_parent[1] = build_parent_launch_receipt(unsigned, AUTHORITY_SECRET)
    report = summarize_smsb_pilot(
        captures,
        replays,
        capture_receipt=capture_receipt,
        replay_receipts=duplicate_parent,
        authority_secret=AUTHORITY_SECRET,
    )
    assert report["status"] == "FAIL"
    assert "parent_launcher_pid_not_unique" in report["errors"]

    post_model_tamper = {
        key: value
        for key, value in replay_receipts[0].items()
        if key not in {"receipt_id", "receipt_mac"}
    }
    post_model_tamper["post_child_model_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="post-child model/GPU"):
        build_parent_launch_receipt(post_model_tamper, AUTHORITY_SECRET)
    post_gpu_tamper = {
        key: value
        for key, value in replay_receipts[0].items()
        if key not in {"receipt_id", "receipt_mac"}
    }
    post_gpu_tamper["post_child_physical_gpu_identity"] = list(
        reversed(GPU_IDENTITIES)
    )
    post_gpu_tamper["post_child_physical_gpu_identity_sha256"] = canonical_sha256(
        post_gpu_tamper["post_child_physical_gpu_identity"]
    )
    with pytest.raises(ValueError, match="post-child model/GPU"):
        build_parent_launch_receipt(post_gpu_tamper, AUTHORITY_SECRET)


def test_smsb_duplicate_process_or_broken_call_schedule_fails_closed() -> None:
    captures = four_captures()
    replays = [
        replay_payload(item, regime, ordinal + 1)
        for ordinal, (item, regime) in enumerate(
            (item, regime) for item in captures for regime in SMSB_REGIMES
        )
    ]
    replays[1]["result"]["process_instance_uuid"] = replays[0]["result"]["process_instance_uuid"]
    capture_receipt, replay_receipts = smsb_parent_receipts(captures, replays)
    report = summarize_smsb_pilot(
        captures,
        replays,
        capture_receipt=capture_receipt,
        replay_receipts=replay_receipts,
        authority_secret=AUTHORITY_SECRET,
    )
    assert report["status"] == "FAIL"
    assert "replay_process_instance_not_unique" in report["errors"]
    broken = copy.deepcopy(captures)
    broken[-1]["final_control_generate_call_index"] = 99
    with pytest.raises(ValueError):
        validate_capture_record(broken[-1])


def test_smsb_revalidates_every_replay_and_rejects_pid_or_credential_reuse() -> None:
    captures = four_captures()
    replays = [
        replay_payload(item, regime, ordinal + 1)
        for ordinal, (item, regime) in enumerate(
            (item, regime) for item in captures for regime in SMSB_REGIMES
        )
    ]

    stale = copy.deepcopy(replays)
    stale[0]["validation"]["execution_valid"] = False
    capture_receipt, replay_receipts = smsb_parent_receipts(captures, replays)
    report = summarize_smsb_pilot(
        captures,
        stale,
        capture_receipt=capture_receipt,
        replay_receipts=replay_receipts,
        authority_secret=AUTHORITY_SECRET,
    )
    assert report["status"] == "FAIL"
    assert any(
        error.startswith("persisted_replay_validation_mismatch:")
        for error in report["errors"]
    )

    duplicate_pid = copy.deepcopy(replays)
    duplicate_pid[1]["result"]["process_pid"] = duplicate_pid[0]["result"][
        "process_pid"
    ]
    duplicate_pid[1]["validation"] = validate_replay(
        captures[0], duplicate_pid[1]["request"], duplicate_pid[1]["result"]
    )
    report = summarize_smsb_pilot(
        captures,
        duplicate_pid,
        capture_receipt=capture_receipt,
        replay_receipts=replay_receipts,
        authority_secret=AUTHORITY_SECRET,
    )
    assert report["status"] == "FAIL"
    assert "replay_process_pid_not_unique" in report["errors"]

    duplicate_credential = copy.deepcopy(replays)
    duplicate_credential[1]["result"]["parent_credential_id"] = (
        duplicate_credential[0]["result"]["parent_credential_id"]
    )
    duplicate_credential[1]["validation"] = validate_replay(
        captures[0],
        duplicate_credential[1]["request"],
        duplicate_credential[1]["result"],
    )
    report = summarize_smsb_pilot(
        captures,
        duplicate_credential,
        capture_receipt=capture_receipt,
        replay_receipts=replay_receipts,
        authority_secret=AUTHORITY_SECRET,
    )
    assert report["status"] == "FAIL"
    assert "replay_parent_credential_not_unique" in report["errors"]


@pytest.mark.parametrize("bad_token", [True, "7", 7.0])
def test_token_ids_reject_bool_numeric_string_and_float(bad_token: object) -> None:
    item = four_captures()[0]
    item["question_token_ids"] = [bad_token]
    with pytest.raises(ValueError, match="integer"):
        validate_capture_record(item)


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("memory_ledger", 0, "turn"), False),
        (("memory_ledger", 0, "request_seed"), "11"),
        (("memory_ledger", 0, "actual_request_seed"), 12),
        (("final_control_actual_request_seed",), True),
        (("execution", "generate_call_count"), 12.0),
        (("sampling_params", "temperature"), "1.0"),
    ],
)
def test_capture_numeric_evidence_is_strict(path: tuple, bad_value: object) -> None:
    item = four_captures()[0]
    target = item
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    with pytest.raises(ValueError):
        validate_capture_record(item)


def test_two_turn_capture_allows_empty_memory_after_immediate_eos() -> None:
    raw = copy.deepcopy(capture(0, 1, process_generate_count=4))
    raw.pop("capture_id")
    first = raw["memory_ledger"][0]
    first.update(text="", token_ids=[])
    second_seed = derive_turn_request_seeds([raw["trajectory_seed"]], [0], 1)[0]
    second_prompt = [10, 20, 11, 12, 36, 13]
    second_chunk = [36]
    raw["memory_ledger"].append(
        {
            "turn": 1,
            "writer_prompt_token_ids": second_prompt,
            "writer_prompt_token_ids_sha256": canonical_sha256(second_prompt),
            "writer_prompt_token_length": len(second_prompt),
            "chunk_start": 1,
            "chunk_end": 2,
            "chunk_token_ids": second_chunk,
            "chunk_token_ids_sha256": canonical_sha256(second_chunk),
            "chunk_token_length": len(second_chunk),
            "input_memory_token_ids": [],
            "input_memory_token_ids_sha256": canonical_sha256([]),
            "input_memory_token_length": 0,
            "text": "second-memory",
            "token_ids": [41],
            "request_seed": second_seed,
            "configured_request_seed": second_seed,
            "actual_request_seed": second_seed,
            "generate_call_index": 2,
        }
    )
    final_seed = derive_turn_request_seeds([raw["trajectory_seed"]], [0], 2)[0]
    raw.update(
        updater_calls=2,
        final_memory_token_ids=[41],
        final_prompt_token_ids=final_prompt(0, [41]),
        request_seed=final_seed,
        final_stochastic_request_seed=final_seed,
        final_control_request_seed=final_seed,
        final_stochastic_configured_request_seed=final_seed,
        final_stochastic_actual_request_seed=final_seed,
        final_control_configured_request_seed=final_seed,
        final_control_actual_request_seed=final_seed,
        final_stochastic_generate_call_index=3,
        final_control_generate_call_index=4,
    )
    rebuilt = build_capture_record(raw)
    assert rebuilt["memory_ledger"][0]["token_ids"] == []
    assert rebuilt["memory_ledger"][1]["input_memory_token_ids"] == []
    assert rebuilt["memory_ledger"][1]["input_memory_token_length"] == 0


def test_replay_actual_prompt_hash_and_strict_counts_are_audited() -> None:
    item = four_captures()[0]
    payload = replay_payload(item, "temperature_zero", 1)
    payload["result"]["prompt_token_ids_sha256"] = "0" * 64
    validation = validate_replay(item, payload["request"], payload["result"])
    assert validation["execution_valid"] is False
    assert "actual_prompt_token_hash_mismatch" in validation["errors"]
    payload = replay_payload(item, "temperature_zero", 1)
    payload["result"]["generate_call_count"] = True
    validation = validate_replay(item, payload["request"], payload["result"])
    assert "generate_call_count_not_one" in validation["errors"]
    payload = replay_payload(item, "temperature_zero", 1)
    payload["result"]["actual_request_seed"] = str(payload["request"]["request_seed"])
    validation = validate_replay(item, payload["request"], payload["result"])
    assert validation["execution_valid"] is False
    assert any(error.startswith("sampling_or_seed_invalid:") for error in validation["errors"])
    payload = replay_payload(item, "temperature_zero", 1)
    payload["result"]["configured_request_seed"] += 1
    validation = validate_replay(item, payload["request"], payload["result"])
    assert "configured_or_actual_request_seed_mismatch" in validation["errors"]


def test_tetrad_exact_4x5_and_competence_gate() -> None:
    authoring, requests, results = tetrad_fixture()
    receipts = tetrad_parent_receipts(results)
    gate = validate_tetrad_manifest(requests)
    assert gate["request_count"] == 20
    report = adjudicate_tetrad_pilot(
        requests,
        authoring,
        results,
        answer_decoder=decode_fixture,
        parent_receipts=receipts,
        authority_secret=AUTHORITY_SECRET,
    )
    assert report["status"] == "PASS"
    assert report["effects_reportable"] is False
    assert report["training_authorized"] is False
    assert report["method_selection_status"] == "PENDING_EVIDENCE_NO_SELECTION"


def test_tetrad_tampering_and_process_reuse_fail_closed() -> None:
    authoring, requests, results = tetrad_fixture()
    receipts = tetrad_parent_receipts(results)
    bad_requests = copy.deepcopy(requests)
    bad_requests[0]["request_seed"] = True
    with pytest.raises(ValueError):
        validate_tetrad_manifest(bad_requests)
    bad_requests = copy.deepcopy(requests)
    bad_requests[0]["temperature"] = "0.0"
    with pytest.raises(ValueError):
        validate_tetrad_manifest(bad_requests)
    bad_results = copy.deepcopy(results)
    bad_results[1]["process_instance_uuid"] = bad_results[0]["process_instance_uuid"]
    with pytest.raises(ValueError, match="distinct Python process"):
        adjudicate_tetrad_pilot(
            requests,
            authoring,
            bad_results,
            answer_decoder=decode_fixture,
            parent_receipts=receipts,
            authority_secret=AUTHORITY_SECRET,
        )


def test_tetrad_actual_tokens_scores_effects_and_fresh_process_are_reauthenticated() -> None:
    authoring, requests, results = tetrad_fixture()
    receipts = tetrad_parent_receipts(results)

    def adjudicate(candidate_results: list[dict]) -> dict:
        return adjudicate_tetrad_pilot(
            requests,
            authoring,
            candidate_results,
            answer_decoder=decode_fixture,
            parent_receipts=receipts,
            authority_secret=AUTHORITY_SECRET,
        )

    prompt_tamper = copy.deepcopy(results)
    prompt_tamper[0]["prompt_token_ids"] = [999]
    prompt_tamper[0]["prompt_token_sha256"] = canonical_sha256([999])
    prompt_tamper[0]["prompt_token_ids_sha256"] = canonical_sha256([999])
    with pytest.raises(ValueError, match="parent launch receipt|execution certificate"):
        adjudicate(prompt_tamper)

    score_tamper = copy.deepcopy(results)
    score_tamper[0]["score"] = 0.125
    with pytest.raises(ValueError, match="parent launch receipt|independent recomputation"):
        adjudicate(score_tamper)

    text_tamper = copy.deepcopy(results)
    text_tamper[0]["answer_text"] = "\\boxed{attacker-substitution}"
    with pytest.raises(ValueError, match="parent launch receipt|answer text/token identity"):
        adjudicate(text_tamper)

    effects_tamper = copy.deepcopy(results)
    effects_tamper[0]["effects_reportable"] = True
    with pytest.raises(ValueError, match="parent launch receipt|execution certificate"):
        adjudicate(effects_tamper)

    duplicate_pid = copy.deepcopy(results)
    duplicate_pid[1]["process_pid"] = duplicate_pid[0]["process_pid"]
    with pytest.raises(ValueError, match="distinct child process PID"):
        adjudicate(duplicate_pid)

    duplicate_credential = copy.deepcopy(results)
    duplicate_credential[1]["parent_credential_id"] = duplicate_credential[0][
        "parent_credential_id"
    ]
    with pytest.raises(ValueError, match="distinct parent credential"):
        adjudicate(duplicate_credential)

    gpu_tamper = copy.deepcopy(results)
    gpu_tamper[0]["physical_gpu_identity"] = list(reversed(GPU_IDENTITIES))
    with pytest.raises(ValueError, match="parent launch receipt|execution certificate"):
        adjudicate(gpu_tamper)

    full_sha_tamper = copy.deepcopy(results)
    full_sha_tamper[0]["full_model_sha_verified_at_child_start"] = False
    with pytest.raises(ValueError, match="parent launch receipt|execution certificate"):
        adjudicate(full_sha_tamper)
    bad_results = copy.deepcopy(results)
    bad_results[0]["score"] = "0.2"
    with pytest.raises(ValueError, match="parent launch receipt|finite number"):
        adjudicate(bad_results)
    bad_results = copy.deepcopy(results)
    bad_results[0]["actual_request_seed"] = True
    with pytest.raises(ValueError, match="parent launch receipt|execution certificate"):
        adjudicate(bad_results)
    bad_results = copy.deepcopy(results)
    bad_results[0]["configured_request_seed"] = str(requests[0]["request_seed"])
    with pytest.raises(ValueError, match="parent launch receipt|execution certificate"):
        adjudicate(bad_results)

    duplicate_parent_receipts = copy.deepcopy(receipts)
    unsigned = {
        key: value
        for key, value in duplicate_parent_receipts[1].items()
        if key not in {"receipt_id", "receipt_mac"}
    }
    unsigned["parent_launcher_pid"] = duplicate_parent_receipts[0][
        "parent_launcher_pid"
    ]
    unsigned["observed_child_ppid"] = duplicate_parent_receipts[0][
        "parent_launcher_pid"
    ]
    duplicate_parent_receipts[1] = build_parent_launch_receipt(
        unsigned, AUTHORITY_SECRET
    )
    with pytest.raises(ValueError, match="distinct parent supervisor PID"):
        adjudicate_tetrad_pilot(
            requests,
            authoring,
            results,
            answer_decoder=decode_fixture,
            parent_receipts=duplicate_parent_receipts,
            authority_secret=AUTHORITY_SECRET,
        )


def test_tetrad_is_blocked_on_invalid_smsb_generated_state() -> None:
    captures = four_captures()
    authoring = authoring_rows(captures)
    authoring[0]["generated"]["smsb_status"] = "fail"
    with pytest.raises(ValueError, match="SMSB PASS"):
        build_tetrad_requests(
            authoring,
            matching={str(index): str((index + 1) % 4) for index in range(4)},
            base_seed=2026,
            prompt_builder=lambda question, memory: [7, *question, 8, *memory, 9],
            prompt_template_sha256="5" * 64,
            capture_prompt_ids={row["example_id"]: row["final_prompt_token_ids"] for row in captures},
        )


def test_derangement_caliper_and_center_truncation() -> None:
    with pytest.raises(ValueError, match="no perfect derangement"):
        best_length_derangement({"a": 1, "b": 100, "c": 200, "d": 300}, maximum_caliper=10)
    assert center_truncate_token_ids(list(range(10)), 6) == [0, 1, 2, 7, 8, 9]
    assert center_truncate_token_ids(list(range(10)), 5) == [0, 1, 8, 9]


def test_single_request_token_capacity_fails_before_vllm_generate() -> None:
    assert validate_single_request_token_budget(
        [1, 2, 3], 2, max_model_len=8, max_num_batched_tokens=5
    ) == {
        "prompt_tokens": 3,
        "max_tokens": 2,
        "total_tokens": 5,
        "capacity_tokens": 5,
    }
    with pytest.raises(ValueError, match="prompt_tokens=4.*max_tokens=2"):
        validate_single_request_token_budget(
            [1, 2, 3, 4],
            2,
            max_model_len=8,
            max_num_batched_tokens=5,
        )
    with pytest.raises(ValueError, match="not bool/float/string"):
        validate_single_request_token_budget(
            [1], True, max_model_len=8, max_num_batched_tokens=8
        )

    class NeverGenerate:
        def generate(self, **kwargs):
            del kwargs
            raise AssertionError("vLLM generate must not be reached")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = SamplingParams
    from tools.h20 import run_qwen25_7b_serialization_credit as runner

    before = runner._GENERATE_CALL_COUNT
    with patch.dict(sys.modules, {"vllm": fake_vllm}), pytest.raises(
        ValueError, match="exceeds frozen vLLM capacity"
    ):
        runner._generate_one(
            NeverGenerate(),
            [1, 2, 3, 4],
            {"max_tokens": 2},
            manifest={
                "backend": {
                    "max_model_len": 5,
                    "max_num_batched_tokens": 5,
                }
            },
        )
    assert runner._GENERATE_CALL_COUNT == before


def test_p0_capacity_audit_covers_every_writer_turn_and_final_call() -> None:
    class TokenVector(list):
        def tolist(self) -> list[int]:
            return list(self)

    class WriterTemplate:
        def format(self, *, prompt, memory, chunk):
            return TokenVector([1, *prompt, 2, *memory, 3, *chunk, 4])

    class FinalTemplate:
        def format(self, *, prompt, memory):
            return TokenVector([5, *prompt, 6, *memory, 7])

    tokenizer = FakeTokenizer()
    rows = [
        {
            "prompt": [{"role": "user", "content": f"q{index}"}],
            "context": "abcd",
        }
        for index in range(4)
    ]
    writer = WriterTemplate()
    selected = []
    no_memory = tokenizer.encode("No previous memory")
    for index, row in enumerate(rows):
        question = tokenizer.encode(row["prompt"][0]["content"])
        first_chunk = tokenizer.encode(row["context"])[:2]
        prompt_ids = writer.format(
            prompt=question, memory=no_memory, chunk=first_chunk
        ).tolist()
        selected.append(
            {
                "example_id": str(index),
                "raw_row_position": index,
                "writer_turn0_prompt_token_sha256": canonical_sha256(
                    prompt_ids
                ),
            }
        )
    manifest = {
        "recurrent": {
            "no_memory_text": "No previous memory",
            "max_context_tokens": 4,
            "chunk_size": 2,
            "max_chunks": 2,
            "max_memory_tokens": 2,
            "max_final_tokens": 2,
        },
        "backend": {"max_model_len": 64, "max_num_batched_tokens": 64},
    }
    audit = build_generation_capacity_audit(
        selected=selected,
        parquet_rows=rows,
        tokenizer=tokenizer,
        writer_template=writer,
        final_template=FinalTemplate(),
        manifest=manifest,
    )
    assert len(audit) == 4
    assert all(len(row["writer_turns"]) == 2 for row in audit)
    assert all(row["final_reader_upper_bound"] for row in audit)
    overflow = copy.deepcopy(manifest)
    overflow["backend"]["max_num_batched_tokens"] = 5
    with pytest.raises(ValueError, match="exceeds frozen vLLM capacity"):
        build_generation_capacity_audit(
            selected=selected,
            parquet_rows=rows,
            tokenizer=tokenizer,
            writer_template=writer,
            final_template=FinalTemplate(),
            manifest=overflow,
        )


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = False
    ) -> str:
        del skip_special_tokens
        return "".join(chr(value) for value in token_ids)


def test_pilot_selection_is_outcome_blind_and_uses_existing_s128() -> None:
    parquet_rows = []
    stable_rows = []
    for index in range(128):
        question = f"q{index}"
        context = "x" * (index + 1)
        truth = [f"answer-{127 - index}"]
        parquet_rows.append(
            {
                "prompt": [{"role": "user", "content": question}],
                "context": context,
                "reward_model": {"ground_truth": truth},
                "extra_info": {"index": index},
            }
        )
        stable_rows.append(
            {
                **identity(index),
                "source_question_hash": hashlib.sha256(question.encode()).hexdigest(),
                "source_context_hash": hashlib.sha256(context.encode()).hexdigest(),
                "ground_truth_hash": canonical_sha256(truth),
            }
        )
    selected = select_pilot_rows(
        parquet_rows=parquet_rows,
        stable_rows=stable_rows,
        tokenizer=FakeTokenizer(),
        writer_prompt_builder=lambda question, memory, chunk: [*question, *memory, *chunk],
        sorted_positions=[15, 47, 79, 111],
        eval_manifest_hash="1" * 64,
    )
    assert [row["raw_row_position"] for row in selected] == [15, 47, 79, 111]
    changed_rows = copy.deepcopy(parquet_rows)
    changed_stable = copy.deepcopy(stable_rows)
    for index, row in enumerate(changed_rows):
        changed_truth = [f"different-outcome-{index}"]
        row["reward_model"]["ground_truth"] = changed_truth
        changed_stable[index]["ground_truth_hash"] = canonical_sha256(changed_truth)
    changed_selected = select_pilot_rows(
        parquet_rows=changed_rows,
        stable_rows=changed_stable,
        tokenizer=FakeTokenizer(),
        writer_prompt_builder=lambda question, memory, chunk: [*question, *memory, *chunk],
        sorted_positions=[15, 47, 79, 111],
        eval_manifest_hash="1" * 64,
    )
    assert [row["raw_row_position"] for row in changed_selected] == [15, 47, 79, 111]


def test_pilot_selection_projects_real_resolved_manifest_hash_into_row_identities() -> None:
    parquet_rows = []
    stable_rows = []
    for index in range(128):
        question = f"resolved-q{index}"
        context = "z" * (index + 1)
        truth = [f"resolved-answer-{index}"]
        parquet_rows.append(
            {
                "prompt": [{"role": "user", "content": question}],
                "context": context,
                "reward_model": {"ground_truth": truth},
                "extra_info": {"index": index},
            }
        )
        # This is the real Stable-I resolved shape: eval_manifest_hash is a
        # manifest-level binding, not a duplicated MANIFEST_ROW_FIELDS member.
        stable_rows.append(
            {
                "example_id": str(index),
                "semantic_dataset_index": index,
                "source_order_index": index,
                "raw_row_position": index,
                "production_effective_position": index,
                "context_token_count": len(context),
                "source_question_hash": hashlib.sha256(question.encode()).hexdigest(),
                "source_context_hash": hashlib.sha256(context.encode()).hexdigest(),
                "ground_truth_hash": canonical_sha256(truth),
            }
        )
    identity_payload = {"schema_version": 1, "rows": stable_rows}
    eval_manifest_hash = canonical_sha256(identity_payload)
    resolved = validate_resolved_manifest(
        {
            "identity_payload": identity_payload,
            "eval_manifest_hash": eval_manifest_hash,
        }
    )

    selected = select_pilot_rows(
        parquet_rows=parquet_rows,
        stable_rows=resolved["identity_payload"]["rows"],
        tokenizer=FakeTokenizer(),
        writer_prompt_builder=lambda question, memory, chunk: [*question, *memory, *chunk],
        sorted_positions=[15, 47, 79, 111],
        eval_manifest_hash=resolved["eval_manifest_hash"],
    )
    assert [row["raw_row_position"] for row in selected] == [15, 47, 79, 111]
    assert {row["eval_manifest_hash"] for row in selected} == {eval_manifest_hash}

    missing_hash = copy.deepcopy(resolved)
    missing_hash.pop("eval_manifest_hash")
    with pytest.raises(ValueError, match="eval_manifest_hash"):
        validate_resolved_manifest(missing_hash)

    tampered_hash = copy.deepcopy(resolved)
    tampered_hash["eval_manifest_hash"] = "0" * 64
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        validate_resolved_manifest(tampered_hash)

    duplicated_mismatch = copy.deepcopy(resolved["identity_payload"]["rows"])
    duplicated_mismatch[0]["eval_manifest_hash"] = "0" * 64
    with pytest.raises(ValueError, match="row evaluation hash disagrees"):
        select_pilot_rows(
            parquet_rows=parquet_rows,
            stable_rows=duplicated_mismatch,
            tokenizer=FakeTokenizer(),
            writer_prompt_builder=lambda question, memory, chunk: [
                *question, *memory, *chunk
            ],
            sorted_positions=[15, 47, 79, 111],
            eval_manifest_hash=resolved["eval_manifest_hash"],
        )


def test_current_binding_mismatch_is_fail_closed() -> None:
    binding = {"git_commit": "a" * 40, "runtime_versions": {"vllm": "0.8.2"}}
    resolved = {"lightweight_current_binding_sha256": canonical_sha256(binding)}
    with patch(
        "tools.h20.preflight_qwen25_7b_serialization_credit.capture_lightweight_current_binding",
        return_value=binding,
    ):
        assert verify_current_binding({}, resolved, full_model_sha=False) == canonical_sha256(binding)
    with patch(
        "tools.h20.preflight_qwen25_7b_serialization_credit.capture_lightweight_current_binding",
        return_value={**binding, "git_commit": "b" * 40},
    ):
        with pytest.raises(ValueError, match="differs from P0"):
            verify_current_binding({}, resolved, full_model_sha=False)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("smsb", "examples"), 5),
        (("smsb", "fresh_replay_processes"), 11),
        (("tetrad", "examples"), 5),
        (("tetrad", "requests"), 19),
        (("tetrad", "fresh_process_per_request"), False),
        (("tetrad", "deterministic_reader"), False),
        (("tetrad", "effects_reportable"), True),
        (("tetrad", "requires_smsb_decision"), "PASS_ANY_REPORT"),
        (("backend", "required_version"), "0.8.3"),
        (("scope", "actor_updates"), 1),
    ],
)
def test_p0_freezes_exact_pilot_cardinality_and_no_effects(
    path: tuple[str, str], value: object
) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    _validate_numeric_contract(manifest)
    tampered = copy.deepcopy(manifest)
    tampered[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        _validate_numeric_contract(tampered)


def test_parent_credential_is_canonical_direct_child_and_frozen_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    credential_path = root / "credential.json"
    parent_pid = 4242
    current_sha = "4" * 64
    authority_sha = hashlib.sha256(AUTHORITY_SECRET).hexdigest()
    manifest = {"run_id": "pilot4", "paths": {"log_root": str(root)}}
    resolved = {
        "runtime_binding_sha256": "5" * 64,
        "execution_binding_sha256": "6" * 64,
        "parent_receipt_authority": {
            "scheme": "hmac-sha256-run-scoped-v1",
            "secret_sha256": authority_sha,
        },
    }

    def signed_credential(
        *, full_model_sha_required: bool = True, secret: bytes = AUTHORITY_SECRET
    ) -> dict:
        unsigned = {
            "schema": "memagent.serialization-credit.parent-child-credential.v2",
            "run_id": "pilot4",
            "git_commit": "a" * 40,
            "child_kind": "smsb_replay",
            "child_identity": "17::temperature_zero",
            "parent_issuer_pid": parent_pid,
            "issued_at": "2026-08-21T00:00:00+00:00",
            "nonce": "7" * 64,
            "current_binding_sha256": current_sha,
            "runtime_binding_sha256": resolved["runtime_binding_sha256"],
            "execution_binding_sha256": resolved["execution_binding_sha256"],
            "child_full_model_sha_required": full_model_sha_required,
            "parent_authority_secret_sha256": authority_sha,
        }
        signed = {**unsigned, "parent_credential_id": canonical_sha256(unsigned)}
        signed["parent_credential_mac"] = parent_authority_mac(
            secret, "child-credential-v2", signed
        )
        return signed

    credential = signed_credential()
    credential_path.write_text(json.dumps(credential), encoding="utf-8")
    with patch.dict(
        os.environ,
        {"MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT": "a" * 40},
        clear=False,
    ):
        evidence = validate_child_credential(
            credential_path,
            manifest=manifest,
            resolved=resolved,
            current_binding_sha=current_sha,
            child_kind="smsb_replay",
            child_identity="17::temperature_zero",
            authority_secret=AUTHORITY_SECRET,
            expected_issuer_pid=parent_pid,
        )
        assert evidence["parent_credential_id"] == credential[
            "parent_credential_id"
        ]

    self_issued = signed_credential(secret=b"attacker-self-issued-secret-value!!")
    credential_path.write_text(json.dumps(self_issued), encoding="utf-8")
    with patch.dict(
        os.environ,
        {"MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT": "a" * 40},
        clear=False,
    ), pytest.raises(ValueError, match="credential MAC differs"):
        validate_child_credential(
            credential_path,
            manifest=manifest,
            resolved=resolved,
            current_binding_sha=current_sha,
            child_kind="smsb_replay",
            child_identity="17::temperature_zero",
            authority_secret=AUTHORITY_SECRET,
            expected_issuer_pid=parent_pid,
        )

    tampered = signed_credential(full_model_sha_required=False)
    credential_path.write_text(json.dumps(tampered), encoding="utf-8")
    with patch.dict(
        os.environ,
        {"MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT": "a" * 40},
        clear=False,
    ), pytest.raises(ValueError, match="binding differs"):
        validate_child_credential(
            credential_path,
            manifest=manifest,
            resolved=resolved,
            current_binding_sha=current_sha,
            child_kind="smsb_replay",
            child_identity="17::temperature_zero",
            authority_secret=AUTHORITY_SECRET,
            expected_issuer_pid=parent_pid,
        )


def test_authoring_is_independently_rebuilt_from_s128_and_flags_are_not_trusted(
    tmp_path: Path,
) -> None:
    parquet_rows: list[dict] = []
    captures: list[dict] = []
    tokenizer = FakeTokenizer()
    for index in range(128):
        question = f"Which linked item identifies record {index}?"
        answer = f"answer-{index}"
        context = (
            f"Document 1:\nAnswerDoc{index}\nThe canonical answer is {answer}.\n"
            f"Document 2:\nAnchorDoc{index}\nThe linked item identifies record {index}.\n"
            "Document 3:\nNeutral\n"
            + ("zzzz neutral filler " * 40)
        )
        parquet_rows.append(
            {
                "prompt": [{"role": "user", "content": question}],
                "context": context,
                "reward_model": {"ground_truth": [answer]},
                "extra_info": {"index": index},
            }
        )
        if index < 4:
            item = capture(index, index * 3 + 1)
            memory = [ord("m")] * 64
            item["source_question_hash"] = hashlib.sha256(
                question.encode("utf-8")
            ).hexdigest()
            item["source_context_hash"] = hashlib.sha256(
                context.encode("utf-8")
            ).hexdigest()
            item["ground_truth_hash"] = canonical_sha256([answer])
            item["ground_truth"] = [answer]
            item["question_token_ids"] = tokenizer.encode(question)
            item["memory_ledger"][0]["token_ids"] = memory
            item["final_memory_token_ids"] = memory
            item["final_prompt_token_ids"] = [
                7,
                *item["question_token_ids"],
                8,
                *memory,
                9,
            ]
            captures.append(build_capture_record(item))

    class FakeTable:
        def to_pylist(self) -> list[dict]:
            return copy.deepcopy(parquet_rows)

    parquet_module = types.ModuleType("pyarrow.parquet")
    parquet_module.read_table = lambda *args, **kwargs: FakeTable()
    pyarrow_module = types.ModuleType("pyarrow")
    pyarrow_module.parquet = parquet_module
    memory_module = types.ModuleType("recurrent.impls.memory")
    memory_module.TEMPLATE_FINAL_BOXED = "Question: {prompt}\nMemory: {memory}"
    utils_module = types.ModuleType("recurrent.utils")
    utils_module.chat_template = lambda tokenizer: "{message}"
    fake_modules = {
        "pyarrow": pyarrow_module,
        "pyarrow.parquet": parquet_module,
        "recurrent.impls.memory": memory_module,
        "recurrent.utils": utils_module,
    }
    manifest = {
        "data": {"validation": str(tmp_path / "fixed-s128.parquet")},
        "tetrad": {"maximum_shuffle_memory_token_caliper": 256},
    }
    resolved = {
        "runtime_binding_sha256": "6" * 64,
        "execution_binding": {"engine_config_sha256": "7" * 64},
    }
    kwargs = {
        "manifest": manifest,
        "resolved": resolved,
        "current_binding_sha": "8" * 64,
        "captures": captures,
        "tokenizer": tokenizer,
    }
    with patch.dict(sys.modules, fake_modules):
        persisted, _ = _rebuild_authoring_from_s128(**kwargs)
        rebuilt, _ = _authenticate_authoring_from_s128(persisted, **kwargs)
        assert rebuilt == persisted

        flag_only_tamper = copy.deepcopy(persisted)
        flag_only_tamper[0]["irrelevant"]["memory_token_ids"][0] = ord("x")
        flag_only_tamper[0]["irrelevant"][
            "support_answer_bridge_leakage_audit"
        ] = "pass"
        with pytest.raises(ValueError, match="independent S128 rebuild"):
            _authenticate_authoring_from_s128(flag_only_tamper, **kwargs)

        ground_truth_tamper = copy.deepcopy(persisted)
        ground_truth_tamper[0]["ground_truth"] = ["attacker-answer"]
        ground_truth_tamper[0]["ground_truth_hash"] = canonical_sha256(
            ["attacker-answer"]
        )
        with pytest.raises(ValueError, match="independent S128 rebuild"):
            _authenticate_authoring_from_s128(ground_truth_tamper, **kwargs)


def test_smsb_writer_chain_is_independently_rebuilt_from_s128() -> None:
    class TokenVector(list):
        def tolist(self) -> list[int]:
            return list(self)

    class FakeTokenTemplate:
        def __init__(self, template_text: str, tokenizer: FakeTokenizer):
            del tokenizer
            self.template_text = template_text

        def format(self, *, prompt, memory, chunk=None):
            if self.template_text == "WRITER":
                return TokenVector([1, *prompt, 2, *memory, 3, *chunk, 4])
            return TokenVector([5, *prompt, 6, *memory, 7])

    tokenizer = FakeTokenizer()
    parquet_rows: list[dict] = []
    captures: list[dict] = []
    for index in range(128):
        question = f"q{index}"
        context = f"context-{index}"
        answer = f"answer-{index}"
        parquet_rows.append(
            {
                "prompt": [{"role": "user", "content": question}],
                "context": context,
                "reward_model": {"ground_truth": [answer]},
                "extra_info": {"index": index},
            }
        )
        if index >= 4:
            continue
        raw = copy.deepcopy(capture(index, index * 3 + 1))
        raw.pop("capture_id")
        question_ids = tokenizer.encode(question)
        chunk_ids = tokenizer.encode(context)
        input_memory_ids = tokenizer.encode("No previous memory")
        memory_ids = tokenizer.encode(f"M{index}")
        writer_ids = FakeTokenTemplate("WRITER", tokenizer).format(
            prompt=question_ids,
            memory=input_memory_ids,
            chunk=chunk_ids,
        ).tolist()
        ledger = raw["memory_ledger"][0]
        ledger.update(
            writer_prompt_token_ids=writer_ids,
            writer_prompt_token_ids_sha256=canonical_sha256(writer_ids),
            writer_prompt_token_length=len(writer_ids),
            chunk_start=0,
            chunk_end=len(chunk_ids),
            chunk_token_ids=chunk_ids,
            chunk_token_ids_sha256=canonical_sha256(chunk_ids),
            chunk_token_length=len(chunk_ids),
            input_memory_token_ids=input_memory_ids,
            input_memory_token_ids_sha256=canonical_sha256(input_memory_ids),
            input_memory_token_length=len(input_memory_ids),
            text=tokenizer.decode(memory_ids),
            token_ids=memory_ids,
        )
        final_ids = FakeTokenTemplate("FINAL", tokenizer).format(
            prompt=question_ids, memory=memory_ids
        ).tolist()
        raw.update(
            question_token_ids=question_ids,
            final_memory_token_ids=memory_ids,
            final_prompt_token_ids=final_ids,
            prompt_template_sha256=hashlib.sha256(b"FINAL").hexdigest(),
            source_question_hash=hashlib.sha256(question.encode()).hexdigest(),
            source_context_hash=hashlib.sha256(context.encode()).hexdigest(),
            ground_truth_hash=canonical_sha256([answer]),
            ground_truth=[answer],
        )
        captures.append(build_capture_record(raw))

    class FakeTable:
        def to_pylist(self) -> list[dict]:
            return copy.deepcopy(parquet_rows)

    parquet_module = types.ModuleType("pyarrow.parquet")
    parquet_module.read_table = lambda *args, **kwargs: FakeTable()
    pyarrow_module = types.ModuleType("pyarrow")
    pyarrow_module.parquet = parquet_module
    memory_module = types.ModuleType("recurrent.impls.memory")
    memory_module.TEMPLATE = "WRITER"
    memory_module.TEMPLATE_FINAL_BOXED = "FINAL"
    utils_module = types.ModuleType("recurrent.utils")
    utils_module.TokenTemplate = FakeTokenTemplate
    utils_module.chat_template = lambda tokenizer: "{message}"
    fake_modules = {
        "pyarrow": pyarrow_module,
        "pyarrow.parquet": parquet_module,
        "recurrent.impls.memory": memory_module,
        "recurrent.utils": utils_module,
    }
    manifest = {
        "data": {"validation": "/fixture/fixed-s128.parquet"},
        "recurrent": {
            "no_memory_text": "No previous memory",
            "max_context_tokens": 100,
            "chunk_size": 100,
            "max_memory_tokens": 16,
            "max_final_tokens": 16,
        },
        "backend": {"max_model_len": 512, "max_num_batched_tokens": 512},
    }
    resolved = {
        "execution_binding": {
            "writer_prompt_template_sha256": hashlib.sha256(b"WRITER").hexdigest(),
            "final_prompt_template_sha256": hashlib.sha256(b"FINAL").hexdigest(),
        }
    }
    with patch.dict(sys.modules, fake_modules):
        _authenticate_capture_chain_from_s128(
            manifest=manifest,
            resolved=resolved,
            captures=captures,
            tokenizer=tokenizer,
        )
        tampered = copy.deepcopy(captures)
        tampered_raw = copy.deepcopy(tampered[0])
        tampered_raw["memory_ledger"][0]["writer_prompt_token_ids"][0] = 99
        tampered_raw["memory_ledger"][0][
            "writer_prompt_token_ids_sha256"
        ] = canonical_sha256(
            tampered_raw["memory_ledger"][0]["writer_prompt_token_ids"]
        )
        tampered[0] = build_capture_record(tampered_raw)
        with pytest.raises(ValueError, match="independent S128 rebuild"):
            _authenticate_capture_chain_from_s128(
                manifest=manifest,
                resolved=resolved,
                captures=tampered,
                tokenizer=tokenizer,
            )


def test_ledger_schema_rejects_numeric_strings_and_bool_indices(tmp_path: Path) -> None:
    schema = REPO / "serialization_credit_pilot_execution_ledger.schema.json"
    record = {
        "record_type": "s0_preflight",
        "experiment_name": "qwen25_7b_serialization_credit_pilots_seed2026",
        "git_commit": "a" * 40,
        "run_id": "pilot4",
        "recorded_at": "2026-08-21T00:00:00+00:00",
        "eval_manifest_hash": "1" * 64,
        "execution_binding_sha256": "2" * 64,
        "runtime_binding_sha256": "3" * 64,
        "current_binding_sha256": "4" * 64,
        "record_index": 0,
        "previous_record_sha256": "0" * 64,
        "record_sha256": "5" * 64,
        "training_authorized": False,
        "method_selection_status": "PENDING_EVIDENCE_NO_SELECTION",
        "parent_authority_secret_path": "/fixture/authority.secret",
        "parent_authority_secret_sha256": "6" * 64,
    }
    assert _schema_failures(schema, [record]) == []
    bad = copy.deepcopy(record)
    bad["record_index"] = "0"
    assert _schema_failures(schema, [bad])
    bad = copy.deepcopy(record)
    bad["record_index"] = False
    assert _schema_failures(schema, [bad])

    supervised = {
        **record,
        "record_type": "smsb_capture",
        "parent_credential_id": "7" * 64,
        "parent_credential_mac": "8" * 64,
        "parent_credential_sha256": "9" * 64,
        "parent_credential_path": "/fixture/credential.json",
        "process_pid": 123,
        "parent_receipt_path": "/fixture/receipt.json",
        "parent_receipt_sha256": "a" * 64,
        "parent_receipt_id": "b" * 64,
        "parent_receipt_mac": "c" * 64,
        "parent_launcher_pid": 122,
        "observed_child_ppid": 122,
        "child_exit_code": 0,
        "child_stdout_artifact": "/fixture/child.log",
        "child_stdout_artifact_sha256": "d" * 64,
        "pre_child_model_manifest_sha256": "e" * 64,
        "post_child_model_manifest_sha256": "e" * 64,
        "post_child_current_binding_sha256": "4" * 64,
        "pre_child_physical_gpu_identity_sha256": "f" * 64,
        "post_child_physical_gpu_identity_sha256": "f" * 64,
        "post_child_full_model_sha_verified": True,
    }
    assert _schema_failures(schema, [supervised]) == []
    missing_post = copy.deepcopy(supervised)
    missing_post.pop("post_child_model_manifest_sha256")
    assert _schema_failures(schema, [missing_post])
    false_post = copy.deepcopy(supervised)
    false_post["post_child_full_model_sha_verified"] = False
    assert _schema_failures(schema, [false_post])


def test_static_freeze_is_strict_vllm_no_training_and_conditionally_ordered() -> None:
    environment = {
        "MEMAGENT_SERIAL_CREDIT_WORK_ROOT": "/tmp/memagent-test",
        "MEMAGENT_SERIAL_CREDIT_REPO_DIR": str(REPO),
        "MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT": "a" * 40,
        "MEMAGENT_SERIAL_CREDIT_RUN_ID": "dynamic_pair_test",
        "MEMAGENT_SERIAL_CREDIT_GPU_PAIR": "4,7",
    }
    manifest = load_manifest(MANIFEST, environment)
    raw_commands = json.loads(COMMANDS.read_text(encoding="utf-8"))
    from tools.h20.preflight_qwen25_7b_serialization_credit import resolve_manifest_environment
    commands = resolve_manifest_environment(raw_commands, environment)
    assert manifest["gpu"]["physical_whitelist"] == [4, 7]
    assert manifest["gpu"]["visible_devices"] == "4,7"
    assert commands["execution"]["physical_gpus"] == [4, 7]
    assert commands["execution"]["visible_devices"] == "4,7"
    assert manifest["gpu"]["project_locks"] == [
        "/tmp/memagent-test/locks/memagent_h20_gpu_4.lock",
        "/tmp/memagent-test/locks/memagent_h20_gpu_7.lock",
    ]
    assert commands["branch"] == manifest["branch"]
    assert all(type(value) is int for value in manifest["gpu"]["physical_whitelist"])
    assert manifest["gpu"]["tensor_parallel_size"] == 2
    assert manifest["backend"]["required_version"] == "0.8.2"
    assert manifest["backend"]["strict_vllm"] is True
    assert manifest["backend"]["allow_huggingface_generation_fallback"] is False
    assert manifest["scope"]["training"] is False
    assert manifest["scope"]["training_authorized"] is False
    assert manifest["tetrad"]["requires_smsb_decision"] == "PASS_E_DET_SINGLE_REQUEST"
    assert commands["required_sequence"][4] == "tetrad_construct_if_and_only_if_smsb_pass"
    assert commands["execution"]["full_model_sha_per_fresh_child"] is True
    assert commands["execution"]["actual_gpu_uuid_name_bound_per_child"] is True
    assert commands["execution"]["parent_issued_single_use_credential_per_child"] is True
    assert commands["execution"]["parent_hmac_authenticated_receipt_per_child"] is True
    assert commands["execution"]["parent_observed_child_pid_ppid_exit_and_stdout_sha"] is True
    assert commands["execution"]["post_child_full_model_sha_per_child"] is True
    assert commands["execution"]["post_child_gpu_identity_query_per_child"] is True
    assert commands["execution"]["p0_and_per_call_token_capacity_gate"] is True
    assert commands["execution"]["writer_turn_token_chain_independently_rebuilt"] is True
    assert commands["execution"]["automatic_post_write_readonly_reaudit"] is True
    assert commands["execution"]["unique_parent_supervisor_pid_required"] is True
    assert commands["execution"]["unique_child_pid_required"] is True
    assert commands["execution"]["per_gpu_locking"] is True


@pytest.mark.parametrize("value", ["", "2", "2,2", "3,2", "02,3", "a,b"])
def test_dynamic_gpu_pair_rejects_noncanonical_or_nondistinct_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_gpu_pair(value)


def test_shell_and_audit_sources_encode_fresh_process_and_readonly_reaudit() -> None:
    tetrad_shell = (REPO / "scripts/h20/run_qwen25_7b_tetrad4.sh").read_text()
    smsb_shell = (REPO / "scripts/h20/run_qwen25_7b_smsb4.sh").read_text()
    common_shell = (REPO / "scripts/h20/serialization_credit_pilots_common.sh").read_text()
    audit_source = (REPO / "tools/h20/audit_qwen25_7b_serialization_credit.py").read_text()
    preflight_source = (
        REPO / "tools/h20/preflight_qwen25_7b_serialization_credit.py"
    ).read_text()
    launcher_source = (
        REPO / "tools/h20/launch_qwen25_7b_serialization_credit_child.py"
    ).read_text()
    launcher = "launch_qwen25_7b_serialization_credit_child.py"
    assert "list-tetrad-requests" in tetrad_shell
    assert "chr(9).join" not in tetrad_shell
    assert "run-tetrad-request" in tetrad_shell
    assert launcher in tetrad_shell
    assert launcher in smsb_shell
    assert "--issue-child-credential" not in preflight_source
    assert launcher_source.index("exit_code = process.wait()") < launcher_source.index(
        "post_child_current_binding_sha = verify_current_binding"
    )
    assert "full_model_sha=True" in launcher_source
    assert "post_child_gpu_identity = _physical_gpu_identity" in launcher_source
    assert "SERIAL_CREDIT_READONLY_REAUDIT" in tetrad_shell
    assert '--output "$SERIAL_CREDIT_READONLY_REAUDIT"' in tetrad_shell
    assert '>"$SERIAL_CREDIT_READONLY_REAUDIT"' not in tetrad_shell
    assert tetrad_shell.index("--write-report") < tetrad_shell.index(
        'assert report["ledger_prefix_record_count"] == 37'
    )
    assert 'assert report["failures"] == []' in tetrad_shell
    assert "serial_credit_wait_idle" in tetrad_shell
    assert "serial_credit_sanitize_inherited_environment" in common_shell
    assert "summarize_smsb_pilot" in audit_source
    assert "adjudicate_tetrad_pilot" in audit_source
    assert "len(records) not in (37, 38)" in audit_source
    assert "authoring_artifact_sha256" in audit_source
    assert "full_model_sha=True" in audit_source


def test_readonly_audit_output_is_not_stdout_capture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "readonly_reaudit.json"
    report = {
        "status": "PASS",
        "decision": "SERIALIZATION_CREDIT_PILOT4_PASS",
        "failures": [],
    }

    def noisy_audit(manifest: Path, *, write: bool) -> dict:
        assert manifest == manifest_path
        assert write is False
        print("INFO vLLM imported on CUDA platform")
        return report

    with patch.object(
        audit_module, "audit", side_effect=noisy_audit
    ), patch.object(
        audit_module,
        "load_manifest",
        return_value={"paths": {"readonly_reaudit": str(output)}},
    ), patch.object(
        sys,
        "argv",
        [
            "audit_qwen25_7b_serialization_credit.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
        ],
    ):
        assert audit_module.main() == 0
    assert "INFO vLLM imported" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with patch.object(
        audit_module, "load_manifest",
        return_value={"paths": {"readonly_reaudit": str(output)}},
    ), pytest.raises(FileExistsError):
        audit_module._write_readonly_audit_output(
            manifest_path, output, report
        )


def test_no_shared_core_or_sources_are_part_of_pilot_code_objects() -> None:
    source = (REPO / "tools/h20/preflight_qwen25_7b_serialization_credit.py").read_text()
    code_object_block = source.split("CODE_OBJECTS = (", 1)[1].split(")\n", 1)[0]
    assert "ray_trainer.py" not in code_object_block
    assert '"sources/' not in code_object_block
    assert "serialization_credit_pilots.py" in code_object_block

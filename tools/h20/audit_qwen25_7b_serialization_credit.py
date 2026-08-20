#!/usr/bin/env python3
"""Authenticate the complete SMSB4 -> Tetrad4 pilot evidence chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.gate_a_execution import validate_jsonl_chain  # noqa: E402
from recurrent.research.serialization_credit_pilots import (  # noqa: E402
    adjudicate_tetrad_pilot,
    best_length_derangement,
    build_tetrad_requests,
    canonical_json,
    canonical_sha256,
    content_words,
    read_jsonl,
    sha256_file,
    summarize_smsb_pilot,
    split_documents,
    validate_replay,
    validate_capture_record,
    validate_tetrad_manifest,
    validate_parent_launch_receipt,
    write_json_exclusive,
)
from tools.h20.preflight_qwen25_7b_serialization_credit import (  # noqa: E402
    MANIFEST_REL,
    load_manifest,
    load_parent_authority_secret,
    record_stage,
    validate_child_credential,
    validate_p0,
    verify_current_binding,
)


def _ledger_prefix_sha(path: Path, count: int) -> str:
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) < count:
        raise ValueError(f"ledger has only {len(lines)} lines; cannot bind prefix {count}")
    return hashlib.sha256(b"".join(lines[:count])).hexdigest()


def _schema_failures(schema_path: Path, records: list[dict[str, Any]]) -> list[str]:
    try:
        import jsonschema

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        )
        return [
            f"record {index} schema: {error.message}"
            for index, record in enumerate(records)
            for error in sorted(
                validator.iter_errors(record),
                key=lambda item: "/".join(str(part) for part in item.path),
            )
        ]
    except Exception as error:
        return [f"cannot strictly validate ledger schema: {error}"]


def _artifact(record: dict[str, Any], expected: Path) -> None:
    if Path(str(record.get("artifact", ""))).resolve() != expected.resolve():
        raise ValueError(
            f"ledger artifact path differs: {record.get('artifact')} != {expected}"
        )
    if not expected.is_file() or record.get("artifact_sha256") != sha256_file(expected):
        raise ValueError(f"ledger artifact digest differs: {expected}")


def _rebuild_authoring_from_s128(
    *,
    manifest: dict[str, Any],
    resolved: dict[str, Any],
    current_binding_sha: str,
    captures: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Rebuild authoring from immutable parquet/token IDs; persisted flags are ignored."""
    import pyarrow.parquet as parquet
    from recurrent.impls.memory import TEMPLATE_FINAL_BOXED
    from recurrent.utils import chat_template

    parquet_rows = parquet.read_table(
        manifest["data"]["validation"],
        columns=["prompt", "context", "reward_model", "extra_info"],
    ).to_pylist()
    if len(parquet_rows) != 128:
        raise ValueError("independent authoring rebuild requires exact S128")
    template_text = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
    template_hash = hashlib.sha256(template_text.encode("utf-8")).hexdigest()
    lengths = {
        str(row["example_id"]): len(row["final_memory_token_ids"])
        for row in captures
    }
    matching = best_length_derangement(
        lengths,
        maximum_caliper=manifest["tetrad"][
            "maximum_shuffle_memory_token_caliper"
        ],
    )
    capture_by_id = {str(row["example_id"]): row for row in captures}
    rebuilt: list[dict[str, Any]] = []
    for example_id in sorted(capture_by_id):
        capture = capture_by_id[example_id]
        raw_position = capture["raw_row_position"]
        if type(raw_position) is not int or not 0 <= raw_position < 128:
            raise ValueError(f"invalid raw S128 position for {example_id}")
        source = parquet_rows[raw_position]
        prompt = source.get("prompt")
        if (
            not isinstance(prompt, list)
            or len(prompt) != 1
            or prompt[0].get("role") != "user"
            or not isinstance(prompt[0].get("content"), str)
        ):
            raise ValueError(f"source prompt drift for {example_id}")
        question = prompt[0]["content"]
        context = source.get("context")
        raw_answers = source.get("reward_model", {}).get("ground_truth")
        if not isinstance(context, str):
            raise ValueError(f"source context is not text for {example_id}")
        if (
            not isinstance(raw_answers, list)
            or not raw_answers
            or any(not isinstance(value, str) or not value.strip() for value in raw_answers)
        ):
            raise ValueError(f"source ground truth missing for {example_id}")
        answers = list(raw_answers)
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != capture[
            "source_question_hash"
        ]:
            raise ValueError(f"source question hash drift for {example_id}")
        if hashlib.sha256(context.encode("utf-8")).hexdigest() != capture[
            "source_context_hash"
        ]:
            raise ValueError(f"source context hash drift for {example_id}")
        if canonical_sha256(raw_answers) != capture["ground_truth_hash"]:
            raise ValueError(f"source ground-truth hash drift for {example_id}")
        question_ids = tokenizer.encode(question, add_special_tokens=False)
        if list(question_ids) != capture["question_token_ids"]:
            raise ValueError(f"source question-token drift for {example_id}")
        answer = answers[0].strip().strip('"')
        answer_norms = [value.strip().strip('"').lower() for value in answers]
        if not answer or any(not value for value in answer_norms):
            raise ValueError(f"ground truth contains an empty answer for {example_id}")
        answer_norm = answer_norms[0]
        question_words = content_words(question)
        documents = split_documents(context)
        for document in documents:
            haystack = f"{document['title']} {document['text']}".lower()
            document["answer_hit"] = any(
                candidate in haystack for candidate in answer_norms
            )
            document["question_overlap"] = len(
                question_words & content_words(haystack)
            )
        answer_documents = sorted(
            (document for document in documents if document["answer_hit"]),
            key=lambda document: (-document["question_overlap"], document["number"]),
        )
        anchor_documents = sorted(
            (document for document in documents if not document["answer_hit"]),
            key=lambda document: (-document["question_overlap"], document["number"]),
        )
        selected_gold: list[dict[str, Any]] = []
        if answer_documents:
            selected_gold.append(answer_documents[0])
        if anchor_documents and anchor_documents[0]["question_overlap"] > 0:
            selected_gold.append(anchor_documents[0])
        evidence = "\n\n".join(
            f"{document['title']}: {document['text']}"
            for document in selected_gold
        )
        gold_text = f"Canonical answer: {answer}.\nCanonical evidence:\n{evidence}"
        gold_ids = tokenizer.encode(gold_text, add_special_tokens=False)[
            : lengths[example_id]
        ]
        if not gold_ids or answer_norm not in tokenizer.decode(
            gold_ids, skip_special_tokens=False
        ).lower():
            raise ValueError(f"gold answer/token budget audit failed for {example_id}")
        distractors = [
            document
            for document in documents
            if not document["answer_hit"]
            and document["question_overlap"] == 0
            and document["number"]
            not in {item["number"] for item in selected_gold}
        ]
        irrelevant_parts: list[str] = []
        irrelevant_titles: list[str] = []
        irrelevant_ids: list[int] = []
        for document in sorted(distractors, key=lambda item: item["number"]):
            irrelevant_titles.append(document["title"])
            irrelevant_parts.append(f"{document['title']}: {document['text']}")
            irrelevant_ids = tokenizer.encode(
                "\n\n".join(irrelevant_parts), add_special_tokens=False
            )
            if len(irrelevant_ids) >= lengths[example_id]:
                break
        if len(irrelevant_ids) < lengths[example_id]:
            raise ValueError(f"insufficient legal irrelevant tokens for {example_id}")
        irrelevant_ids = irrelevant_ids[: lengths[example_id]]
        irrelevant_text = tokenizer.decode(
            irrelevant_ids, skip_special_tokens=False
        ).lower()
        if any(
            candidate in irrelevant_text for candidate in answer_norms
        ) or question_words & content_words(irrelevant_text):
            raise ValueError(f"independent irrelevant leakage audit failed for {example_id}")
        rebuilt.append(
            {
                **{
                    field: capture[field]
                    for field in (
                        "example_id", "semantic_dataset_index", "source_order_index",
                        "raw_row_position", "production_effective_position",
                        "eval_manifest_hash", "source_question_hash",
                        "source_context_hash", "ground_truth_hash",
                    )
                },
                "question": question,
                "question_token_ids": list(question_ids),
                "ground_truth": answers,
                "question_type": "hotpot_multihop",
                "answer_type": "hotpot_short_span",
                "checkpoint_hash": capture["hashes"]["model"],
                "model_hash": capture["hashes"]["model"],
                "tokenizer_hash": capture["hashes"]["tokenizer"],
                "hashes": dict(capture["hashes"]),
                "vllm_version": capture["vllm_version"],
                "runtime_binding_sha256": resolved["runtime_binding_sha256"],
                "engine_config_sha256": resolved["execution_binding"][
                    "engine_config_sha256"
                ],
                "current_binding_sha256": current_binding_sha,
                "full_model_sha_verified_at_tetrad_start": True,
                "prompt_protocol_hash": template_hash,
                "prompt_outside_memory_span_hash": canonical_sha256(
                    {"template": template_hash, "question_token_ids": question_ids}
                ),
                "physical_gpu_identity": list(
                    capture["execution"]["physical_gpu_identity"]
                ),
                "cuda_device_order": capture["execution"]["cuda_device_order"],
                "generated": {
                    "state_id": f"{example_id}:generated",
                    "memory_token_ids": list(capture["final_memory_token_ids"]),
                    "validity_status": "pass",
                    "smsb_status": "pass",
                },
                "empty": {
                    "state_id": f"{example_id}:empty",
                    "memory_token_ids": [],
                    "validity_status": "pass",
                },
                "irrelevant": {
                    "state_id": f"{example_id}:within-example-distractor",
                    "memory_token_ids": irrelevant_ids,
                    "validity_status": "pass",
                    "support_answer_bridge_leakage_audit": "pass",
                    "length_match_audit": "pass",
                    "selected_document_titles": irrelevant_titles,
                    "audit_definition": "no_ground_truth_substring_and_zero_frozen_question_keyword_overlap",
                },
                "gold": {
                    "state_id": f"{example_id}:canonical-positive-control",
                    "memory_token_ids": gold_ids,
                    "validity_status": "pass",
                    "canonical_authoring_audit": "pass",
                    "source_mode": (
                        "answer_document_plus_question_anchor"
                        if answer_documents
                        else "answer_control_fallback"
                    ),
                    "selected_document_titles": [
                        item["title"] for item in selected_gold
                    ],
                    "contains_ground_truth_by_design": True,
                },
                "shuffle_approved_donor_ids": [matching[example_id]],
                "shuffle_memory_token_delta": abs(
                    lengths[example_id] - lengths[matching[example_id]]
                ),
                "generated_memory_token_length": lengths[example_id],
                "gold_memory_token_length": len(gold_ids),
                "irrelevant_memory_token_length": len(irrelevant_ids),
            }
        )
    return rebuilt, matching


def _authenticate_authoring_from_s128(
    persisted_authoring: list[dict[str, Any]],
    **rebuild_kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rebuilt, matching = _rebuild_authoring_from_s128(**rebuild_kwargs)
    if canonical_json(persisted_authoring) != canonical_json(rebuilt):
        raise ValueError(
            "persisted Tetrad authoring differs from independent S128 rebuild"
        )
    return rebuilt, matching


def _verify_parent_credential_record(
    *,
    record: dict[str, Any],
    result: dict[str, Any],
    child_kind: str,
    child_identity: str,
    manifest: dict[str, Any],
    resolved: dict[str, Any],
    current_binding_sha: str,
    artifact_payload: Any,
    authority_secret: bytes,
) -> str:
    path = Path(str(record.get("parent_credential_path", "")))
    if not path.is_file() or not path.resolve().is_relative_to(
        Path(manifest["paths"]["log_root"]).resolve()
    ):
        raise ValueError("parent credential path is missing or outside the run root")
    receipt_path = Path(str(record.get("parent_receipt_path", "")))
    if not receipt_path.is_file() or not receipt_path.resolve().is_relative_to(
        Path(manifest["paths"]["log_root"]).resolve()
    ):
        raise ValueError("parent receipt path is missing or outside the run root")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    credential_evidence = validate_child_credential(
        path,
        manifest=manifest,
        resolved=resolved,
        current_binding_sha=current_binding_sha,
        child_kind=child_kind,
        child_identity=child_identity,
        authority_secret=authority_secret,
        expected_issuer_pid=receipt.get("parent_launcher_pid"),
    )
    if any(record.get(key) != value for key, value in credential_evidence.items()):
        raise ValueError("ledger differs from authenticated parent credential")
    if any(result.get(key) != value for key, value in credential_evidence.items()):
        raise ValueError("child result differs from authenticated parent credential")
    validated_receipt = validate_parent_launch_receipt(
        receipt,
        authority_secret=authority_secret,
        artifact_payload=artifact_payload,
        child_evidence=result,
        child_kind=child_kind,
        child_identity=child_identity,
    )
    expected_record = {
        "process_pid": result.get("process_pid"),
        "parent_receipt_path": str(receipt_path.resolve()),
        "parent_receipt_sha256": sha256_file(receipt_path),
        "parent_receipt_id": validated_receipt["receipt_id"],
        "parent_receipt_mac": validated_receipt["receipt_mac"],
        "parent_launcher_pid": validated_receipt["parent_launcher_pid"],
        "observed_child_ppid": validated_receipt["observed_child_ppid"],
        "child_exit_code": 0,
        "child_stdout_artifact": validated_receipt["stdout_artifact"],
        "child_stdout_artifact_sha256": validated_receipt[
            "stdout_artifact_sha256"
        ],
    }
    if any(record.get(key) != value for key, value in expected_record.items()):
        raise ValueError("ledger differs from authenticated parent launch receipt")
    if (
        Path(validated_receipt["artifact"]).resolve()
        != Path(str(record.get("artifact"))).resolve()
        or validated_receipt["artifact_sha256"] != record.get("artifact_sha256")
        or not Path(validated_receipt["stdout_artifact"]).resolve().is_relative_to(
            Path(manifest["paths"]["log_root"]).resolve()
        )
        or not Path(validated_receipt["stdout_artifact"]).is_file()
        or sha256_file(validated_receipt["stdout_artifact"])
        != validated_receipt["stdout_artifact_sha256"]
        or validated_receipt["runner_code_sha256"]
        != sha256_file(
            Path(manifest["repository"])
            / "tools/h20/run_qwen25_7b_serialization_credit.py"
        )
    ):
        raise ValueError("parent receipt artifact/stdout/runner digest differs")
    return str(validated_receipt["receipt_id"])


def authenticate_smsb_gate(manifest_path: Path) -> dict[str, Any]:
    """Authenticate the exact 15-record SMSB prefix before Tetrad construction."""
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    current_binding_sha = verify_current_binding(
        manifest, resolved, full_model_sha=True
    )
    authority_secret = load_parent_authority_secret(manifest, resolved)
    paths = {
        key: Path(value)
        for key, value in manifest["paths"].items()
        if key != "log_root"
    }
    records = read_jsonl(paths["execution_ledger"])
    failures = list(validate_jsonl_chain(records))
    failures.extend(
        _schema_failures(
            Path(manifest["repository"]) / manifest["ledger_schema"], records
        )
    )
    expected_types = ["s0_preflight", "smsb_capture"] + ["smsb_replay"] * 12 + [
        "smsb_adjudication"
    ]
    if [row.get("record_type") for row in records] != expected_types:
        failures.append("SMSB ledger is not the exact authenticated 15-record prefix")
    for index, record in enumerate(records):
        if (
            record.get("git_commit")
            != os.environ["MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT"]
            or record.get("run_id") != manifest["run_id"]
            or record.get("eval_manifest_hash") != resolved["eval_manifest_hash"]
            or record.get("execution_binding_sha256")
            != resolved["execution_binding_sha256"]
            or record.get("runtime_binding_sha256")
            != resolved["runtime_binding_sha256"]
            or record.get("current_binding_sha256") != current_binding_sha
            or record.get("training_authorized") is not False
            or record.get("method_selection_status")
            != "PENDING_EVIDENCE_NO_SELECTION"
        ):
            failures.append(f"SMSB ledger record {index} binding differs")
    captures: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    smsb_report: dict[str, Any] = {}
    try:
        if len(records) != 15:
            raise ValueError("SMSB ledger record count is not 15")
        _artifact(records[1], paths["smsb_captures"])
        captures = read_jsonl(paths["smsb_captures"])
        if len(captures) != 4:
            raise ValueError("SMSB capture count is not four")
        for capture in captures:
            validate_capture_record(capture)
        capture_receipt_id = _verify_parent_credential_record(
            record=records[1],
            result=captures[0]["execution"],
            child_kind="smsb_capture",
            child_identity="capture4",
            manifest=manifest,
            resolved=resolved,
            current_binding_sha=current_binding_sha,
            artifact_payload=captures,
            authority_secret=authority_secret,
        )
        replay_paths = sorted(paths["smsb_replays"].glob("*.json"))
        credential_paths = sorted(paths["smsb_root"].joinpath("credentials").glob("*.json"))
        receipt_paths = sorted(paths["smsb_receipts"].glob("*.json"))
        if (
            len(replay_paths) != 12
            or len(credential_paths) != 13
            or len(receipt_paths) != 13
        ):
            raise ValueError("SMSB replay/credential/receipt count is not 12/13/13")
        if {
            Path(str(record.get("artifact", ""))).resolve()
            for record in records[2:14]
        } != {path.resolve() for path in replay_paths}:
            raise ValueError("SMSB replay ledger/artifact path bijection failed")
        credential_ids: set[str] = set()
        child_pids: set[int] = set()
        parent_launcher_pids = {records[1].get("parent_launcher_pid")}
        expected_keys = {
            (str(capture["example_id"]), regime)
            for capture in captures
            for regime in ("temperature_zero", "matched_seed", "independent_seed")
        }
        observed_keys: set[tuple[str, str]] = set()
        for record in records[2:14]:
            artifact = Path(str(record["artifact"]))
            _artifact(record, artifact)
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            capture_matches = [
                capture
                for capture in captures
                if capture["capture_id"] == payload.get("capture_id")
            ]
            if len(capture_matches) != 1:
                raise ValueError("SMSB replay capture reference is not unique")
            recomputed = validate_replay(
                capture_matches[0], payload["request"], payload["result"]
            )
            if canonical_json(recomputed) != canonical_json(payload.get("validation")):
                raise ValueError("SMSB persisted validation differs from recomputation")
            key = (str(recomputed["example_id"]), str(recomputed["regime"]))
            if (record.get("example_id"), record.get("regime")) != key:
                raise ValueError("SMSB ledger replay key differs")
            credential_ids.add(
                _verify_parent_credential_record(
                    record=record,
                    result=payload["result"],
                    child_kind="smsb_replay",
                    child_identity=f"{key[0]}::{key[1]}",
                    manifest=manifest,
                    resolved=resolved,
                    current_binding_sha=current_binding_sha,
                    artifact_payload=payload,
                    authority_secret=authority_secret,
                )
            )
            child_pids.add(payload["result"]["process_pid"])
            parent_launcher_pids.add(record.get("parent_launcher_pid"))
            observed_keys.add(key)
            payloads.append(payload)
        if (
            observed_keys != expected_keys
            or len(credential_ids | {capture_receipt_id}) != 13
            or len(child_pids) != 12
            or len(parent_launcher_pids) != 13
            or any(type(value) is not int or value < 1 for value in parent_launcher_pids)
            or {capture["execution"]["process_pid"] for capture in captures}
            & child_pids
        ):
            raise ValueError("SMSB replay key/PID/credential uniqueness gate failed")
        if {
            Path(str(record["parent_credential_path"])).resolve()
            for record in records[1:14]
        } != {path.resolve() for path in credential_paths}:
            raise ValueError("SMSB credential ledger/artifact path bijection failed")
        if {
            Path(str(record["parent_receipt_path"])).resolve()
            for record in records[1:14]
        } != {path.resolve() for path in receipt_paths}:
            raise ValueError("SMSB receipt ledger/artifact path bijection failed")
        _artifact(records[14], paths["smsb_report"])
        smsb_report = json.loads(paths["smsb_report"].read_text(encoding="utf-8"))
        receipt_rows = [
            json.loads(path.read_text(encoding="utf-8")) for path in receipt_paths
        ]
        capture_receipts = [
            row for row in receipt_rows if row.get("child_kind") == "smsb_capture"
        ]
        replay_receipts = [
            row for row in receipt_rows if row.get("child_kind") == "smsb_replay"
        ]
        recomputed_report = summarize_smsb_pilot(
            captures,
            payloads,
            expected_examples=4,
            capture_receipt=(capture_receipts[0] if len(capture_receipts) == 1 else None),
            replay_receipts=replay_receipts,
            authority_secret=authority_secret,
        )
        recomputed_report.update(
            capture_sha256=sha256_file(paths["smsb_captures"]),
            replay_artifact_sha256={
                path.name: sha256_file(path) for path in replay_paths
            },
            parent_receipt_artifact_sha256={
                path.name: sha256_file(path) for path in receipt_paths
            },
            training_authorized=False,
            method_selection_status="PENDING_EVIDENCE_NO_SELECTION",
            current_binding_sha256=current_binding_sha,
            runtime_binding_sha256=resolved["runtime_binding_sha256"],
            execution_binding_sha256=resolved["execution_binding_sha256"],
        )
        if canonical_json(smsb_report) != canonical_json(recomputed_report):
            raise ValueError("SMSB report differs from independent recomputation")
        if (
            smsb_report.get("status") != "PASS"
            or smsb_report.get("decision") != "PASS_E_DET_SINGLE_REQUEST"
            or smsb_report.get("E_det_pass") is not True
        ):
            raise ValueError("SMSB E_det decision is not an authenticated PASS")
    except Exception as error:
        failures.append(f"SMSB artifact authentication failed: {error}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "decision": (
            "SMSB_AUTHENTICATED_GATE_PASS"
            if not failures
            else "SERIALIZATION_CREDIT_NO_GO:SMSB_AUTHENTICATION"
        ),
        "failures": failures,
        "current_binding_sha256": current_binding_sha,
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "training_authorized": False,
        "method_selection_status": "PENDING_EVIDENCE_NO_SELECTION",
    }


def audit(manifest_path: Path, *, write: bool) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    _, resolved = validate_p0(manifest_path)
    current_binding_sha = verify_current_binding(
        manifest, resolved, full_model_sha=True
    )
    authority_secret = load_parent_authority_secret(manifest, resolved)
    paths = {key: Path(value) for key, value in manifest["paths"].items() if key != "log_root"}
    ledger_path = paths["execution_ledger"]
    records = read_jsonl(ledger_path)
    failures: list[str] = list(validate_jsonl_chain(records))
    failures.extend(
        _schema_failures(Path(manifest["repository"]) / manifest["ledger_schema"], records)
    )
    expected_commit = os.environ["MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT"]
    for index, record in enumerate(records):
        if record.get("git_commit") != expected_commit:
            failures.append(f"record {index} Git commit differs")
        if record.get("run_id") != manifest["run_id"]:
            failures.append(f"record {index} run_id differs")
        if record.get("eval_manifest_hash") != resolved["eval_manifest_hash"]:
            failures.append(f"record {index} eval manifest hash differs")
        if record.get("execution_binding_sha256") != resolved["execution_binding_sha256"]:
            failures.append(f"record {index} execution binding differs")
        if record.get("runtime_binding_sha256") != resolved["runtime_binding_sha256"]:
            failures.append(f"record {index} runtime binding differs")
        if record.get("current_binding_sha256") != current_binding_sha:
            failures.append(f"record {index} current binding differs")
        if record.get("training_authorized") is not False:
            failures.append(f"record {index} improperly authorizes training")
        if record.get("method_selection_status") != "PENDING_EVIDENCE_NO_SELECTION":
            failures.append(f"record {index} improperly selects a method")
    core_types = (
        ["s0_preflight", "smsb_capture"]
        + ["smsb_replay"] * 12
        + ["smsb_adjudication", "tetrad_construct"]
        + ["tetrad_replay"] * 20
        + ["tetrad_adjudication"]
    )
    expected_types = core_types + (["audit_result"] if len(records) == 38 else [])
    if len(records) not in (37, 38) or [record.get("record_type") for record in records] != expected_types:
        failures.append(
            "ledger sequence is not exactly the 37-record pilot prefix with an optional authenticated audit tail"
        )
    if write and len(records) != 37:
        failures.append("--write-report requires the exact 37-record pre-audit prefix")

    captures: list[dict[str, Any]] = []
    smsb_report: dict[str, Any] = {}
    tetrad_rows: list[dict[str, Any]] = []
    authoring_rows: list[dict[str, Any]] = []
    tetrad_report: dict[str, Any] = {}
    replay_payloads: list[dict[str, Any]] = []
    tetrad_results: list[dict[str, Any]] = []
    if len(records) >= 37:
        try:
            _artifact(records[1], paths["smsb_captures"])
            captures = read_jsonl(paths["smsb_captures"])
            if len(captures) != 4 or len({row["example_id"] for row in captures}) != 4:
                failures.append("SMSB capture artifact is not four unique examples")
            pilot_ids = {str(row["example_id"]) for row in resolved["pilot_rows"]}
            if {str(row["example_id"]) for row in captures} != pilot_ids:
                failures.append("SMSB captures differ from P0-selected pilot IDs")
            pilot_by_id = {str(row["example_id"]): row for row in resolved["pilot_rows"]}
            for capture in captures:
                validate_capture_record(capture)
                pilot = pilot_by_id[str(capture["example_id"])]
                for field in (
                    "semantic_dataset_index", "source_order_index", "raw_row_position",
                    "production_effective_position", "eval_manifest_hash",
                    "source_question_hash", "source_context_hash", "ground_truth_hash",
                    "trajectory_seed",
                ):
                    if capture.get(field) != pilot.get(field):
                        failures.append(f"SMSB capture {capture.get('example_id')} P0 field differs: {field}")
                if capture.get("current_binding_sha256") != current_binding_sha:
                    failures.append("SMSB capture current binding differs")
            _verify_parent_credential_record(
                record=records[1],
                result=captures[0]["execution"],
                child_kind="smsb_capture",
                child_identity="capture4",
                manifest=manifest,
                resolved=resolved,
                current_binding_sha=current_binding_sha,
                artifact_payload=captures,
                authority_secret=authority_secret,
            )
            replay_records = records[2:14]
            replay_paths = sorted(paths["smsb_replays"].glob("*.json"))
            smsb_credential_paths = sorted(paths["smsb_root"].joinpath("credentials").glob("*.json"))
            smsb_receipt_paths = sorted(paths["smsb_receipts"].glob("*.json"))
            if len(replay_paths) != 12:
                failures.append("SMSB replay directory does not contain exactly 12 JSON files")
            if len(smsb_credential_paths) != 13:
                failures.append("SMSB credential directory does not contain exactly 13 JSON files")
            if len(smsb_receipt_paths) != 13:
                failures.append("SMSB receipt directory does not contain exactly 13 JSON files")
            if {
                Path(str(record.get("artifact", ""))).resolve()
                for record in replay_records
            } != {path.resolve() for path in replay_paths}:
                failures.append("SMSB replay ledger paths differ from the exact replay directory")
            for record in replay_records:
                artifact = Path(str(record["artifact"]))
                _artifact(record, artifact)
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                replay_payloads.append(payload)
                validation = payload["validation"]
                if record.get("example_id") != validation.get("example_id"):
                    failures.append("SMSB replay ledger example differs from validation")
                if record.get("regime") != validation.get("regime"):
                    failures.append("SMSB replay ledger regime differs from validation")
                capture = next(
                    row for row in captures if row["capture_id"] == payload.get("capture_id")
                )
                recomputed_validation = validate_replay(
                    capture, payload["request"], payload["result"]
                )
                if canonical_json(validation) != canonical_json(recomputed_validation):
                    failures.append("persisted SMSB validation differs from independent recomputation")
                _verify_parent_credential_record(
                    record=record,
                    result=payload["result"],
                    child_kind="smsb_replay",
                    child_identity=f"{validation.get('example_id')}::{validation.get('regime')}",
                    manifest=manifest,
                    resolved=resolved,
                    current_binding_sha=current_binding_sha,
                    artifact_payload=payload,
                    authority_secret=authority_secret,
                )
            if {
                Path(str(record.get("parent_credential_path", ""))).resolve()
                for record in records[1:14]
            } != {path.resolve() for path in smsb_credential_paths}:
                failures.append("SMSB credential ledger/artifact path bijection failed")
            if {
                Path(str(record.get("parent_receipt_path", ""))).resolve()
                for record in records[1:14]
            } != {path.resolve() for path in smsb_receipt_paths}:
                failures.append("SMSB receipt ledger/artifact path bijection failed")
            expected_replay_keys = {
                (example_id, regime)
                for example_id in pilot_ids
                for regime in ("temperature_zero", "matched_seed", "independent_seed")
            }
            if {(row.get("example_id"), row.get("regime")) for row in replay_records} != expected_replay_keys:
                failures.append("SMSB replay ledger lacks exact example/regime coverage")
            _artifact(records[14], paths["smsb_report"])
            smsb_report = json.loads(paths["smsb_report"].read_text(encoding="utf-8"))
            smsb_receipts = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in smsb_receipt_paths
            ]
            capture_receipts = [
                row for row in smsb_receipts if row.get("child_kind") == "smsb_capture"
            ]
            recomputed_smsb = summarize_smsb_pilot(
                captures,
                replay_payloads,
                expected_examples=4,
                capture_receipt=(capture_receipts[0] if len(capture_receipts) == 1 else None),
                replay_receipts=[
                    row for row in smsb_receipts if row.get("child_kind") == "smsb_replay"
                ],
                authority_secret=authority_secret,
            )
            recomputed_smsb.update(
                capture_sha256=sha256_file(paths["smsb_captures"]),
                replay_artifact_sha256={
                    path.name: sha256_file(path) for path in replay_paths
                },
                parent_receipt_artifact_sha256={
                    path.name: sha256_file(path) for path in smsb_receipt_paths
                },
                training_authorized=False,
                method_selection_status="PENDING_EVIDENCE_NO_SELECTION",
                current_binding_sha256=current_binding_sha,
                runtime_binding_sha256=resolved["runtime_binding_sha256"],
                execution_binding_sha256=resolved["execution_binding_sha256"],
            )
            if canonical_json(smsb_report) != canonical_json(recomputed_smsb):
                failures.append("persisted SMSB report differs from independent recomputation")
            if (
                smsb_report.get("status") != "PASS"
                or smsb_report.get("decision") != "PASS_E_DET_SINGLE_REQUEST"
                or smsb_report.get("E_det_pass") is not True
            ):
                failures.append("SMSB E_det did not pass")
            _artifact(records[15], paths["tetrad_manifest"])
            if not paths["tetrad_authoring"].is_file():
                failures.append("Tetrad authoring artifact is missing")
            else:
                authoring_rows = read_jsonl(paths["tetrad_authoring"])
            if (
                records[15].get("authoring_artifact")
                != str(paths["tetrad_authoring"].resolve())
                or records[15].get("authoring_artifact_sha256")
                != sha256_file(paths["tetrad_authoring"])
            ):
                failures.append("Tetrad authoring is not bound by the construct ledger record")
            tetrad_rows = read_jsonl(paths["tetrad_manifest"])
            validate_tetrad_manifest(tetrad_rows)
            from transformers import AutoTokenizer
            from recurrent.impls.memory import TEMPLATE_FINAL_BOXED
            from recurrent.utils import TokenTemplate, chat_template

            tokenizer = AutoTokenizer.from_pretrained(
                manifest["model"]["path"], trust_remote_code=True, local_files_only=True
            )
            independently_rebuilt_authoring, expected_matching = (
                _authenticate_authoring_from_s128(
                    authoring_rows,
                    manifest=manifest,
                    resolved=resolved,
                    current_binding_sha=current_binding_sha,
                    captures=captures,
                    tokenizer=tokenizer,
                )
            )
            template_text = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
            template = TokenTemplate(template_text, tokenizer)
            matching = {
                str(row["example_id"]): str(row["memory_source_example_id"])
                for row in tetrad_rows
                if row["state_role"] == "shuffle"
            }
            if matching != expected_matching:
                failures.append(
                    "persisted Tetrad shuffle matching differs from independent optimal derangement"
                )
            independently_rebuilt_tetrad = build_tetrad_requests(
                authoring_rows,
                matching=matching,
                base_seed=manifest["backend"]["engine_seed"],
                prompt_builder=lambda question, memory: template.format(
                    prompt=question, memory=memory
                ).tolist(),
                prompt_template_sha256=hashlib.sha256(
                    template_text.encode("utf-8")
                ).hexdigest(),
                capture_prompt_ids={
                    str(row["example_id"]): row["final_prompt_token_ids"]
                    for row in captures
                },
            )
            if canonical_json(tetrad_rows) != canonical_json(independently_rebuilt_tetrad):
                failures.append("persisted Tetrad request manifest differs from independent rebuild")
            tetrad_replay_records = records[16:36]
            result_paths = sorted(paths["tetrad_results"].glob("*.json"))
            tetrad_credential_paths = sorted(
                paths["tetrad_root"].joinpath("credentials").glob("*.json")
            )
            tetrad_receipt_paths = sorted(paths["tetrad_receipts"].glob("*.json"))
            if len(result_paths) != 20:
                failures.append("Tetrad result directory does not contain exactly 20 JSON files")
            if len(tetrad_credential_paths) != 20:
                failures.append("Tetrad credential directory does not contain exactly 20 JSON files")
            if len(tetrad_receipt_paths) != 20:
                failures.append("Tetrad receipt directory does not contain exactly 20 JSON files")
            if {
                Path(str(record.get("artifact", ""))).resolve()
                for record in tetrad_replay_records
            } != {path.resolve() for path in result_paths}:
                failures.append("Tetrad ledger paths differ from the exact result directory")
            expected_request_roles = {
                (str(row["request_id"]), str(row["state_role"])) for row in tetrad_rows
            }
            observed_request_roles: set[tuple[str, str]] = set()
            for record in tetrad_replay_records:
                artifact = Path(str(record["artifact"]))
                _artifact(record, artifact)
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                tetrad_results.append(payload)
                observed_request_roles.add(
                    (str(record.get("request_id")), str(record.get("state_role")))
                )
                if (
                    record.get("request_id") != payload.get("request_id")
                    or record.get("state_role") != payload.get("state_role")
                    or record.get("example_id") != payload.get("example_id")
                ):
                    failures.append("Tetrad replay ledger identity differs from result")
                _verify_parent_credential_record(
                    record=record,
                    result=payload,
                    child_kind="tetrad_replay",
                    child_identity=str(payload.get("request_id")),
                    manifest=manifest,
                    resolved=resolved,
                    current_binding_sha=current_binding_sha,
                    artifact_payload=payload,
                    authority_secret=authority_secret,
                )
            if {
                Path(str(record.get("parent_credential_path", ""))).resolve()
                for record in tetrad_replay_records
            } != {path.resolve() for path in tetrad_credential_paths}:
                failures.append("Tetrad credential ledger/artifact path bijection failed")
            if {
                Path(str(record.get("parent_receipt_path", ""))).resolve()
                for record in tetrad_replay_records
            } != {path.resolve() for path in tetrad_receipt_paths}:
                failures.append("Tetrad receipt ledger/artifact path bijection failed")
            if observed_request_roles != expected_request_roles:
                failures.append("Tetrad replay ledger lacks exact request/role coverage")
            _artifact(records[36], paths["tetrad_report"])
            tetrad_report = json.loads(paths["tetrad_report"].read_text(encoding="utf-8"))
            recomputed_tetrad = adjudicate_tetrad_pilot(
                tetrad_rows,
                authoring_rows,
                tetrad_results,
                answer_decoder=lambda ids: tokenizer.decode(
                    ids, skip_special_tokens=False
                ),
                parent_receipts=[
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in tetrad_receipt_paths
                ],
                authority_secret=authority_secret,
                competence_score_threshold=float(
                    manifest["tetrad"]["canonical_competence_score_threshold_f1"]
                ),
                competence_rate_floor=float(
                    manifest["tetrad"]["canonical_competence_rate_floor"]
                ),
            )
            recomputed_tetrad.update(
                manifest_sha256=sha256_file(paths["tetrad_manifest"]),
                authoring_sha256=sha256_file(paths["tetrad_authoring"]),
                result_artifact_sha256={
                    path.name: sha256_file(path) for path in result_paths
                },
                parent_receipt_artifact_sha256={
                    path.name: sha256_file(path) for path in tetrad_receipt_paths
                },
                current_binding_sha256=current_binding_sha,
                runtime_binding_sha256=resolved["runtime_binding_sha256"],
                execution_binding_sha256=resolved["execution_binding_sha256"],
            )
            if canonical_json(tetrad_report) != canonical_json(recomputed_tetrad):
                failures.append("persisted Tetrad report differs from independent recomputation")
            if (
                tetrad_report.get("status") != "PASS"
                or tetrad_report.get("decision") != "TETRAD_PILOT4_CONSTRUCTION_GATE_PASS"
            ):
                failures.append("Tetrad4 construction gate did not pass")
            if tetrad_report.get("effects_reportable") is not False:
                failures.append("Tetrad4 improperly marks effects reportable")
            if tetrad_report.get("training_authorized") is not False:
                failures.append("Tetrad4 improperly authorizes training")
            capture_processes = {
                row.get("execution", {}).get("process_instance_uuid") for row in captures
            }
            replay_processes = {
                row.get("result", {}).get("process_instance_uuid") for row in replay_payloads
            }
            tetrad_processes = {
                row.get("process_instance_uuid") for row in tetrad_results
            }
            if (
                len(capture_processes) != 1
                or len(replay_processes) != 12
                or len(tetrad_processes) != 20
                or capture_processes & replay_processes
                or capture_processes & tetrad_processes
                or replay_processes & tetrad_processes
            ):
                failures.append("capture/replay/Tetrad Python process instances are not disjoint")
            capture_pids = {
                row.get("execution", {}).get("process_pid") for row in captures
            }
            replay_pids = {
                row.get("result", {}).get("process_pid") for row in replay_payloads
            }
            tetrad_pids = {row.get("process_pid") for row in tetrad_results}
            replay_credentials = {
                row.get("result", {}).get("parent_credential_id")
                for row in replay_payloads
            }
            tetrad_credentials = {
                row.get("parent_credential_id") for row in tetrad_results
            }
            supervised_records = records[1:14] + records[16:36]
            parent_launcher_pids = {
                row.get("parent_launcher_pid") for row in supervised_records
            }
            parent_receipt_ids = {
                row.get("parent_receipt_id") for row in supervised_records
            }
            if (
                len(capture_pids) != 1
                or len(replay_pids) != 12
                or len(tetrad_pids) != 20
                or capture_pids & replay_pids
                or capture_pids & tetrad_pids
                or replay_pids & tetrad_pids
                or len(replay_credentials) != 12
                or len(tetrad_credentials) != 20
                or replay_credentials & tetrad_credentials
                or len(parent_launcher_pids) != 33
                or any(
                    type(value) is not int or value < 1
                    for value in parent_launcher_pids
                )
                or len(parent_receipt_ids) != 33
            ):
                failures.append(
                    "capture/replay/Tetrad PID, supervisor, receipt, or credential uniqueness failed"
                )
        except Exception as error:
            failures.append(f"artifact authentication failed: {error}")

    prefix_sha = _ledger_prefix_sha(ledger_path, 37) if len(records) >= 37 else None
    passed = not failures
    report = {
        "schema": "memagent.serialization-credit.pilot4.final-report.v1",
        "status": "PASS" if passed else "FAIL",
        "decision": "SERIALIZATION_CREDIT_PILOT4_PASS" if passed else "SERIALIZATION_CREDIT_PILOT4_NO_GO",
        "gates": {
            "P0": "PASS",
            "SMSB_E_det": "PASS" if smsb_report.get("E_det_pass") is True else "FAIL",
            "Tetrad4_construction": (
                "PASS"
                if tetrad_report.get("decision") == "TETRAD_PILOT4_CONSTRUCTION_GATE_PASS"
                else "FAIL"
            ),
        },
        "pilot_example_ids": [str(row["example_id"]) for row in resolved["pilot_rows"]],
        "smsb": {
            "decision": smsb_report.get("decision"),
            "L2_exact_count_report_only": smsb_report.get("L2_exact_count_report_only"),
            "L2_total_report_only": smsb_report.get("L2_total_report_only"),
            "claim_boundary": smsb_report.get("claim_boundary"),
        },
        "tetrad": {
            "decision": tetrad_report.get("decision"),
            "canonical_competent_count": tetrad_report.get("canonical_competent_count"),
            "canonical_competence_rate": tetrad_report.get("canonical_competence_rate"),
            "effects_reportable": False,
            "claim_boundary": "construction_and_execution_only_no_utilization_effect",
        },
        "ledger_prefix_record_count": len(records),
        "ledger_prefix_sha256": prefix_sha,
        "audit_result_tail_expected": True,
        "current_binding_sha256": current_binding_sha,
        "runtime_binding_sha256": resolved["runtime_binding_sha256"],
        "execution_binding_sha256": resolved["execution_binding_sha256"],
        "full_model_sha_verified_at_final_audit": True,
        "model_manifest_sha256": resolved["execution_binding"]["model_manifest_sha256"],
        "artifact_sha256": {
            "p0": sha256_file(paths["p0_certificate"]),
            "resolved_manifest": sha256_file(paths["resolved_manifest"]),
            "smsb_captures": sha256_file(paths["smsb_captures"]) if paths["smsb_captures"].is_file() else None,
            "smsb_report": sha256_file(paths["smsb_report"]) if paths["smsb_report"].is_file() else None,
            "tetrad_authoring": sha256_file(paths["tetrad_authoring"]) if paths["tetrad_authoring"].is_file() else None,
            "tetrad_manifest": sha256_file(paths["tetrad_manifest"]) if paths["tetrad_manifest"].is_file() else None,
            "tetrad_report": sha256_file(paths["tetrad_report"]) if paths["tetrad_report"].is_file() else None,
        },
        "scope": {
            "checkpoint": "base recurrent-I Qwen2.5-7B-Instruct",
            "dataset": "existing fixed HotpotQA S128 only",
            "examples": 4,
            "training_updates": 0,
            "paper_performance_result": False,
            "audit32_status": "NOT_RUN_BY_THIS_PILOT",
            "method_selection_status": "PENDING_EVIDENCE_NO_SELECTION",
            "training_authorized": False,
        },
        "failures": failures,
    }
    report["ledger_prefix_record_count"] = 37
    if len(records) == 38:
        tail = records[37]
        try:
            _artifact(tail, paths["final_report"])
            persisted = json.loads(paths["final_report"].read_text(encoding="utf-8"))
            if canonical_json(persisted) != canonical_json(report):
                failures.append("persisted final report differs from read-only independent re-audit")
            if (
                tail.get("status") != persisted.get("status")
                or tail.get("decision") != persisted.get("decision")
            ):
                failures.append("audit-result tail status/decision differs from final report")
        except Exception as error:
            failures.append(f"audit-result tail authentication failed: {error}")
        if failures:
            report["status"] = "FAIL"
            report["decision"] = "SERIALIZATION_CREDIT_PILOT4_NO_GO"
            report["failures"] = failures
    if write:
        if failures:
            return report
        output = paths["final_report"]
        if output.exists():
            raise FileExistsError("refuse to overwrite the final pilot report")
        write_json_exclusive(output, report)
        record_stage(
            manifest_path,
            record_type="audit_result",
            artifact=output,
            example_id=None,
            regime=None,
            request_id=None,
            state_role=None,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / MANIFEST_REL)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--smsb-gate-only", action="store_true")
    args = parser.parse_args()
    if args.smsb_gate_only and args.write_report:
        parser.error("--smsb-gate-only cannot be combined with --write-report")
    report = (
        authenticate_smsb_gate(args.manifest)
        if args.smsb_gate_only
        else audit(args.manifest, write=args.write_report)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

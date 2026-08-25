#!/usr/bin/env python3
"""Authority preparation, finalization, and replay for MIC-v2 Lbar_ref."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from recurrent.research.mic_v2 import (
    CONTRACT_SHA256,
    MATERIALIZATION_PARSER_VERSION,
    canonical_json,
    materialized_memory_receipt,
    sampled_policy_mask_receipt,
    sha256_file,
    sha256_json,
    write_json_new,
)
from recurrent.research.trajectory_seeding import derive_turn_request_seeds
from recurrent.research.serialization_credit_pilots import center_truncate_token_ids


MANIFEST_REL = Path("manifests/h20/qwen25_7b_mic_v2_reference_length_calibration.json")
RUNNER_REL = Path("tools/h20/run_qwen25_7b_mic_v2_reference_length_calibration.py")
CORE_REL = Path("tools/h20/mic_v2_reference_length_calibration.py")
MIC_V2_CORE_REL = Path("recurrent/research/mic_v2.py")
MEMORY_AGENT_REL = Path("recurrent/impls/memory.py")
RECURRENT_UTILS_REL = Path("recurrent/utils.py")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_GPU_INPUT_FIELDS = {
    "answer", "gold", "ground_truth", "ground_truth_sha256", "label", "reward",
    "score", "s128", "token_f1", "exact_match", "full_example_id",
}
MAX_SEED = (1 << 63) - 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MIC_V2_NO_GO: {message}")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def stable_trajectory_seed(base_seed: int, content_root_id: str, replica: int) -> int:
    payload = canonical_json([
        "memagent-mic-v2-reference-length-trajectory-v1",
        base_seed,
        content_root_id,
        replica,
    ]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % MAX_SEED


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON authority is not an object: {path}")
    return value


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    digest = value.get(field)
    _require(isinstance(digest, str) and HEX64_RE.fullmatch(digest) is not None,
             f"missing canonical digest {field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    _require(digest == sha256_json(unsigned), f"canonical digest differs: {field}")
    return digest


def _finite(value: Any, path: str = "payload") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"non-finite value at {path}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def _manifest(repo: Path) -> tuple[dict[str, Any], Path, str]:
    path = repo / MANIFEST_REL
    value = _load(path)
    _require(value.get("schema") == "memagent.mic.v2.reference-length-calibration-preregistration"
             and value.get("status") == "FROZEN_BEFORE_CALIBRATION_OUTPUTS"
             and value.get("scientific_contract_sha256") == CONTRACT_SHA256,
             "calibration preregistration drifted")
    return value, path, sha256_file(path)


def _runtime(repo: Path, expected_commit: str, output_root: Path, run_id: str) -> None:
    _require(sys.flags.optimize == 0, "optimized Python is forbidden")
    _require(repo.is_absolute() and output_root.is_absolute(), "paths must be absolute")
    _require(RUN_ID_RE.fullmatch(run_id) is not None, "unsafe calibration run ID")
    _require(_git(repo, "rev-parse", "HEAD") == expected_commit, "exact Git commit mismatch")
    _require(not _git(repo, "status", "--porcelain"), "worktree is dirty")
    _require(sha256_file(repo / "docs/papers/mic_v2_scientific_contract_20260825.md")
             == CONTRACT_SHA256, "scientific contract differs")


def _verify_data_freeze(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = manifest["data_freeze_authority"]
    root = Path(authority["root"])
    certificate_path = root / "certificates/data_freeze.json"
    resolved_path = root / "resolved_split_manifest.json"
    _require(sha256_file(certificate_path) == authority["certificate_file_sha256"],
             "data-freeze certificate file SHA differs")
    _require(sha256_file(resolved_path) == authority["resolved_file_sha256"],
             "resolved data file SHA differs")
    certificate = _load(certificate_path)
    resolved = _load(resolved_path)
    _require(_self_digest(certificate, "certificate_sha256")
             == authority["certificate_canonical_sha256"],
             "data-freeze certificate canonical SHA differs")
    _require(_self_digest(resolved, "resolved_manifest_sha256")
             == authority["resolved_canonical_sha256"],
             "resolved data canonical SHA differs")
    _require(certificate.get("status") == "PASS"
             and certificate.get("decision") == "MIC_V2_DATA_FREEZE_PASS"
             and certificate.get("run_id") == authority["run_id"],
             "data-freeze authority is not PASS")
    return certificate, resolved


def _label_blind_inputs(manifest: Mapping[str, Any], resolved: Mapping[str, Any]) -> dict[str, Any]:
    authority = manifest["data_freeze_authority"]
    split = resolved.get("splits", {}).get(authority["split"], {})
    rows = split.get("rows", [])
    _require(len(rows) == authority["content_roots"] == 64,
             "reference-length split coverage differs")
    projected = []
    for row in rows:
        item = {
            "source_position": row.get("source_position"),
            "semantic_dataset_index": row.get("semantic_dataset_index"),
            "question_sha256": row.get("question_sha256"),
            "context_sha256": row.get("context_sha256"),
            "content_root_id": row.get("content_root_id"),
        }
        _require(type(item["source_position"]) is int
                 and type(item["semantic_dataset_index"]) is int,
                 "label-blind row integer identity differs")
        _require(all(isinstance(item[key], str) and HEX64_RE.fullmatch(item[key])
                     for key in ("question_sha256", "context_sha256", "content_root_id")),
                 "label-blind row hash identity differs")
        projected.append(item)
    _require(len({row["content_root_id"] for row in projected}) == 64,
             "reference-length roots are not unique")
    value = {
        "schema": "memagent.mic.v2.reference-length-label-blind-inputs",
        "status": "FROZEN",
        "source_path": manifest["source"]["path"],
        "source_sha256": manifest["source"]["sha256"],
        "rows": projected,
    }
    text = canonical_json(value).lower()
    _require(not any(f'"{field}"' in text for field in FORBIDDEN_GPU_INPUT_FIELDS),
             "forbidden outcome field entered GPU input projection")
    value["inputs_sha256"] = sha256_json(value)
    return value


def _verify_model(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(manifest["model"]["path"])
    entries = list(root.rglob("*"))
    _require(not any(path.is_symlink() for path in entries),
             "fresh-base recursive inventory contains a symlink")
    expected_paths = {item["path"] for item in manifest["model"]["files"]}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in entries
        if path.is_file()
    }
    _require(actual_paths == expected_paths,
             "fresh-base recursive file inventory differs")
    receipts = []
    for expected in manifest["model"]["files"]:
        path = root / expected["path"]
        _require(not path.is_symlink(), f"model file is a symlink: {path}")
        _require(path.is_file() and path.stat().st_size == expected["size"],
                 f"model file size differs: {path}")
        digest = sha256_file(path)
        _require(digest == expected["sha256"], f"model file SHA differs: {path}")
        receipts.append({"path": str(path), "size": expected["size"], "sha256": digest})
    return receipts


def _source_artifact(
    manifest: Mapping[str, Any], inputs: Mapping[str, Any], path: Path,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    columns = manifest["source"]["cpu_visible_columns"]
    _require(columns == ["prompt", "context", "extra_info"],
             "CPU label-blind source columns differ")
    table = pq.read_table(manifest["source"]["path"], columns=columns).to_pylist()
    records = []
    allowed_extra = {"index", "num_docs", "question"}
    for frozen in inputs["rows"]:
        source = table[frozen["source_position"]]
        _require(set(source) == set(columns), "CPU source projection columns differ")
        prompt = source.get("prompt")
        extra = source.get("extra_info")
        _require(isinstance(prompt, list) and len(prompt) == 1
                 and isinstance(prompt[0], Mapping)
                 and set(prompt[0]) == {"role", "content"}
                 and prompt[0].get("role") == "user"
                 and isinstance(prompt[0].get("content"), str)
                 and isinstance(source.get("context"), str)
                 and isinstance(extra, Mapping)
                 and set(extra).issubset(allowed_extra),
                 "CPU label-blind source schema differs")
        question = prompt[0]["content"]
        context = source["context"]
        _require(extra.get("index") == frozen["semantic_dataset_index"],
                 "CPU source semantic index differs")
        auxiliary = extra.get("question")
        _require(auxiliary is None or auxiliary == question,
                 "CPU auxiliary question differs from policy prompt")
        _require(hashlib.sha256(question.encode("utf-8")).hexdigest()
                 == frozen["question_sha256"], "CPU question authority differs")
        _require(hashlib.sha256(context.encode("utf-8")).hexdigest()
                 == frozen["context_sha256"], "CPU context authority differs")
        records.append({**frozen, "question": question, "context": context})
    _require(all(set(record) == set(manifest["source"]["gpu_input_fields"])
                 for record in records), "GPU source artifact schema differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        _require(existing == records, "existing GPU source artifact differs")
    else:
        with path.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(canonical_json(record) + "\n")
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "rows": len(records),
        "schema_sha256": sha256_json(manifest["source"]["gpu_input_fields"]),
    }


def _tokenization_authority(
    manifest: Mapping[str, Any], source_artifact_path: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from recurrent.impls.memory import TEMPLATE, TEMPLATE_FINAL_BOXED
    from recurrent.utils import chat_template

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"]["path"], trust_remote_code=True, local_files_only=True,
    )
    generation = _load(Path(manifest["model"]["path"]) / "generation_config.json")
    terminators = generation.get("eos_token_id")
    _require(terminators == manifest["backend"]["termination_token_ids"]
             and set(terminators)
                 == {int(tokenizer.eos_token_id), int(tokenizer.pad_token_id)},
             "fresh-base termination token authority differs")
    records = [
        json.loads(line)
        for line in source_artifact_path.read_text(encoding="utf-8").splitlines()
    ]
    receipts = []
    for row in records:
        question_ids = list(tokenizer.encode(row["question"], add_special_tokens=False))
        _require(len(question_ids) <= manifest["recurrent"]["max_question_tokens"],
                 "question exceeds frozen policy prompt budget")
        context_ids = center_truncate_token_ids(
            list(tokenizer.encode(row["context"], add_special_tokens=False)),
            manifest["recurrent"]["max_context_tokens"],
        )
        chunk_size = manifest["recurrent"]["chunk_size"]
        chunks = [context_ids[offset:offset + chunk_size]
                  for offset in range(0, len(context_ids), chunk_size)]
        _require(1 <= len(chunks) <= manifest["recurrent"]["max_writer_slots"],
                 "writer horizon differs from frozen slot program")
        receipts.append({
            "content_root_id": row["content_root_id"],
            "question_token_count": len(question_ids),
            "question_token_ids_sha256": sha256_json(question_ids),
            "context_token_count": len(context_ids),
            "context_token_ids_sha256": sha256_json(context_ids),
            "active_writer_slots": len(chunks),
            "chunk_token_ids_sha256": [sha256_json(chunk) for chunk in chunks],
        })
    writer_template = chat_template(tokenizer).format(message=TEMPLATE)
    answer_template = chat_template(tokenizer).format(message=TEMPLATE_FINAL_BOXED)
    return {
        "tokenizer_vocab_size": int(len(tokenizer)),
        "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
        "tokenizer_pad_token_id": int(tokenizer.pad_token_id),
        "termination_token_ids": terminators,
        "writer_template_sha256": hashlib.sha256(writer_template.encode("utf-8")).hexdigest(),
        "answer_template_sha256": hashlib.sha256(answer_template.encode("utf-8")).hexdigest(),
        "receipts": receipts,
        "receipts_sha256": sha256_json(receipts),
    }


def _seed_authority(
    manifest: Mapping[str, Any], inputs: Mapping[str, Any],
    tokenization: Mapping[str, Any],
) -> dict[str, Any]:
    horizons = {
        row["content_root_id"]: row["active_writer_slots"]
        for row in tokenization["receipts"]
    }
    trajectories = []
    requests = []
    for frozen in inputs["rows"]:
        root = frozen["content_root_id"]
        for replica in range(manifest["sampling"]["replicas"]):
            trajectory_seed = stable_trajectory_seed(
                manifest["sampling"]["base_seed"], root, replica,
            )
            trajectories.append([root, replica, trajectory_seed])
            for slot in range(horizons[root]):
                requests.append([
                    root, replica, slot,
                    derive_turn_request_seeds([trajectory_seed], [0], slot)[0],
                ])
            answer_slot = manifest["recurrent"]["max_writer_slots"]
            requests.append([
                root, replica, answer_slot,
                derive_turn_request_seeds([trajectory_seed], [0], answer_slot)[0],
            ])
    trajectory_values = [row[-1] for row in trajectories]
    request_values = [row[-1] for row in requests]
    _require(len(set(trajectory_values)) == len(trajectory_values),
             "trajectory seed schedule contains a collision")
    _require(len(set(request_values)) == len(request_values),
             "active request seed schedule contains a collision")
    _require(not set(trajectory_values).intersection(request_values),
             "trajectory and request seed namespaces collide")
    return {
        "trajectory_count": len(trajectories),
        "trajectory_schedule_sha256": sha256_json(trajectories),
        "active_request_count": len(requests),
        "active_request_schedule_sha256": sha256_json(requests),
        "all_trajectory_seeds_unique": True,
        "all_active_request_seeds_unique": True,
        "trajectory_request_namespaces_disjoint": True,
    }


def _source_firewall(repo: Path) -> dict[str, str]:
    runner = repo / RUNNER_REL
    source = runner.read_text(encoding="utf-8").lower()
    forbidden = ("reward_model", "ground_truth", "token_f1", "exact_match", "hotpotqa_dev", "s128")
    _require(not any(term in source for term in forbidden),
             "GPU calibration runner source firewall failed")
    return {
        str(path): sha256_file(repo / path)
        for path in (
            RUNNER_REL, CORE_REL, MIC_V2_CORE_REL, MEMORY_AGENT_REL,
            RECURRENT_UTILS_REL,
        )
    }


def preflight(repo: Path, expected_commit: str, output_root: Path, run_id: str,
              gpu_pair: str) -> dict[str, Any]:
    _runtime(repo, expected_commit, output_root, run_id)
    manifest, manifest_path, manifest_sha = _manifest(repo)
    _certificate, resolved = _verify_data_freeze(manifest)
    source = Path(manifest["source"]["path"])
    _require(source.is_file() and sha256_file(source) == manifest["source"]["sha256"],
             "training parquet SHA differs")
    pair = [int(item) for item in gpu_pair.split(",")]
    _require(len(pair) == 2 and pair == sorted(set(pair)) and all(item >= 0 for item in pair),
             "GPU pair must be two unique canonical ascending indices")
    inputs = _label_blind_inputs(manifest, resolved)
    inputs_path = output_root / "authorities/label_blind_inputs.json"
    source_artifact_path = output_root / "authorities/label_blind_source.jsonl"
    p0_path = output_root / "certificates/p0.json"
    code = _source_firewall(repo)
    source_artifact = _source_artifact(manifest, inputs, source_artifact_path)
    model_files = _verify_model(manifest)
    tokenization = _tokenization_authority(manifest, source_artifact_path)
    seed_authority = _seed_authority(manifest, inputs, tokenization)
    p0 = {
        "schema": "memagent.mic.v2.reference-length-p0",
        "status": "PASS",
        "decision": "MIC_V2_REFERENCE_LENGTH_P0_PASS",
        "git_commit": expected_commit,
        "run_id": run_id,
        "output_root": str(output_root),
        "gpu_pair": pair,
        "scientific_contract_sha256": CONTRACT_SHA256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "data_freeze_certificate_file_sha256": manifest["data_freeze_authority"]["certificate_file_sha256"],
        "data_freeze_resolved_file_sha256": manifest["data_freeze_authority"]["resolved_file_sha256"],
        "source_sha256": manifest["source"]["sha256"],
        "model_files": model_files,
        "tokenization_authority": tokenization,
        "seed_authority": seed_authority,
        "materialization_authority": {
            "parser_version": MATERIALIZATION_PARSER_VERSION,
            "sampled_mask_rule": manifest["materialization"]["sampled_mask_rule"],
            "mic_v2_core_sha256": code[str(MIC_V2_CORE_REL)],
            "memory_agent_sha256": code[str(MEMORY_AGENT_REL)],
            "recurrent_utils_sha256": code[str(RECURRENT_UTILS_REL)],
        },
        "code_sha256": code,
        "label_blind_inputs": str(inputs_path),
        "label_blind_inputs_sha256": inputs["inputs_sha256"],
        "label_blind_source": source_artifact,
        "expected_trajectories": 64 * manifest["sampling"]["replicas"],
        "expected_scheduled_slots": 64 * manifest["sampling"]["replicas"] * 9,
    }
    p0["p0_sha256"] = sha256_json(p0)
    if inputs_path.exists() or p0_path.exists():
        _require(inputs_path.is_file() and p0_path.is_file(), "partial P0 artifacts exist")
        _require(_load(inputs_path) == inputs and _load(p0_path) == p0,
                 "existing P0 differs; recovery refused")
    else:
        write_json_new(inputs_path, inputs)
        write_json_new(p0_path, p0)
    return p0


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    previous = "0" * 64
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        digest = row.pop("record_sha256", None)
        _require(row.get("record_index") == index
                 and row.get("previous_record_sha256") == previous,
                 "trajectory ledger chain differs")
        _require(digest == sha256_json(row), "trajectory ledger record SHA differs")
        row["record_sha256"] = digest
        rows.append(row)
        previous = digest
    return rows


def _validate_ledger(
    manifest: Mapping[str, Any], inputs: Mapping[str, Any], ledger: Path,
    *, expected_commit: str | None = None, run_id: str | None = None,
    p0_sha256: str | None = None,
    tokenization_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = _read_ledger(ledger)
    replicas = manifest["sampling"]["replicas"]
    expected = [(root, replica)
                for root in inputs["rows"] for replica in range(replicas)]
    observed = [(row.get("content_root_id"), row.get("replica")) for row in rows]
    expected_identity = [(root["content_root_id"], replica) for root, replica in expected]
    _require(observed == expected_identity[:len(rows)],
             "trajectory ledger order/identity differs")
    total_tokens = 0
    active_writer_slots = 0
    writer_tokens = 0
    answer_tokens = 0
    termination_ids = manifest["backend"]["termination_token_ids"]
    writer_maximum = manifest["sampling"]["writer"]["max_tokens"]
    answer_maximum = manifest["sampling"]["answer"]["max_tokens"]
    receipt_by_root = {
        receipt["content_root_id"]: receipt
        for receipt in (
            tokenization_authority.get("receipts", [])
            if tokenization_authority is not None else []
        )
    }
    row_fields = {
        "schema", "git_commit", "run_id", "p0_sha256", "content_root_id",
        "source_position", "replica", "trajectory_seed", "active_writer_slots",
        "sampled_policy_tokens", "slots", "record_index",
        "previous_record_sha256", "record_sha256",
    }
    slot_fields = {
        "slot_index", "role", "active", "request_seed", "sampled_token_ids",
        "sampled_policy_tokens", "completion_sha256", "termination",
        "sampled_eos_counted", "sampled_terminal_token_id",
        "sampled_mask_width", "sampled_mask_true_count", "sampled_mask_sha256",
        "parsed_memory_token_ids", "parsed_memory_sha256", "parser_version",
        "afterstate_sha256",
    }
    for row_index, row in enumerate(rows):
        frozen, replica = expected[row_index]
        trajectory_seed = stable_trajectory_seed(
            manifest["sampling"]["base_seed"], frozen["content_root_id"], replica,
        )
        _require(set(row) == row_fields
                 and row.get("schema") == "memagent.mic.v2.reference-length-trajectory"
                 and row.get("source_position") == frozen["source_position"]
                 and row.get("trajectory_seed") == trajectory_seed,
                 "trajectory authority differs")
        if expected_commit is not None:
            _require(row.get("git_commit") == expected_commit,
                     "trajectory Git authority differs")
        if run_id is not None:
            _require(row.get("run_id") == run_id, "trajectory run authority differs")
        if p0_sha256 is not None:
            _require(row.get("p0_sha256") == p0_sha256, "trajectory P0 authority differs")
        slots = row.get("slots", [])
        _require(len(slots) == 9 and [slot.get("slot_index") for slot in slots] == list(range(9)),
                 "scheduled slot receipt differs")
        _require(all(set(slot) == slot_fields for slot in slots),
                 "slot receipt schema differs")
        _require([slot.get("role") for slot in slots]
                 == ["writer"] * 8 + ["answer"],
                 "scheduled slot roles differ")
        _require(all(type(slot.get("active")) is bool for slot in slots),
                 "slot activity type differs")
        active_writer_count = row.get("active_writer_slots")
        _require(type(active_writer_count) is int and 1 <= active_writer_count <= 8,
                 "active writer slot count differs")
        if tokenization_authority is not None:
            _require(frozen["content_root_id"] in receipt_by_root
                     and active_writer_count
                         == receipt_by_root[frozen["content_root_id"]]["active_writer_slots"],
                     "writer horizon differs from tokenized source authority")
        _require([slot.get("active") for slot in slots[:8]]
                 == [turn < active_writer_count for turn in range(8)],
                 "writer activity is not a contiguous exogenous horizon")
        memory_history_sha256: list[str] = []
        chunk_history_sha256 = (
            receipt_by_root[frozen["content_root_id"]]["chunk_token_ids_sha256"]
            if tokenization_authority is not None else None
        )
        for slot in slots:
            count = slot.get("sampled_policy_tokens")
            token_ids = slot.get("sampled_token_ids")
            _require(type(count) is int and count >= 0
                     and isinstance(token_ids, list)
                     and all(type(token) is int and token >= 0 for token in token_ids)
                     and count == len(token_ids), "slot token count differs")
            if tokenization_authority is not None:
                _require(all(token < tokenization_authority["tokenizer_vocab_size"]
                             for token in token_ids),
                         "sampled token ID exceeds frozen tokenizer vocabulary")
            if slot["role"] == "writer":
                if slot["active"]:
                    active_writer_slots += 1
                    _require(0 < count <= writer_maximum,
                             "active writer slot token budget differs")
                    _require(slot.get("termination") in ("sampled_eos", "forced_truncation")
                             and slot.get("sampled_eos_counted")
                                 == (slot.get("termination") == "sampled_eos")
                             and isinstance(slot.get("completion_sha256"), str)
                             and HEX64_RE.fullmatch(slot["completion_sha256"]) is not None
                             and slot["completion_sha256"] == sha256_json(token_ids),
                             "active writer termination receipt differs")
                    expected_request_seed = derive_turn_request_seeds(
                        [trajectory_seed], [0], slot["slot_index"],
                    )[0]
                    _require(slot.get("request_seed") == expected_request_seed,
                             "writer request seed differs")
                    mask_receipt = sampled_policy_mask_receipt(
                        token_ids=token_ids, termination=slot["termination"],
                        termination_token_ids=termination_ids,
                        token_width=writer_maximum,
                    )
                    _require(all(slot.get(key) == value
                                 for key, value in mask_receipt.items()),
                             "writer sampled mask receipt differs")
                    _require(chunk_history_sha256 is not None,
                             "writer afterstate lacks tokenization authority")
                    parsed, afterstate = materialized_memory_receipt(
                        token_ids=token_ids,
                        termination_token_ids=termination_ids,
                        content_root_id=frozen["content_root_id"],
                        trajectory_seed=trajectory_seed,
                        turn_index=slot["slot_index"],
                        arrived_chunk_token_sha256=chunk_history_sha256[
                            :slot["slot_index"] + 1
                        ],
                        prior_memory_token_sha256=memory_history_sha256,
                    )
                    _require(slot.get("parsed_memory_token_ids") == parsed
                             and all(slot.get(key) == value
                                     for key, value in afterstate.items()),
                             "writer materialization/afterstate receipt differs")
                    memory_history_sha256.append(afterstate["parsed_memory_sha256"])
                else:
                    _require(count == 0 and token_ids == []
                             and slot.get("completion_sha256") is None
                             and slot.get("termination") == "exogenous_termination"
                             and slot.get("sampled_eos_counted") is False,
                             "inactive writer slot is not exactly zero")
                    _require(slot.get("request_seed") is None,
                             "inactive writer slot has a request seed")
                    zero_mask = {
                        "sampled_mask_width": writer_maximum,
                        "sampled_mask_true_count": 0,
                        "sampled_mask_sha256": sha256_json([False] * writer_maximum),
                    }
                    _require(all(slot.get(key) == value
                                 for key, value in zero_mask.items())
                             and slot.get("parsed_memory_token_ids") is None
                             and slot.get("parsed_memory_sha256") is None
                             and slot.get("parser_version") is None
                             and slot.get("afterstate_sha256") is None,
                             "inactive writer evidence is not exactly zero")
                writer_tokens += count
            else:
                _require(slot["active"] and 0 < count <= answer_maximum,
                         "answer slot receipt differs")
                _require(slot.get("termination") in ("sampled_eos", "forced_truncation")
                         and slot.get("sampled_eos_counted")
                             == (slot.get("termination") == "sampled_eos")
                         and isinstance(slot.get("completion_sha256"), str)
                         and HEX64_RE.fullmatch(slot["completion_sha256"]) is not None
                         and slot["completion_sha256"] == sha256_json(token_ids),
                         "answer termination receipt differs")
                expected_answer_seed = derive_turn_request_seeds(
                    [trajectory_seed], [0], 8,
                )[0]
                _require(slot.get("request_seed") == expected_answer_seed,
                         "answer request seed differs")
                answer_mask = sampled_policy_mask_receipt(
                    token_ids=token_ids, termination=slot["termination"],
                    termination_token_ids=termination_ids,
                    token_width=answer_maximum,
                )
                _require(all(slot.get(key) == value
                             for key, value in answer_mask.items())
                         and slot.get("parsed_memory_token_ids") is None
                         and slot.get("parsed_memory_sha256") is None
                         and slot.get("parser_version") is None
                         and slot.get("afterstate_sha256") is None,
                         "answer sampled mask/materialization receipt differs")
                answer_tokens += count
            if slot["active"] and slot["termination"] == "sampled_eos":
                _require(token_ids[-1] in termination_ids
                         and not any(token in termination_ids for token in token_ids[:-1])
                         and slot.get("sampled_terminal_token_id") == token_ids[-1],
                         "sampled terminal token receipt differs")
            elif slot["active"]:
                maximum = writer_maximum if slot["role"] == "writer" else answer_maximum
                _require(slot["termination"] == "forced_truncation"
                         and count == maximum
                         and not any(token in termination_ids for token in token_ids)
                         and slot.get("sampled_terminal_token_id") is None,
                         "forced truncation receipt differs")
            else:
                _require(slot.get("sampled_terminal_token_id") is None,
                         "inactive slot has a terminal token")
            total_tokens += count
        _require(row.get("sampled_policy_tokens") == sum(
            slot["sampled_policy_tokens"] for slot in slots),
            "trajectory token total differs")
    return {
        "trajectory_count": len(rows),
        "scheduled_slot_count": len(rows) * 9,
        "active_writer_slot_count": active_writer_slots,
        "writer_policy_tokens": writer_tokens,
        "answer_policy_tokens": answer_tokens,
        "total_policy_tokens": total_tokens,
        "ledger_tail_sha256": rows[-1]["record_sha256"] if rows else "0" * 64,
        "ledger_file_sha256": sha256_file(ledger) if ledger.exists() else None,
    }


def finalize(repo: Path, expected_commit: str, output_root: Path, run_id: str) -> dict[str, Any]:
    _runtime(repo, expected_commit, output_root, run_id)
    manifest, _path, manifest_sha = _manifest(repo)
    required_inputs = (
        output_root / "certificates/p0.json",
        output_root / "certificates/execution.json",
        output_root / "certificates/gpu_replay.json",
        output_root / "authorities/label_blind_inputs.json",
        output_root / "authorities/label_blind_source.jsonl",
        output_root / "trajectories/length_receipts.jsonl",
    )
    _require(all(path.is_file() for path in required_inputs),
             "finalize requires complete producer evidence")
    existing_p0 = _load(output_root / "certificates/p0.json")
    pair = existing_p0.get("gpu_pair")
    _require(isinstance(pair, list) and len(pair) == 2
             and all(type(item) is int for item in pair), "P0 GPU pair differs")
    p0 = preflight(
        repo, expected_commit, output_root, run_id,
        ",".join(str(item) for item in pair),
    )
    inputs = _load(output_root / "authorities/label_blind_inputs.json")
    _self_digest(p0, "p0_sha256")
    _self_digest(inputs, "inputs_sha256")
    _require(p0.get("git_commit") == expected_commit and p0.get("run_id") == run_id
             and p0.get("manifest_sha256") == manifest_sha,
             "P0 identity differs")
    ledger = output_root / "trajectories/length_receipts.jsonl"
    summary = _validate_ledger(
        manifest, inputs, ledger,
        expected_commit=expected_commit, run_id=run_id, p0_sha256=p0["p0_sha256"],
        tokenization_authority=p0["tokenization_authority"],
    )
    _require(summary["trajectory_count"] == p0["expected_trajectories"]
             and summary["scheduled_slot_count"] == p0["expected_scheduled_slots"],
             "calibration rollout is incomplete")
    value = summary["total_policy_tokens"] / summary["scheduled_slot_count"]
    _require(math.isfinite(value) and value > 0, "Lbar_ref is not positive finite")
    execution_path = output_root / "certificates/execution.json"
    execution = _load(execution_path)
    _self_digest(execution, "execution_sha256")
    execution_fields = {
        "schema", "status", "git_commit", "run_id", "p0_sha256", "gpu_pair",
        "physical_gpu_identity", "vllm_version", "strict_vllm",
        "tensor_parallel_size", "prefix_cache_enabled", "termination_token_ids",
        "trainer_attached", "actor_updates", "new_generate_calls_this_session",
        "represented_generate_calls", "trajectory_count", "ledger_file_sha256",
        "execution_sha256",
    }
    _require(set(execution) == execution_fields
             and execution.get("status") == "PASS"
             and execution.get("schema")
                 == "memagent.mic.v2.reference-length-execution"
             and execution.get("git_commit") == expected_commit
             and execution.get("run_id") == run_id
             and execution.get("p0_sha256") == p0["p0_sha256"]
             and execution.get("gpu_pair") == p0["gpu_pair"]
             and isinstance(execution.get("physical_gpu_identity"), list)
             and len(execution["physical_gpu_identity"]) == 2
             and all(set(item) == {"index", "uuid", "name"}
                     for item in execution["physical_gpu_identity"])
             and [item.get("index") for item in execution["physical_gpu_identity"]]
                 == p0["gpu_pair"]
             and len({item.get("uuid") for item in execution["physical_gpu_identity"]}) == 2
             and all(isinstance(item.get("uuid"), str)
                     and item["uuid"].startswith("GPU-")
                     and item.get("name") == "NVIDIA H20"
                     for item in execution["physical_gpu_identity"])
             and execution.get("vllm_version")
                 == manifest["model"]["required_vllm_version"]
             and execution.get("strict_vllm") is True
             and execution.get("tensor_parallel_size") == 2
             and execution.get("prefix_cache_enabled") is False
             and execution.get("termination_token_ids")
                 == p0["tokenization_authority"]["termination_token_ids"]
             and execution.get("trainer_attached") is False
             and execution.get("actor_updates") == 0
             and type(execution.get("new_generate_calls_this_session")) is int
             and 0 <= execution["new_generate_calls_this_session"]
                 <= execution["represented_generate_calls"]
             and execution.get("represented_generate_calls")
                 == summary["active_writer_slot_count"] + summary["trajectory_count"]
             and execution.get("trajectory_count") == summary["trajectory_count"]
             and execution.get("ledger_file_sha256") == summary["ledger_file_sha256"],
             "strict-vLLM execution receipt differs")
    replay = _load(output_root / "certificates/gpu_replay.json")
    _self_digest(replay, "gpu_replay_sha256")
    replay_fields = {
        "schema", "status", "decision", "git_commit", "run_id", "p0_sha256",
        "execution_sha256", "gpu_pair", "physical_gpu_identity", "vllm_version",
        "termination_token_ids", "trajectory_count", "regenerated_generate_calls",
        "exact_token_match_count", "ledger_file_sha256", "gpu_replay_sha256",
    }
    _require(set(replay) == replay_fields
             and replay.get("schema")
                 == "memagent.mic.v2.reference-length-gpu-replay"
             and replay.get("status") == "PASS"
             and replay.get("decision") == "MIC_V2_REFERENCE_LENGTH_GPU_REPLAY_PASS"
             and replay.get("git_commit") == expected_commit
             and replay.get("run_id") == run_id
             and replay.get("p0_sha256") == p0["p0_sha256"]
             and replay.get("execution_sha256") == execution["execution_sha256"]
             and replay.get("gpu_pair") == p0["gpu_pair"]
             and replay.get("physical_gpu_identity")
                 == execution["physical_gpu_identity"]
             and all(set(item) == {"index", "uuid", "name"}
                     for item in replay["physical_gpu_identity"])
             and replay.get("vllm_version")
                 == manifest["model"]["required_vllm_version"]
             and replay.get("termination_token_ids")
                 == p0["tokenization_authority"]["termination_token_ids"]
             and replay.get("trajectory_count") == summary["trajectory_count"]
             and replay.get("regenerated_generate_calls")
                 == execution["represented_generate_calls"]
             and replay.get("exact_token_match_count")
                 == execution["represented_generate_calls"]
             and replay.get("ledger_file_sha256") == summary["ledger_file_sha256"],
             "independent GPU replay receipt differs")
    report = {
        "schema": "memagent.mic.v2.reference-length-certificate",
        "status": "PASS",
        "decision": "MIC_V2_REFERENCE_LENGTH_CALIBRATION_PASS",
        "git_commit": expected_commit,
        "run_id": run_id,
        "output_root": str(output_root),
        "scientific_contract_sha256": CONTRACT_SHA256,
        "manifest_sha256": manifest_sha,
        "p0_sha256": p0["p0_sha256"],
        "label_blind_inputs_sha256": inputs["inputs_sha256"],
        "execution_sha256": execution["execution_sha256"],
        "gpu_replay_sha256": replay["gpu_replay_sha256"],
        "seed_authority": p0["seed_authority"],
        "materialization_authority": p0["materialization_authority"],
        "statistic": "arithmetic_mean_sampled_policy_tokens_over_scheduled_slots",
        "lbar_ref": value,
        **summary,
    }
    report["certificate_sha256"] = sha256_json(report)
    path = output_root / "certificates/reference_length.json"
    if path.exists():
        _require(_load(path) == report, "existing calibration certificate differs")
    else:
        write_json_new(path, report)
    return report


def verify(repo: Path, expected_commit: str, output_root: Path, run_id: str) -> dict[str, Any]:
    required = (
        output_root / "certificates/p0.json",
        output_root / "certificates/execution.json",
        output_root / "certificates/gpu_replay.json",
        output_root / "certificates/reference_length.json",
        output_root / "authorities/label_blind_inputs.json",
        output_root / "authorities/label_blind_source.jsonl",
        output_root / "trajectories/length_receipts.jsonl",
    )
    _require(all(path.is_file() for path in required),
             "verify requires a complete immutable calibration attempt")
    report = finalize(repo, expected_commit, output_root, run_id)
    _finite(report)
    _require(_self_digest(report, "certificate_sha256") == report["certificate_sha256"],
             "calibration certificate digest differs")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "finalize", "verify"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-pair", default="")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.repo.resolve(), args.expected_commit,
                           args.output_root.resolve(), args.run_id, args.gpu_pair)
    elif args.command == "finalize":
        result = finalize(args.repo.resolve(), args.expected_commit,
                          args.output_root.resolve(), args.run_id)
    else:
        result = verify(args.repo.resolve(), args.expected_commit,
                        args.output_root.resolve(), args.run_id)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

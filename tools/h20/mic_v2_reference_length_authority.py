#!/usr/bin/env python3
"""Read-only verifier for the Git-frozen MIC-v2 reference-length authority."""

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

AUTHORITY_REL = Path("manifests/h20/qwen25_7b_mic_v2_reference_length_authority.json")
AUTHORITY_COMMIT = "ecd9c5e7829245da9fc158e1da0cc1953054faf7"
AUTHORITY_FILE_SHA256 = "c8b4f3d3e93099cffc65e2d4ad2465b42598cca381b2e9b70d81a45adf162096"
PRODUCER_COMMIT = "141f6ea90e68de09f3650e8baae2976725869ebf"
GENERATION_MANIFEST_REL = Path(
    "manifests/h20/qwen25_7b_mic_v2_reference_length_calibration.json"
)
EVIDENCE_SCOPE = (
    "transcript-bound raw SHA evidence supplied by the H20 operator and accepted "
    "by independent review; no local re-hash of H20 files"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMIT40 = re.compile(r"^[0-9a-f]{40}$")

CANONICAL_FIELDS = {
    "p0": "p0_sha256", "execution": "execution_sha256",
    "gpu_replay": "gpu_replay_sha256", "reference_length": "certificate_sha256",
    "label_blind_inputs": "inputs_sha256",
}
ARTIFACT_SUFFIXES = {
    "p0": "certificates/p0.json", "execution": "certificates/execution.json",
    "gpu_replay": "certificates/gpu_replay.json",
    "reference_length": "certificates/reference_length.json",
    "label_blind_inputs": "authorities/label_blind_inputs.json",
    "label_blind_source": "authorities/label_blind_source.jsonl",
    "trajectory_ledger": "trajectories/length_receipts.jsonl",
    "calibration_log": "calibration.log",
}
CODE_PATHS = {
    "gpu_producer_and_replay": "tools/h20/run_qwen25_7b_mic_v2_reference_length_calibration.py",
    "memory_agent": "recurrent/impls/memory.py",
    "mic_v2_core": "recurrent/research/mic_v2.py",
    "recurrent_utils": "recurrent/utils.py",
    "scientific_contract": "docs/papers/mic_v2_scientific_contract_20260825.md",
    "statistic_and_finalizer": "tools/h20/mic_v2_reference_length_calibration.py",
}
EXECUTION_FIELDS = {
    "schema", "status", "git_commit", "run_id", "p0_sha256", "gpu_pair",
    "physical_gpu_identity", "vllm_version", "config_loader_environment", "strict_vllm",
    "tensor_parallel_size", "prefix_cache_enabled", "termination_token_ids",
    "trainer_attached", "actor_updates", "new_generate_calls_this_session",
    "represented_generate_calls", "trajectory_count", "ledger_file_sha256",
    "execution_sha256",
}
REPLAY_FIELDS = {
    "schema", "status", "decision", "git_commit", "run_id", "p0_sha256",
    "execution_sha256", "gpu_pair", "physical_gpu_identity", "vllm_version",
    "config_loader_environment", "termination_token_ids", "trajectory_count",
    "regenerated_generate_calls", "exact_token_match_count", "ledger_file_sha256",
    "gpu_replay_sha256",
}
P0_FIELDS = {
    "schema", "status", "decision", "git_commit", "run_id", "output_root", "gpu_pair",
    "scientific_contract_sha256", "manifest_path", "manifest_sha256",
    "data_freeze_certificate_file_sha256", "data_freeze_resolved_file_sha256",
    "source_sha256", "model_files", "tokenization_authority", "seed_authority",
    "materialization_authority", "code_sha256", "label_blind_inputs",
    "label_blind_inputs_sha256", "label_blind_source", "expected_trajectories",
    "expected_scheduled_slots", "p0_sha256",
}
REFERENCE_FIELDS = {
    "schema", "status", "decision", "git_commit", "run_id", "output_root",
    "scientific_contract_sha256", "manifest_sha256", "p0_sha256",
    "label_blind_inputs_sha256", "execution_sha256", "gpu_replay_sha256",
    "seed_authority", "materialization_authority", "statistic", "lbar_ref",
    "trajectory_count", "scheduled_slot_count", "active_writer_slot_count",
    "writer_policy_tokens", "answer_policy_tokens", "total_policy_tokens",
    "ledger_tail_sha256", "ledger_file_sha256", "certificate_sha256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MIC_V2_NO_GO: {message}")


def canonical_json(value: Any) -> str:
    """Frozen authority canonicalization; never imported from mutable E1 code."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    _require(isinstance(value, dict), f"JSON authority is not an object: {label}")
    return value


def _load(path: Path) -> dict[str, Any]:
    return _json_bytes(path.read_bytes(), str(path))


def _self_digest(value: Mapping[str, Any], field: str) -> str:
    digest = value.get(field)
    _require(isinstance(digest, str) and HEX64.fullmatch(digest) is not None,
             f"missing canonical digest {field}")
    unsigned = dict(value)
    unsigned.pop(field, None)
    _require(sha256_json(unsigned) == digest, f"canonical digest differs: {field}")
    return digest


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL,
    ).strip()


def _safe_repo_relative(relative: str) -> Path:
    _require(isinstance(relative, str), "repository path is not a string")
    candidate = Path(relative)
    _require(relative == candidate.as_posix() and not candidate.is_absolute()
             and ".." not in candidate.parts and "." not in candidate.parts,
             "repository path is unsafe")
    return candidate


def _git_blob(repo: Path, commit: str, relative: str) -> bytes:
    _safe_repo_relative(relative)
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:{relative}"],
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"MIC_V2_NO_GO: historical Git blob unavailable: {relative}") from exc


def _ancestor(repo: Path, older: str, newer: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _under(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    _require(isinstance(relative, str) and relative == candidate.as_posix()
             and not candidate.is_absolute() and ".." not in candidate.parts,
             "authority artifact path is unsafe")
    target = (root / candidate).resolve()
    _require(target == root or root in target.parents,
             "authority artifact escapes work root")
    return target


def _verify_data_freeze_frozen(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = manifest.get("data_freeze_authority", {})
    _require(set(authority) == {
        "root", "run_id", "certificate_file_sha256", "certificate_canonical_sha256",
        "resolved_file_sha256", "resolved_canonical_sha256", "split", "content_roots",
    }, "generation manifest data-freeze schema differs")
    root = Path(authority["root"])
    _require(root.is_absolute(), "data-freeze root is not absolute")
    certificate_path = root / "certificates/data_freeze.json"
    resolved_path = root / "resolved_split_manifest.json"
    _require(certificate_path.is_file() and resolved_path.is_file()
             and sha256_file(certificate_path) == authority["certificate_file_sha256"]
             and sha256_file(resolved_path) == authority["resolved_file_sha256"],
             "data-freeze raw authority differs")
    certificate, resolved = _load(certificate_path), _load(resolved_path)
    _require(_self_digest(certificate, "certificate_sha256")
             == authority["certificate_canonical_sha256"]
             and _self_digest(resolved, "resolved_manifest_sha256")
                 == authority["resolved_canonical_sha256"],
             "data-freeze canonical authority differs")
    _require(certificate.get("schema") == "memagent.mic.v2.data-freeze-certificate"
             and certificate.get("status") == "PASS"
             and certificate.get("decision") == "MIC_V2_DATA_FREEZE_PASS"
             and certificate.get("run_id") == authority["run_id"]
             and resolved.get("schema") == "memagent.mic.v2.resolved-data-split"
             and resolved.get("status") == "PASS"
             and resolved.get("decision") == "MIC_V2_DATA_SPLIT_FROZEN"
             and resolved.get("run_id") == authority["run_id"],
             "data-freeze identity/status differs")
    return certificate, resolved


def _label_blind_inputs_frozen(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any],
) -> dict[str, Any]:
    authority = manifest["data_freeze_authority"]
    split = resolved.get("splits", {}).get(authority["split"], {})
    rows = split.get("rows", [])
    _require(isinstance(rows, list) and len(rows) == authority["content_roots"] == 64,
             "reference-length split coverage differs")
    projected = []
    for row in rows:
        _require(isinstance(row, dict), "resolved label-blind row is not an object")
        item = {
            "source_position": row.get("source_position"),
            "semantic_dataset_index": row.get("semantic_dataset_index"),
            "question_sha256": row.get("question_sha256"),
            "context_sha256": row.get("context_sha256"),
            "content_root_id": row.get("content_root_id"),
        }
        _require(type(item["source_position"]) is int
                 and type(item["semantic_dataset_index"]) is int
                 and all(isinstance(item[key], str) and HEX64.fullmatch(item[key])
                         for key in ("question_sha256", "context_sha256", "content_root_id")),
                 "label-blind row identity differs")
        projected.append(item)
    _require(len({row["content_root_id"] for row in projected}) == 64,
             "reference-length content roots are not unique")
    value = {
        "schema": "memagent.mic.v2.reference-length-label-blind-inputs",
        "status": "FROZEN",
        "source_path": manifest["source"]["path"],
        "source_sha256": manifest["source"]["sha256"],
        "rows": projected,
    }
    value["inputs_sha256"] = sha256_json(value)
    return value


def _exact_result(reference: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key, value in expected.items():
        if key == "exact_token_match_count":
            continue
        actual = reference.get(key)
        if isinstance(value, float):
            _require(isinstance(actual, (float, int)) and math.isfinite(actual)
                     and float(actual) == value, f"reference result differs: {key}")
        else:
            _require(actual == value, f"reference result differs: {key}")
    _require(expected["writer_policy_tokens"] + expected["answer_policy_tokens"]
             == expected["total_policy_tokens"], "authority token accounting differs")
    _require(expected["trajectory_count"] * 9 == expected["scheduled_slot_count"],
             "authority slot accounting differs")
    _require(expected["active_writer_slot_count"] + expected["trajectory_count"]
             == expected["exact_token_match_count"],
             "authority active-request accounting differs")
    _require(expected["total_policy_tokens"] / expected["scheduled_slot_count"]
             == expected["lbar_ref"], "authority Lbar_ref arithmetic differs")


def _fixed_authority(repo: Path, head: str) -> dict[str, Any]:
    _require(COMMIT40.fullmatch(AUTHORITY_COMMIT) is not None
             and _ancestor(repo, AUTHORITY_COMMIT, head),
             "frozen authority commit is not an ancestor of current HEAD")
    raw = _git_blob(repo, AUTHORITY_COMMIT, AUTHORITY_REL.as_posix())
    _require(hashlib.sha256(raw).hexdigest() == AUTHORITY_FILE_SHA256,
             "frozen authority Git blob SHA differs")
    current = repo / AUTHORITY_REL
    _require(current.is_file() and sha256_file(current) == AUTHORITY_FILE_SHA256,
             "current authority file differs from frozen Git blob")
    return _json_bytes(raw, f"{AUTHORITY_COMMIT}:{AUTHORITY_REL}")


def _verify_authority(repo: Path, work_root: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema", "status", "decision", "evidence_scope", "producer", "artifacts",
        "data_freeze_authority", "generation_authority", "code_authority",
        "seed_authority", "result", "authority_sha256",
    }
    _require(set(authority) == expected_top
             and authority.get("schema") == "memagent.mic.v2.reference-length-authority"
             and authority.get("status") == "PASS"
             and authority.get("decision") == "MIC_V2_REFERENCE_LENGTH_AUTHORITY_FROZEN"
             and authority.get("evidence_scope") == EVIDENCE_SCOPE,
             "reference-length authority schema/status/scope differs")
    _self_digest(authority, "authority_sha256")

    producer = authority["producer"]
    _require(set(producer) == {"git_commit", "run_id", "output_root_relative_to_work_root"}
             and producer.get("git_commit") == PRODUCER_COMMIT
             and COMMIT40.fullmatch(producer["git_commit"]) is not None
             and _ancestor(repo, PRODUCER_COMMIT, AUTHORITY_COMMIT),
             "producer identity or ancestry differs")
    output_rel = producer["output_root_relative_to_work_root"]
    output_root = _under(work_root, output_rel)

    artifacts = authority["artifacts"]
    _require(set(artifacts) == set(ARTIFACT_SUFFIXES), "authority artifact set differs")
    loaded: dict[str, dict[str, Any]] = {}
    raw_sha: dict[str, str] = {}
    for name, suffix in ARTIFACT_SUFFIXES.items():
        receipt = artifacts[name]
        canonical_field = CANONICAL_FIELDS.get(name)
        expected_fields = {"path_relative_to_work_root", "file_sha256"}
        if canonical_field:
            expected_fields |= {"canonical_field", "canonical_sha256"}
        exact_relative = f"{output_rel}/{suffix}"
        _require(set(receipt) == expected_fields
                 and receipt.get("path_relative_to_work_root") == exact_relative
                 and HEX64.fullmatch(receipt.get("file_sha256", "")) is not None,
                 f"authority artifact receipt/path differs: {name}")
        path = _under(work_root, exact_relative)
        _require(path.is_file() and sha256_file(path) == receipt["file_sha256"],
                 f"authority artifact file SHA differs: {name}")
        raw_sha[name] = receipt["file_sha256"]
        if canonical_field:
            _require(receipt.get("canonical_field") == canonical_field
                     and HEX64.fullmatch(receipt.get("canonical_sha256", "")) is not None,
                     f"authority canonical mapping differs: {name}")
            value = _load(path)
            _require(_self_digest(value, canonical_field) == receipt["canonical_sha256"],
                     f"authority artifact canonical SHA differs: {name}")
            loaded[name] = value

    p0, execution, replay = loaded["p0"], loaded["execution"], loaded["gpu_replay"]
    reference, inputs = loaded["reference_length"], loaded["label_blind_inputs"]
    _require(set(p0) == P0_FIELDS and p0.get("schema") == "memagent.mic.v2.reference-length-p0"
             and p0.get("status") == "PASS", "P0 schema differs")
    _require(set(execution) == EXECUTION_FIELDS
             and execution.get("schema") == "memagent.mic.v2.reference-length-execution"
             and execution.get("status") == "PASS", "execution schema differs")
    _require(set(replay) == REPLAY_FIELDS
             and replay.get("schema") == "memagent.mic.v2.reference-length-gpu-replay"
             and replay.get("status") == "PASS", "GPU replay schema differs")
    _require(set(reference) == REFERENCE_FIELDS
             and reference.get("schema") == "memagent.mic.v2.reference-length-certificate"
             and reference.get("status") == "PASS", "reference certificate schema differs")
    _require(set(inputs) == {"schema", "status", "source_path", "source_sha256", "rows", "inputs_sha256"}
             and inputs.get("schema") == "memagent.mic.v2.reference-length-label-blind-inputs"
             and inputs.get("status") == "FROZEN", "label-blind input schema differs")
    _require(all(value.get("git_commit") == PRODUCER_COMMIT
                 and value.get("run_id") == producer["run_id"]
                 for value in (p0, execution, replay, reference)),
             "producer commit/run chain differs")
    _require(p0.get("output_root") == str(output_root)
             and reference.get("output_root") == str(output_root)
             and p0.get("label_blind_inputs")
                 == str(output_root / ARTIFACT_SUFFIXES["label_blind_inputs"]),
             "producer output-root chain differs")

    generation = authority["generation_authority"]
    _require(set(generation) == {"config_loader_environment", "gpu_pair", "manifest_path",
                                "manifest_sha256", "physical_gpu_identity", "vllm_version"}
             and generation.get("manifest_path") == GENERATION_MANIFEST_REL.as_posix()
             and HEX64.fullmatch(generation.get("manifest_sha256", "")) is not None,
             "generation authority schema/path differs")
    manifest_raw = _git_blob(repo, PRODUCER_COMMIT, GENERATION_MANIFEST_REL.as_posix())
    _require(hashlib.sha256(manifest_raw).hexdigest() == generation["manifest_sha256"],
             "historical generation manifest differs")
    manifest = _json_bytes(manifest_raw, str(GENERATION_MANIFEST_REL))
    _require(p0.get("manifest_sha256") == generation["manifest_sha256"]
             and reference.get("manifest_sha256") == generation["manifest_sha256"]
             and execution.get("vllm_version") == replay.get("vllm_version")
                 == generation["vllm_version"]
             and execution.get("config_loader_environment")
                 == replay.get("config_loader_environment")
                 == generation["config_loader_environment"]
             and execution.get("gpu_pair") == replay.get("gpu_pair") == generation["gpu_pair"]
             and execution.get("physical_gpu_identity")
                 == replay.get("physical_gpu_identity") == generation["physical_gpu_identity"],
             "generation execution/replay chain differs")

    code = authority["code_authority"]
    _require(set(code) == set(CODE_PATHS), "code authority set differs")
    for name, exact_path in CODE_PATHS.items():
        receipt = code[name]
        _require(set(receipt) == {"path", "sha256"} and receipt.get("path") == exact_path
                 and HEX64.fullmatch(receipt.get("sha256", "")) is not None,
                 f"code authority receipt differs: {name}")
        _require(hashlib.sha256(_git_blob(repo, PRODUCER_COMMIT, exact_path)).hexdigest()
                 == receipt["sha256"], f"historical producer code differs: {name}")
    contract_sha = code["scientific_contract"]["sha256"]
    _require(p0.get("scientific_contract_sha256") == contract_sha
             and reference.get("scientific_contract_sha256") == contract_sha,
             "scientific contract chain differs")
    p0_code = p0.get("code_sha256", {})
    _require(all(p0_code.get(receipt["path"]) == receipt["sha256"]
                 for name, receipt in code.items() if name != "scientific_contract"),
             "P0 code authority differs")

    freeze = authority["data_freeze_authority"]
    _require(set(freeze) == {"certificate_canonical_sha256", "certificate_file_sha256",
                             "content_roots", "resolved_canonical_sha256",
                             "resolved_file_sha256", "run_id", "split"}
             and freeze.get("split") == "reference_length_calibration"
             and freeze.get("content_roots") == 64,
             "data-freeze authority schema differs")
    manifest_freeze = manifest.get("data_freeze_authority", {})
    _require({key: manifest_freeze.get(key) for key in freeze} == freeze
             and set(manifest_freeze) == set(freeze) | {"root"}
             and p0.get("data_freeze_certificate_file_sha256")
                 == freeze["certificate_file_sha256"]
             and p0.get("data_freeze_resolved_file_sha256") == freeze["resolved_file_sha256"],
             "data-freeze authority chain differs")
    _certificate, resolved = _verify_data_freeze_frozen(manifest)
    _require(_label_blind_inputs_frozen(manifest, resolved) == inputs,
             "label-blind inputs do not reconstruct from frozen data split")

    seed = authority["seed_authority"]
    _require(set(seed) == {"active_request_count", "active_request_schedule_sha256",
                           "all_active_request_seeds_unique", "all_trajectory_seeds_unique",
                           "trajectory_count", "trajectory_request_namespaces_disjoint",
                           "trajectory_schedule_sha256"}
             and p0.get("seed_authority") == seed and reference.get("seed_authority") == seed,
             "seed authority chain differs")
    result = authority["result"]
    _require(set(result) == {"active_writer_slot_count", "answer_policy_tokens",
                             "exact_token_match_count", "lbar_ref", "scheduled_slot_count",
                             "statistic", "total_policy_tokens", "trajectory_count",
                             "writer_policy_tokens"}, "result schema differs")
    _exact_result(reference, result)
    active_count = seed["active_request_count"]
    _require(active_count == result["exact_token_match_count"]
             == execution.get("represented_generate_calls")
             == replay.get("regenerated_generate_calls")
             == replay.get("exact_token_match_count")
             and seed["trajectory_count"] == result["trajectory_count"]
                 == execution.get("trajectory_count") == replay.get("trajectory_count")
             and execution.get("p0_sha256") == replay.get("p0_sha256")
                 == reference.get("p0_sha256") == p0.get("p0_sha256")
             and replay.get("execution_sha256") == execution.get("execution_sha256")
                 == reference.get("execution_sha256")
             and reference.get("gpu_replay_sha256") == replay.get("gpu_replay_sha256")
             and execution.get("ledger_file_sha256") == replay.get("ledger_file_sha256")
                 == reference.get("ledger_file_sha256") == raw_sha["trajectory_ledger"]
             and reference.get("label_blind_inputs_sha256") == inputs.get("inputs_sha256")
                 == p0.get("label_blind_inputs_sha256")
             and p0.get("label_blind_source", {}).get("file_sha256")
                 == raw_sha["label_blind_source"],
             "result/artifact cross-chain differs")

    return {
        "schema": "memagent.mic.v2.reference-length-authority-verification",
        "status": "PASS", "decision": "MIC_V2_REFERENCE_LENGTH_AUTHORITY_PASS",
        "authority_commit": AUTHORITY_COMMIT,
        "authority_file_sha256": AUTHORITY_FILE_SHA256,
        "authority_sha256": authority["authority_sha256"],
        "producer_git_commit": producer["git_commit"], "producer_run_id": producer["run_id"],
        "reference_length_file_sha256": raw_sha["reference_length"],
        "reference_length_canonical_sha256": artifacts["reference_length"]["canonical_sha256"],
        "lbar_ref": result["lbar_ref"],
    }


def verify_reference_length_authority(repo: Path, work_root: Path) -> dict[str, Any]:
    repo, work_root = repo.resolve(), work_root.resolve()
    _require(sys.flags.optimize == 0, "optimized Python is forbidden")
    _require(repo.is_dir() and work_root.is_dir(), "authority roots are unavailable")
    authority = _fixed_authority(repo, _git(repo, "rev-parse", "HEAD"))
    return _verify_authority(repo, work_root, authority)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_reference_length_authority(args.repo, args.work_root),
                     sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

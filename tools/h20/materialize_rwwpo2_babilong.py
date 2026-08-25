#!/usr/bin/env python3
"""Materialize frozen BABILong MemAgent parquets and stable identity manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.rwwpo2_babilong import (
    CHUNK_SIZE, GENERATION_SEED, LENGTHS, MAX_CHUNKS, SOURCE_DATASET_ID,
    SOURCE_REVISION, SOURCE_ROWS_PER_CELL, TASK_DEPTH, adapt_partition,
    validate_frozen_contract,
)
from recurrent.research.stable_eval_identity import canonical_sha256, validate_resolved_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_bundle(
    root: Path, *, expected_sha256: str, expected_commit: str,
    adapter_manifest_sha256: str,
) -> tuple[dict, dict]:
    bundle_path = root / "bundle_manifest.json"
    if root.is_symlink() or bundle_path.is_symlink() or not bundle_path.is_file() \
            or sha256_file(bundle_path) != expected_sha256:
        raise ValueError("source bundle manifest SHA")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    declared = bundle.pop("report_sha256", None)
    actual = hashlib.sha256(json.dumps(
        bundle, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if declared != actual or bundle.get("status") != "PASS" \
            or bundle.get("decision") != "RWWPO2_BABILONG_SOURCE_BUNDLE_PASS" \
            or bundle.get("git_commit") != expected_commit \
            or bundle.get("adapter_manifest_sha256") != adapter_manifest_sha256 \
            or bundle.get("dataset_id") != SOURCE_DATASET_ID \
            or bundle.get("dataset_revision") != SOURCE_REVISION:
        raise ValueError("source bundle authentication")
    sources = {}
    cells = bundle.get("cells")
    if not isinstance(cells, list) or len(cells) != len(LENGTHS) * len(TASK_DEPTH) \
            or canonical_sha256(cells) != bundle.get("cell_inventory_sha256"):
        raise ValueError("source cell inventory")
    inventory = {}
    for item in cells:
        if not isinstance(item, dict):
            raise ValueError("source cell inventory row")
        key = (item.get("length"), item.get("task"))
        if key in inventory:
            raise ValueError("duplicate source cell")
        inventory[key] = item
    for length in LENGTHS:
        for task in TASK_DEPTH:
            item = inventory.get((length, task))
            if not isinstance(item, dict):
                raise ValueError(f"source cell missing: {length}/{task}")
            expected_relative = f"source/{length}/{task}.jsonl"
            if item.get("relative_path") != expected_relative \
                    or int(item.get("rows", -1)) != SOURCE_ROWS_PER_CELL:
                raise ValueError(f"source cell path/count: {length}/{task}")
            path = (root / expected_relative)
            if path.resolve().parent != (root / "source" / length).resolve() \
                    or path.is_symlink() or not path.is_file() \
                    or sha256_file(path) != item.get("file_sha256"):
                raise ValueError(f"source cell SHA: {length}/{task}")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != SOURCE_ROWS_PER_CELL \
                    or any(set(row) != {"input", "question", "target"}
                           or any(not isinstance(row[field], str) or not row[field].strip()
                                  for field in ("input", "question", "target"))
                           for row in rows) \
                    or canonical_sha256(rows) != item.get("canonical_rows_sha256"):
                raise ValueError(f"source cell canonical SHA: {length}/{task}")
            sources[(length, task)] = rows
    return {**bundle, "report_sha256": declared}, sources


def tokenizer_inventory(root: Path) -> list[dict]:
    rows = []
    for name in (
        "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "tokenizer.model", "sentencepiece.bpe.model",
        "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    ):
        path = root / name
        if path.is_file() and not path.is_symlink():
            rows.append({"path": name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    if not rows:
        raise ValueError("tokenizer inventory is empty")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--tokenizer-root", required=True)
    parser.add_argument("--partition", choices=("development", "confirmation"), required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_BABILONG_MATERIALIZE_NO_GO:checkout")
    manifest_path = Path(args.manifest).resolve()
    source_root = Path(args.source_root).resolve()
    tokenizer_root = Path(args.tokenizer_root).resolve()
    output_root = Path(args.output_root)
    if any(path.is_symlink() for path in (manifest_path, source_root, tokenizer_root)) \
            or output_root.exists() or output_root.is_symlink():
        raise SystemExit("RWWPO2_BABILONG_MATERIALIZE_NO_GO:symlink/one-use root")
    if sha256_file(manifest_path) != args.manifest_sha256:
        raise SystemExit("RWWPO2_BABILONG_MATERIALIZE_NO_GO:manifest SHA")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_frozen_contract(manifest)
    try:
        bundle, sources = read_source_bundle(
            source_root, expected_sha256=args.source_manifest_sha256,
            expected_commit=head, adapter_manifest_sha256=args.manifest_sha256,
        )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_root, local_files_only=True, trust_remote_code=False
        )
        adapted = adapt_partition(
            sources, partition=args.partition,
            context_token_length=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        )
        import pyarrow as pa
        import pyarrow.parquet as pq
    except (ImportError, OSError, ValueError) as error:
        raise SystemExit("RWWPO2_BABILONG_MATERIALIZE_NO_GO:" + str(error)) from error
    inventory = tokenizer_inventory(tokenizer_root)
    output_root.mkdir(parents=True)
    outputs = []
    for length in LENGTHS:
        rows, identities = adapted[length]
        validation = output_root / f"babilong_{length}_{args.partition}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), validation, compression="zstd")
        data_sha = sha256_file(validation)
        identity_payload = {
            "schema_version": 1,
            "namespace": "rwwpo2-babilong-stable-identity-v1",
            "source_dataset": {
                "role": f"babilong_{args.partition}",
                "dataset_id": SOURCE_DATASET_ID,
                "dataset_revision": SOURCE_REVISION,
                "source_bundle_manifest_sha256": args.source_manifest_sha256,
                "parquet_sha256": data_sha,
                "length": length, "tasks": list(TASK_DEPTH),
                "raw_rows": len(rows), "production_effective_rows": len(rows),
                "shuffle": False, "filter_overlong_prompts": True,
                "production_effective_prompt_limit": CHUNK_SIZE * MAX_CHUNKS[length],
            },
            "base_model_protocol": {
                "id": "Qwen/Qwen2.5-7B-Instruct",
                "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
            },
            "tokenizer": {
                "files": inventory, "manifest_sha256": canonical_sha256(inventory),
            },
            "identity_construction": {
                "version": 1,
                "example_id": "length/depth/source-index integer namespace",
                "source_order_index": "depth-major order over frozen source-index membership",
                "row_hashes": "UTF-8 prompt/context and canonical-JSON target hashes",
            },
            "decode": dict(manifest["decode"]),
            "backend": {"rollout": "vllm", "strict": True, "hf_fallback": False},
            "rows": identities,
        }
        resolved = {
            "schema_version": 1,
            "identity_payload": identity_payload,
            "eval_manifest_hash": canonical_sha256(identity_payload),
            "babilong_binding": {
                "adapter_manifest_path": str(manifest_path),
                "adapter_manifest_sha256": args.manifest_sha256,
                "source_bundle_path": str(source_root),
                "source_bundle_manifest_sha256": args.source_manifest_sha256,
                "source_bundle_report_sha256": bundle["report_sha256"],
                "validation_path": str(validation.resolve()),
                "validation_sha256": data_sha,
                "partition": args.partition, "length": length,
                "tasks": list(TASK_DEPTH), "task_depth": TASK_DEPTH,
                "generation_seed": GENERATION_SEED,
                "chunk_size": CHUNK_SIZE, "max_chunks": MAX_CHUNKS[length],
                "examples": len(rows),
                "outcomes_consumed_for_membership": False,
            },
        }
        validate_resolved_manifest(resolved)
        resolved_path = output_root / f"babilong_{length}_{args.partition}_resolved.json"
        with resolved_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(resolved, sort_keys=True, indent=2) + "\n")
        outputs.append({
            "length": length, "rows": len(rows),
            "validation_path": str(validation.resolve()),
            "validation_sha256": data_sha,
            "resolved_path": str(resolved_path.resolve()),
            "resolved_sha256": sha256_file(resolved_path),
            "eval_manifest_hash": resolved["eval_manifest_hash"],
        })
    report = {
        "schema_version": "rwwpo2-babilong-materialization-v1",
        "status": "PASS", "decision": "RWWPO2_BABILONG_MATERIALIZATION_PASS",
        "git_commit": head, "adapter_manifest_sha256": args.manifest_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_report_sha256": bundle["report_sha256"],
        "partition": args.partition, "outputs": outputs,
        "output_inventory_sha256": canonical_sha256(outputs),
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    report_path = output_root / "materialization_report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": report["decision"],
        "partition": args.partition, "outputs": outputs,
        "report": str(report_path.resolve()), "report_sha256": sha256_file(report_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

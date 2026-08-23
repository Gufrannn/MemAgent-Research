#!/usr/bin/env python3
"""Materialize a content-blind, commit-bound RWWPO-2 confirmation-set seal.

The sealer reads source rows only to compute canonical identities. It emits no
question, context, answer, row-level identity, or model output. The receipt
proves file identity, uniqueness, minimum size, and disjointness from the exact
actor-training prefix and the already adaptive S128 development set.
"""
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

from tools.h20.audit_tf_rwwpo_budget_leakage import (
    EXPECTED_S128_RESOLVED_SHA256,
    EXPECTED_TRAIN_SHA256,
    _actor_consumed_rows,
    _load_s128,
    _read_parquet,
    sha256_file,
    source_identity,
)
from recurrent.research.stable_eval_identity import (
    canonical_sha256, validate_resolved_manifest,
)
from tools.h20.preflight_qwen25_7b_stable_i4x2 import freeze_existing_s128_rows


EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"


def aggregate_digest(values: set[str]) -> str:
    payload = json.dumps(sorted(values), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--tokenizer-root", required=True)
    parser.add_argument("--s128-resolved", required=True)
    parser.add_argument("--s128-resolved-sha256", required=True)
    parser.add_argument("--confirmation-data", required=True)
    parser.add_argument("--confirmation-data-sha256", required=True)
    parser.add_argument("--seal-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resolved-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:checkout")
    if not args.seal_id.startswith("rwwpo2-confirm-") \
            or not all(character.isalnum() or character in "-_" for character in args.seal_id):
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:semantic seal id")

    raw_inputs = tuple(Path(value) for value in (
        args.manifest, args.train, args.tokenizer_root,
        args.s128_resolved, args.confirmation_data,
    ))
    if any(path.is_symlink() for path in raw_inputs):
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:symlink")
    manifest_path, train_path, tokenizer_root, s128_resolved, confirmation_path = (
        path.resolve() for path in raw_inputs
    )
    resolved_output = Path(args.resolved_output)
    output = Path(args.output)
    for path in (resolved_output.parent, output.parent):
        if path.is_symlink():
            raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:symlink")
    if output.exists() or output.is_symlink() \
            or resolved_output.exists() or resolved_output.is_symlink():
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:output exists")
    if sha256_file(manifest_path) != args.manifest_sha256:
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:manifest SHA")
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA256:
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:train SHA")
    if args.s128_resolved_sha256 != EXPECTED_S128_RESOLVED_SHA256 \
            or sha256_file(s128_resolved) != EXPECTED_S128_RESOLVED_SHA256:
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:S128 resolved SHA")
    if sha256_file(confirmation_path) != args.confirmation_data_sha256:
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:confirmation SHA")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("program") != "RWWPO-2":
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:program")
    minimum = int(manifest["performance"]["confirmation_minimum_examples"])
    training = manifest["training"]
    actor_target = int(training["target_rounds"]) * int(training["train_batch_size"])
    effective_filter = int(training["runtime_effective_prompt_filter_length"])
    actor_rows, actor_scan = _actor_consumed_rows(
        _read_parquet(train_path), tokenizer_root=tokenizer_root,
        target_prompt_groups=actor_target,
        effective_max_prompt_length=effective_filter,
    )
    _, s128_rows = _load_s128(s128_resolved)
    raw_confirmation_rows = list(_read_parquet(confirmation_path))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root, local_files_only=True, trust_remote_code=False
    )
    frozen_rows, filter_evidence = freeze_existing_s128_rows(
        raw_confirmation_rows,
        prompt_token_length=lambda prompt: len(
            tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        ),
        context_token_length=lambda context: len(
            tokenizer.encode(context, add_special_tokens=False)
        ),
        max_prompt_length=int(training["runtime_effective_prompt_filter_length"]),
        max_context_length=(
            int(training["chunk_size"]) * int(training["max_chunks"])
        ),
    )
    if filter_evidence["rejected_raw_positions"]:
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:overlong rows")
    confirmation_rows = [source_identity(row) for row in raw_confirmation_rows]
    for row in confirmation_rows:
        row.pop("prompt", None)
    if len(frozen_rows) != len(confirmation_rows) or len(frozen_rows) < minimum:
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:minimum row count")

    actor_content = {str(row["content_key"]) for row in actor_rows}
    actor_roots = {str(row["root_key"]) for row in actor_rows}
    s128_content = {str(row["content_key"]) for row in s128_rows}
    s128_roots = {str(row["root_key"]) for row in s128_rows}
    confirmation_content = {str(row["content_key"]) for row in confirmation_rows}
    confirmation_roots = {str(row["root_key"]) for row in confirmation_rows}
    duplicates = {
        "content": len(confirmation_rows) - len(confirmation_content),
        "root": len(confirmation_rows) - len(confirmation_roots),
    }
    overlaps = {
        "actor_training_content": len(actor_content & confirmation_content),
        "actor_training_root": len(actor_roots & confirmation_roots),
        "adaptive_s128_content": len(s128_content & confirmation_content),
        "adaptive_s128_root": len(s128_roots & confirmation_roots),
        # Capture32 is an authenticated subset of this exact S128 inventory;
        # zero S128 overlap therefore proves zero Capture32 overlap without
        # opening or duplicating its outcome artifacts.
        "capture32_content": len(s128_content & confirmation_content),
        "capture32_root": len(s128_roots & confirmation_roots),
    }
    if any(duplicates.values()) or any(overlaps.values()):
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:duplicate or overlap")

    performance = manifest["performance"]
    tokenizer_inventory = []
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = tokenizer_root / name
        if path.is_file():
            tokenizer_inventory.append({
                "path": name, "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    if not tokenizer_inventory:
        raise SystemExit("RWWPO2_CONFIRMATION_SEAL_NO_GO:tokenizer inventory")
    identity_payload = {
        "schema_version": 1,
        "namespace": "rwwpo2-sealed-confirmation-stable-identity-v1",
        "source_dataset": {
            "role": "sealed_disjoint_confirmation",
            "parquet_sha256": args.confirmation_data_sha256,
            "raw_rows": len(raw_confirmation_rows),
            "production_effective_rows": len(frozen_rows),
            "shuffle": False,
            "filter_overlong_prompts": True,
            "production_effective_prompt_limit": int(
                training["runtime_effective_prompt_filter_length"]
            ),
        },
        "base_model_protocol": dict(manifest["model"]),
        "tokenizer": {
            "files": tokenizer_inventory,
            "manifest_sha256": canonical_sha256(tokenizer_inventory),
        },
        "identity_construction": {
            "version": 1,
            "example_id": "string form of source extra_info.index",
            "source_order_index": "position after frozen production prompt filter",
            "row_hashes": (
                "UTF-8 SHA-256 for question/context and canonical-JSON SHA-256 "
                "for ground truth"
            ),
        },
        "decode": {
            **performance["confirmation_decoding"], "top_k": -1,
        },
        "backend": dict(manifest["backend"]),
        "rows": frozen_rows,
    }
    eval_manifest_hash = canonical_sha256(identity_payload)
    resolved = {
        "schema_version": 1,
        "identity_payload": identity_payload,
        "eval_manifest_hash": eval_manifest_hash,
        "confirmation_binding": {
            "seal_id": args.seal_id,
            "manifest_sha256": args.manifest_sha256,
            "confirmation_data_path": str(confirmation_path),
            "confirmation_data_sha256": args.confirmation_data_sha256,
            "generation_seed": int(performance["confirmation_generation_seed"]),
            "decoding": {**performance["confirmation_decoding"], "top_k": -1},
            "producer": "experiments/7b_gate_a/run_gate_a.sh:TaskRunner._validate",
        },
    }
    validate_resolved_manifest(resolved)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(resolved, sort_keys=True, indent=2) + "\n")
    resolved_sha256 = sha256_file(resolved_output)
    report = {
        "schema_version": "rwwpo2-confirmation-seal-v1",
        "status": "PASS",
        "decision": "RWWPO2_CONFIRMATION_SEAL_PASS",
        "git_commit": head,
        "seal_id": args.seal_id,
        "manifest_sha256": args.manifest_sha256,
        "confirmation_data_path": str(confirmation_path),
        "confirmation_data_sha256": args.confirmation_data_sha256,
        "row_count": len(frozen_rows),
        "minimum_row_count": minimum,
        "unique_content_count": len(confirmation_content),
        "unique_root_count": len(confirmation_roots),
        "content_inventory_digest": aggregate_digest(confirmation_content),
        "root_inventory_digest": aggregate_digest(confirmation_roots),
        "filter_replay": filter_evidence,
        "duplicates": duplicates,
        "overlaps": overlaps,
        "actor_training_identity": {
            "data_sha256": EXPECTED_TRAIN_SHA256,
            "planned_consumed_prompt_roots": actor_target,
            **actor_scan,
        },
        "adaptive_s128_identity": {
            "resolved_sha256": EXPECTED_S128_RESOLVED_SHA256,
            "row_count": len(s128_rows),
            "capture32_relation": "authenticated subset; disjointness inherited",
        },
        "identity_definition": {
            "root_key": "sha256(canonical(question_hash,context_hash))",
            "content_key": "sha256(canonical(question_hash,context_hash,ground_truth_hash))",
            "raw_row_values_emitted": False,
        },
        "generation_contract": {
            "seed": int(performance["confirmation_generation_seed"]),
            "decoding": performance["confirmation_decoding"],
            "model": manifest["model"],
            "max_prompt_length": int(training["max_prompt_length"]),
            "max_response_length": int(training["max_response_length"]),
            "producer": "experiments/7b_gate_a/run_gate_a.sh:TaskRunner._validate",
            "stable_join": "canonical root/content identity exact join",
            "missingness": "all sealed roots remain in denominator; missing generation fails",
        },
        "resolved_identity_manifest_path": str(resolved_output.resolve()),
        "resolved_identity_manifest_sha256": resolved_sha256,
        "eval_manifest_hash": eval_manifest_hash,
        "opening_policy": (
            "one evaluation opening only after all B/D/E eight-seed R400 attempts, "
            "attempt audits, mechanism reports, and code are frozen"
        ),
        "opened_for_training_or_selection": False,
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": report["decision"],
        "seal_id": args.seal_id, "row_count": len(confirmation_rows),
        "output": str(output.resolve()), "report_sha256": report["report_sha256"],
        "resolved_output": str(resolved_output.resolve()),
        "resolved_sha256": resolved_sha256, "eval_manifest_hash": eval_manifest_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

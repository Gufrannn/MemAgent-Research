#!/usr/bin/env python3
"""Evidence-producing six-cell fixture audit for the BABILong adapter."""
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
    LENGTHS, TASK_DEPTH, adapt_source_row, score_babilong_output,
    validate_frozen_contract,
)
from recurrent.research.stable_eval_identity import canonical_sha256


FIXTURES = {
    ("32k", "qa1"): ("Mary went to the kitchen.", "Where is Mary?", "The most recent location of Mary is kitchen."),
    ("32k", "qa2"): ("John got the apple. John went to the hall.", "Where is the apple?", "The apple is in the hall."),
    ("32k", "qa3"): ("Mary got the apple. Mary went to the hall. Mary went to the kitchen.", "Where was the apple before the kitchen?", "Before the kitchen the apple was in the hall."),
    ("128k", "qa1"): ("Daniel went to the office.", "Where is Daniel?", "The most recent location of Daniel is office."),
    ("128k", "qa2"): ("Sandra got the football. Sandra went to the garden.", "Where is the football?", "The football is in the garden."),
    ("128k", "qa3"): ("Sandra got the football. Sandra went to the garden. Sandra went to the bedroom.", "Where was the football before the bedroom?", "Before the bedroom the football was in the garden."),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_BABILONG_FIXTURE_NO_GO:checkout")
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output)
    if manifest_path.is_symlink() or output.exists() or output.is_symlink():
        raise SystemExit("RWWPO2_BABILONG_FIXTURE_NO_GO:symlink/append-only")
    if sha256_file(manifest_path) != args.manifest_sha256:
        raise SystemExit("RWWPO2_BABILONG_FIXTURE_NO_GO:manifest SHA")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_frozen_contract(manifest)
    rows = []
    identities = set()
    for order, ((length, task), (context, question, target)) in enumerate(FIXTURES.items()):
        adapted, identity = adapt_source_row(
            {"input": context, "question": question, "target": target},
            length=length, task=task, source_index=0, partition="fixture",
            source_order_index=order, context_token_length=lambda text: len(text.split()),
        )
        positive = score_babilong_output(f"\\boxed{{{target}}}", target)
        negative = score_babilong_output("\\boxed{definitely wrong}", target)
        if positive["official_accuracy"] != 1.0 or positive["exact_match"] != 1.0 \
                or positive["format_success"] != 1.0 \
                or negative["official_accuracy"] != 0.0 \
                or negative["exact_match"] != 0.0:
            raise SystemExit("RWWPO2_BABILONG_FIXTURE_NO_GO:metric reconstruction")
        source_identity = identity["babilong_source_identity"]
        if source_identity in identities:
            raise SystemExit("RWWPO2_BABILONG_FIXTURE_NO_GO:identity collision")
        identities.add(source_identity)
        rows.append({
            "length": length, "task": task, "depth": TASK_DEPTH[task],
            "prompt_sha256": identity["source_question_hash"],
            "context_sha256": identity["source_context_hash"],
            "ground_truth_sha256": identity["ground_truth_hash"],
            "source_identity": source_identity,
            "adapted_schema_sha256": canonical_sha256(adapted),
            "positive": {key: positive[key] for key in (
                "official_accuracy", "exact_match", "token_f1", "format_success"
            )},
            "negative": {key: negative[key] for key in (
                "official_accuracy", "exact_match", "token_f1", "format_success"
            )},
        })
    expected_cells = {(length, task) for length in LENGTHS for task in TASK_DEPTH}
    if {(row["length"], row["task"]) for row in rows} != expected_cells:
        raise SystemExit("RWWPO2_BABILONG_FIXTURE_NO_GO:six-cell coverage")
    report = {
        "schema_version": "rwwpo2-babilong-fixture-audit-v1",
        "status": "PASS", "decision": "RWWPO2_BABILONG_FIXTURE_AUDIT_PASS",
        "git_commit": head, "manifest_sha256": args.manifest_sha256,
        "fixture_count": len(rows), "cell_count": len(expected_cells),
        "fixture_inventory_sha256": canonical_sha256(rows), "fixtures": rows,
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "decision": report["decision"],
        "fixtures": len(rows), "output": str(output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

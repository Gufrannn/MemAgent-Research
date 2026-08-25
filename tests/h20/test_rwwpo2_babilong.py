import copy
import hashlib
import json
from pathlib import Path

import pytest

from recurrent.research.rwwpo2_babilong import (
    DEVELOPMENT_SOURCE_INDICES,
    LENGTHS,
    TASK_DEPTH,
    adapt_partition,
    adapt_source_row,
    expected_development_indices,
    official_substring_accuracy,
    paired_descriptive_difference,
    partition_indices,
    score_babilong_output,
    summarize_babilong_metrics,
    validate_frozen_contract,
)
from recurrent.research.stable_eval_identity import canonical_sha256
from tools.h20.compare_rwwpo2_babilong import authenticate_reports
from tools.h20.materialize_rwwpo2_babilong import read_source_bundle, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def source_rows():
    result = {}
    for length in LENGTHS:
        for task, depth in TASK_DEPTH.items():
            result[(length, task)] = [
                {
                    "input": f"noise {index}. fact for {length} depth {depth}.",
                    "question": f"fixture question {length} {task} {index}?",
                    "target": f"Fixture answer {length} {task} {index}.",
                }
                for index in range(100)
            ]
    return result


def test_contract_and_outcome_blind_membership_are_frozen():
    manifest = json.loads(
        (ROOT / "manifests/h20/rwwpo2_babilong_pilot_v1.json").read_text()
    )
    validate_frozen_contract(manifest)
    for length in LENGTHS:
        for task in TASK_DEPTH:
            assert expected_development_indices(length, task) == DEVELOPMENT_SOURCE_INDICES[length][task]
            assert len(partition_indices(length, task, "development")) == 8
            assert len(partition_indices(length, task, "confirmation")) == 92
            assert not set(partition_indices(length, task, "development")) & set(
                partition_indices(length, task, "confirmation")
            )


def test_six_cell_fixture_adapter_has_stable_unique_identity():
    adapted = adapt_partition(
        source_rows(), partition="development",
        context_token_length=lambda text: len(text.split()),
    )
    all_ids = set()
    for length in LENGTHS:
        rows, identities = adapted[length]
        assert len(rows) == len(identities) == 24
        assert [row["source_order_index"] for row in identities] == list(range(24))
        assert {row["extra_info"]["babilong_depth"] for row in rows} == {1, 2, 3}
        assert all("\\boxed{<answer sentence>}" in row["prompt"][0]["content"] for row in rows)
        assert all(identity["example_id"] not in all_ids for identity in identities)
        all_ids.update(identity["example_id"] for identity in identities)
    assert len(all_ids) == 48


@pytest.mark.parametrize(
    "length,task,context,question,target",
    [
        ("32k", "qa1", "Mary went to the kitchen.", "Where is Mary?", "The most recent location of Mary is kitchen."),
        ("32k", "qa2", "John got the apple. John went to the hall.", "Where is the apple?", "The apple is in the hall."),
        ("32k", "qa3", "Mary got the apple. Mary went to the hall. Mary went to the kitchen.", "Where was the apple before the kitchen?", "Before the kitchen the apple was in the hall."),
        ("128k", "qa1", "Daniel went to the office.", "Where is Daniel?", "The most recent location of Daniel is office."),
        ("128k", "qa2", "Sandra got the football. Sandra went to the garden.", "Where is the football?", "The football is in the garden."),
        ("128k", "qa3", "Sandra got the football. Sandra went to the garden. Sandra went to the bedroom.", "Where was the football before the bedroom?", "Before the bedroom the football was in the garden."),
    ],
)
def test_manual_six_cell_fixture_scoring(length, task, context, question, target):
    row, identity = adapt_source_row(
        {"input": context, "question": question, "target": target},
        length=length, task=task, source_index=0, partition="fixture",
        source_order_index=0, context_token_length=lambda text: len(text.split()),
    )
    assert row["reward_model"]["ground_truth"] == [target]
    assert identity["babilong_depth"] == TASK_DEPTH[task]
    score = score_babilong_output(f"\\boxed{{{target}}}", target)
    assert score["official_accuracy"] == 1.0
    assert score["exact_match"] == 1.0
    assert score["format_success"] == 1.0


def test_official_metric_and_strict_metric_are_both_visible():
    target = "The apple is in the kitchen."
    verbose = f"Reasoning that would fail strict extraction. {target} Extra words."
    assert official_substring_accuracy(verbose, target) == 1.0
    score = score_babilong_output(verbose, target)
    assert score["official_accuracy"] == 1.0
    assert score["exact_match"] == 0.0


def test_capacity_and_source_schema_fail_closed():
    source = {"input": "x", "question": "q?", "target": "answer."}
    with pytest.raises(ValueError, match="truncation is forbidden"):
        adapt_source_row(
            source, length="32k", task="qa1", source_index=0,
            partition="fixture", source_order_index=0,
            context_token_length=lambda _: 40_001,
        )
    broken = copy.deepcopy(source_rows())
    broken[("32k", "qa1")].pop()
    with pytest.raises(ValueError, match="exactly 100"):
        adapt_partition(
            broken, partition="development", context_token_length=lambda _: 1
        )


def test_grouped_metrics_and_paired_descriptive_difference():
    rows = []
    for length in LENGTHS:
        for task, depth in TASK_DEPTH.items():
            target = f"answer {length} {task}"
            score = score_babilong_output(f"\\boxed{{{target}}}", target)
            rows.append({
                **score, "babilong_length": length, "babilong_task": task,
                "babilong_depth": depth,
                "babilong_source_identity": f"{length}-{task}",
            })
    aggregate = summarize_babilong_metrics(rows)
    assert aggregate["overall"]["official_accuracy"] == 1.0
    assert len(aggregate["by_cell"]) == 6
    right = [dict(row, official_accuracy=0.0, token_f1=0.0, exact_match=0.0) for row in rows]
    paired = paired_descriptive_difference(rows, right, left_name="B", right_name="D")
    assert paired["denominator"] == 6
    assert paired["official_accuracy"]["mean_difference"] == 1.0


def _write_signed_source_bundle(root: Path, *, commit: str, manifest_sha: str) -> Path:
    cells = []
    for length in LENGTHS:
        for task in TASK_DEPTH:
            rows = source_rows()[(length, task)]
            path = root / "source" / length / f"{task}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ))
            cells.append({
                "length": length, "task": task, "rows": 100,
                "relative_path": f"source/{length}/{task}.jsonl",
                "file_sha256": sha256_file(path),
                "canonical_rows_sha256": canonical_sha256(rows),
            })
    report = {
        "schema_version": "rwwpo2-babilong-source-bundle-v1",
        "status": "PASS", "decision": "RWWPO2_BABILONG_SOURCE_BUNDLE_PASS",
        "git_commit": commit, "adapter_manifest_sha256": manifest_sha,
        "dataset_id": "RMT-team/babilong",
        "dataset_revision": "e3a924b6686759422257925a695cbbb4b2684936",
        "cells": cells, "cell_inventory_sha256": canonical_sha256(cells),
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    path = root / "bundle_manifest.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return path


def test_source_bundle_authentication_rejects_path_and_inventory_drift(tmp_path):
    commit = "a" * 40
    manifest_sha = "b" * 64
    bundle_path = _write_signed_source_bundle(
        tmp_path, commit=commit, manifest_sha=manifest_sha
    )
    _, rows = read_source_bundle(
        tmp_path, expected_sha256=sha256_file(bundle_path),
        expected_commit=commit, adapter_manifest_sha256=manifest_sha,
    )
    assert set(rows) == {(length, task) for length in LENGTHS for task in TASK_DEPTH}

    report = json.loads(bundle_path.read_text())
    report["cells"][0]["relative_path"] = "../escaped.jsonl"
    report["cell_inventory_sha256"] = canonical_sha256(report["cells"])
    report.pop("report_sha256")
    report["report_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    bundle_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="path/count"):
        read_source_bundle(
            tmp_path, expected_sha256=sha256_file(bundle_path),
            expected_commit=commit, adapter_manifest_sha256=manifest_sha,
        )


def _write_eval_report(path: Path, rows: Path, *, commit: str, cell: str, length: str):
    report = {
        "schema_version": "rwwpo2-babilong-eval-audit-v1",
        "status": "DIAGNOSTIC_ONLY",
        "decision": "RWWPO2_BABILONG_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "git_commit": commit, "partition": "development", "cell": cell,
        "length": length, "evaluation_step": 20,
        "metric_rows_path": str(rows.resolve()),
        "metric_rows_sha256": sha256_file(rows),
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


def test_paired_report_authentication_binds_metric_rows(tmp_path):
    commit = "c" * 40
    reports = []
    rows = []
    for length in LENGTHS:
        row_path = tmp_path / f"{length}.jsonl"
        row_path.write_text(json.dumps({"length": length}) + "\n")
        report_path = tmp_path / f"{length}.json"
        _write_eval_report(
            report_path, row_path, commit=commit, cell="B", length=length
        )
        rows.append(str(row_path))
        reports.append(str(report_path))
    authenticated = authenticate_reports(reports, rows, expected_commit=commit)
    assert {row["length"] for row in authenticated} == set(LENGTHS)
    Path(rows[0]).write_text(json.dumps({"length": "tampered"}) + "\n")
    with pytest.raises(ValueError, match="authentication"):
        authenticate_reports(reports, rows, expected_commit=commit)

#!/usr/bin/env python3
"""Run and freeze the mandatory RWWPO-2 related-regression suite."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.h20.verify_rwwpo2_release_tests import (
    EXPECTED_BRANCH, TEST_INVENTORY, canonical_sha, junit_summary,
    node_evidence, pytest_command, pytest_environment, runtime_environment,
    sha256_file,
)


def evidence(path: Path, root: Path) -> dict:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_logged(command: list[str], *, log: Path) -> int:
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, env=pytest_environment(), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            stream.write(line)
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT,
                                     text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                    text=True).strip()
    if head != args.expected_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_RELEASE_TEST_NO_GO:checkout")
    manifest = Path(args.manifest)
    if manifest.is_symlink() or not manifest.is_file() \
            or sha256_file(manifest) != args.manifest_sha256:
        raise SystemExit("RWWPO2_RELEASE_TEST_NO_GO:manifest")
    if json.loads(manifest.read_text()).get("program") != "RWWPO-2":
        raise SystemExit("RWWPO2_RELEASE_TEST_NO_GO:program")
    work_root = Path(args.work_root).resolve()
    output_root = Path(args.output_root)
    expected_parent = work_root / "logs/rwwpo2_release_tests"
    if not output_root.is_absolute() or output_root.parent.resolve() != expected_parent \
            or not output_root.name:
        raise SystemExit("RWWPO2_RELEASE_TEST_NO_GO:one-use output root")
    source_sha256 = {}
    for relative in TEST_INVENTORY:
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise SystemExit("RWWPO2_RELEASE_TEST_NO_GO:test inventory")
        source_sha256[relative] = sha256_file(source)
    expected_parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    tombstone = output_root / "RUN_ID_CONSUMED"
    tombstone.write_text(f"{head}:{args.manifest_sha256}\n", encoding="utf-8")
    collect_log = output_root / "pytest_collect.log"
    collection_json = output_root / "collection.json"
    collection_returncode = run_logged(
        pytest_command(mode="collect", evidence_path=collection_json),
        log=collect_log,
    )
    try:
        collection = node_evidence(collection_json, mode="collect")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        collection = {
            "pytest_exitstatus": collection_returncode,
            "collected_node_ids": [],
        }
    log = output_root / "pytest.log"
    junit = output_root / "pytest.xml"
    execution_json = output_root / "execution.json"
    if collection_returncode == 0 and collection.get("collected_node_ids"):
        returncode = run_logged(
            pytest_command(
                mode="execute", evidence_path=execution_json, junit=junit),
            log=log,
        )
    else:
        log.write_text("execution withheld: collection did not PASS\n", encoding="utf-8")
        returncode = collection_returncode or 1
    try:
        execution = node_evidence(execution_json, mode="execute")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        execution = {"pytest_exitstatus": returncode, "collected_node_ids": []}
    summary = junit_summary(junit) if junit.is_file() else {
        "tests": 0, "failures": 0, "errors": 1, "skipped": 0,
        "unique_test_ids": 0, "test_case_ids_sha256": "0" * 64,
    }
    post_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    post_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    post_dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    source_stable = all(
        sha256_file(ROOT / relative) == digest
        for relative, digest in source_sha256.items()
    )
    checkout_postcondition = post_head == head and post_branch == branch \
        and not post_dirty and source_stable \
        and sha256_file(manifest) == args.manifest_sha256
    nodes = collection.get("collected_node_ids", [])
    passed = collection_returncode == 0 and returncode == 0 \
        and int(collection.get("pytest_exitstatus", -1)) == 0 \
        and int(execution.get("pytest_exitstatus", -1)) == 0 \
        and nodes == execution.get("collected_node_ids") \
        and summary["tests"] == len(nodes) and summary["tests"] > 0 \
        and summary["tests"] == summary["unique_test_ids"] \
        and all(summary[name] == 0 for name in ("failures", "errors", "skipped")) \
        and checkout_postcondition
    row = {
        "schema_version": "rwwpo2-release-tests-v1",
        "status": "PASS" if passed else "NO_GO",
        "decision": "RWWPO2_RELEASE_TESTS_PASS" if passed else "RWWPO2_RELEASE_TESTS_NO_GO",
        "git_commit": head,
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": args.manifest_sha256,
        "run_id": output_root.name,
        "runtime_environment": runtime_environment(),
        "pytest_collection_returncode": collection_returncode,
        "pytest_returncode": returncode,
        "checkout_postcondition": checkout_postcondition,
        "test_inventory": list(TEST_INVENTORY),
        "test_source_sha256": source_sha256,
        "collected_node_ids": nodes,
        "run_id_tombstone": evidence(tombstone, output_root),
        "pytest_collect_log": evidence(collect_log, output_root),
        "collection_evidence": (
            evidence(collection_json, output_root) if collection_json.is_file() else None
        ),
        "pytest_log": evidence(log, output_root),
        "execution_evidence": (
            evidence(execution_json, output_root) if execution_json.is_file() else None
        ),
        "junit_xml": evidence(junit, output_root) if junit.is_file() else None,
        "junit_summary": summary,
    }
    row["report_sha256"] = canonical_sha(row)
    report = output_root / "release_tests.json"
    with report.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": row["status"], "decision": row["decision"],
        "output": str(report), "tests": summary["tests"],
    }, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

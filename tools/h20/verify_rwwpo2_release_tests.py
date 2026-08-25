#!/usr/bin/env python3
"""Verify an append-only RWWPO-2 related-regression receipt and raw evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_BRANCH = "h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822"
TEST_INVENTORY = (
    "tests/h20/test_rwwpo2_program.py",
    "tests/h20/test_rwwpo2_babilong.py",
    "tests/h20/test_rwwpo2_tensor_ledger.py",
    "tests/h20/test_rwwpo_core.py",
    "tests/h20/test_rwwpo_entrypoints.py",
    "tests/h20/test_rwwpo_transaction.py",
    "tests/h20/test_tf_rwwpo_budget_leakage.py",
    "recurrent/research/tests/test_actor_batch.py",
)
PYTEST_PLUGIN = "tools.h20.rwwpo2_pytest_evidence_plugin"
CRITICAL_IMPORTS = ("pytest", "torch", "transformers", "ray", "vllm", "verl")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(row: dict) -> str:
    payload = dict(row)
    payload.pop("report_sha256", None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def runtime_environment() -> dict:
    executable = Path(os.path.realpath(sys.executable))
    distributions = sorted(
        (str(distribution.metadata.get("Name", "")).lower(), distribution.version)
        for distribution in importlib.metadata.distributions()
    )
    distribution_raw = json.dumps(distributions, separators=(",", ":"))
    origins = {}
    origin_sha256 = {}
    for name in CRITICAL_IMPORTS:
        spec = importlib.util.find_spec(name)
        origins[name] = None if spec is None or spec.origin is None \
            else os.path.realpath(spec.origin)
        origin = None if origins[name] is None else Path(origins[name])
        origin_sha256[name] = (
            None if origin is None or not origin.is_file() else sha256_file(origin)
        )
    try:
        import torch
        torch_runtime = {
            "version": str(torch.__version__),
            "cuda_build": str(torch.version.cuda),
            "git_version": str(torch.version.git_version),
            "build_config_sha256": hashlib.sha256(
                torch.__config__.show().encode()).hexdigest(),
            "origin": os.path.realpath(torch.__file__),
        }
    except (ImportError, OSError) as error:
        torch_runtime = {"error": f"{type(error).__name__}:{error}"}
    try:
        pytest_version = importlib.metadata.version("pytest")
    except importlib.metadata.PackageNotFoundError:
        pytest_version = "MISSING"
    row = {
        "python_executable": str(executable),
        "python_executable_sha256": sha256_file(executable),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_cache_tag": sys.implementation.cache_tag,
        "pytest_version": pytest_version,
        "torch_runtime": torch_runtime,
        "critical_import_origins": origins,
        "critical_import_origin_sha256": origin_sha256,
        "sys_path": [os.path.realpath(value or os.getcwd()) for value in sys.path],
        "installed_distribution_count": len(distributions),
        "installed_distributions_sha256": hashlib.sha256(
            distribution_raw.encode()).hexdigest(),
    }
    row["environment_sha256"] = canonical_sha(row)
    return row


def pytest_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTEST_ADDOPTS", None)
    return environment


def pytest_command(
        *, mode: str, evidence_path: Path, junit: Path | None = None) -> list[str]:
    command = [
        sys.executable, "-m", "pytest", "-q", "--noconftest",
        "-p", PYTEST_PLUGIN,
        f"--rwwpo2-evidence-json={evidence_path}",
        f"--rwwpo2-evidence-mode={mode}",
    ]
    if mode == "collect":
        command.append("--collect-only")
    if junit is not None:
        command.append(f"--junitxml={junit}")
    return [*command, *TEST_INVENTORY]


def node_evidence(path: Path, *, mode: str) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    nodeids = row.get("collected_node_ids")
    if row.get("schema_version") != "rwwpo2-pytest-node-evidence-v1" \
            or row.get("mode") != mode or not isinstance(nodeids, list) \
            or not nodeids or any(not isinstance(value, str) or not value
                                   for value in nodeids) \
            or len(nodeids) != len(set(nodeids)):
        raise ValueError("release-test node inventory")
    for relative in TEST_INVENTORY:
        if not any(nodeid.startswith(relative + "::") for nodeid in nodeids):
            raise ValueError("release-test file missing from node inventory")
    if mode == "collect":
        if row.get("phase_reports") != {nodeid: [] for nodeid in nodeids}:
            raise ValueError("release-test collect-only outcomes")
    else:
        reports = row.get("phase_reports")
        if not isinstance(reports, dict) or set(reports) != set(nodeids):
            raise ValueError("release-test execution outcomes")
        for nodeid in nodeids:
            phases = reports[nodeid]
            if [(phase.get("when"), phase.get("outcome"), phase.get("wasxfail"))
                    for phase in phases] != [
                        ("setup", "passed", False),
                        ("call", "passed", False),
                        ("teardown", "passed", False),
                    ]:
                raise ValueError("release-test non-PASS/skip/xfail phase")
    return row


def collect_current_node_ids() -> list[str]:
    with tempfile.TemporaryDirectory(prefix="rwwpo2-collect-") as temp:
        evidence = Path(temp) / "collection.json"
        process = subprocess.run(
            pytest_command(mode="collect", evidence_path=evidence),
            cwd=ROOT, env=pytest_environment(), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
        if process.returncode != 0 or not evidence.is_file():
            raise ValueError("release-test current collection failed")
        row = node_evidence(evidence, mode="collect")
        if int(row.get("pytest_exitstatus", -1)) != 0:
            raise ValueError("release-test current collection exit status")
        return row["collected_node_ids"]


def junit_summary(path: Path) -> dict:
    root = ET.parse(path).getroot()
    cases = []
    failures = errors = skipped = 0
    for case in root.iter("testcase"):
        identifier = f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
        cases.append(identifier)
        failures += int(case.find("failure") is not None)
        errors += int(case.find("error") is not None)
        skipped += int(case.find("skipped") is not None)
    ordered = sorted(cases)
    raw = json.dumps(ordered, separators=(",", ":"))
    return {
        "tests": len(ordered),
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "unique_test_ids": len(set(ordered)),
        "test_case_ids_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }


def _authenticated_child(root: Path, evidence: dict) -> Path:
    relative = evidence.get("relative_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("release-test evidence path")
    path = root / relative
    resolved = path.resolve()
    if root not in resolved.parents or path.is_symlink() or not path.is_file():
        raise ValueError("release-test evidence escape/symlink/missing")
    if path.stat().st_size != int(evidence.get("size", -1)) \
            or sha256_file(path) != evidence.get("sha256"):
        raise ValueError("release-test evidence byte drift")
    return path


def verify_release_test_receipt(
        receipt_path: str | Path, *, receipt_sha256: str,
        expected_commit: str, manifest_path: str | Path,
        manifest_sha256: str, work_root: str | Path) -> dict:
    receipt = Path(receipt_path)
    if receipt.is_symlink() or not receipt.is_file() \
            or sha256_file(receipt) != receipt_sha256:
        raise ValueError("release-test receipt file identity")
    row = json.loads(receipt.read_text(encoding="utf-8"))
    if row.get("schema_version") != "rwwpo2-release-tests-v1" \
            or canonical_sha(row) != row.get("report_sha256") \
            or row.get("status") != "PASS" \
            or row.get("decision") != "RWWPO2_RELEASE_TESTS_PASS" \
            or row.get("git_commit") != expected_commit \
            or row.get("manifest_sha256") != manifest_sha256 \
            or row.get("test_inventory") != list(TEST_INVENTORY) \
            or row.get("checkout_postcondition") is not True \
            or int(row.get("pytest_collection_returncode", -1)) != 0 \
            or int(row.get("pytest_returncode", -1)) != 0:
        raise ValueError("release-test receipt semantics")
    manifest = Path(manifest_path)
    if manifest.is_symlink() or not manifest.is_file() \
            or sha256_file(manifest) != manifest_sha256:
        raise ValueError("release-test manifest identity")
    if Path(str(row.get("manifest_path", ""))).resolve() != manifest.resolve():
        raise ValueError("release-test runtime/manifest binding")
    root = receipt.resolve().parent
    expected_parent = Path(work_root).resolve() / "logs/rwwpo2_release_tests"
    if root.parent != expected_parent or root.name != row.get("run_id") \
            or root.is_symlink():
        raise ValueError("release-test one-use root identity")
    if {path.name for path in root.iterdir()} != {
            "RUN_ID_CONSUMED", "pytest_collect.log", "collection.json",
            "pytest.log", "execution.json", "pytest.xml", "release_tests.json"} \
            or any(path.is_symlink() or not path.is_file() for path in root.iterdir()):
        raise ValueError("release-test root inventory")
    tombstone = _authenticated_child(root, row.get("run_id_tombstone", {}))
    if tombstone.read_text(encoding="utf-8").strip() != \
            f"{expected_commit}:{manifest_sha256}":
        raise ValueError("release-test tombstone identity")
    _authenticated_child(root, row.get("pytest_collect_log", {}))
    collection_path = _authenticated_child(root, row.get("collection_evidence", {}))
    collection = node_evidence(collection_path, mode="collect")
    _authenticated_child(root, row.get("pytest_log", {}))
    execution_path = _authenticated_child(root, row.get("execution_evidence", {}))
    execution = node_evidence(execution_path, mode="execute")
    junit = _authenticated_child(root, row.get("junit_xml", {}))
    if int(collection.get("pytest_exitstatus", -1)) != 0 \
            or int(execution.get("pytest_exitstatus", -1)) != 0 \
            or collection["collected_node_ids"] != execution["collected_node_ids"] \
            or collection["collected_node_ids"] != row.get("collected_node_ids") \
            or collect_current_node_ids() != collection["collected_node_ids"]:
        raise ValueError("release-test collected/executed node drift")
    if junit_summary(junit) != row.get("junit_summary") \
            or row["junit_summary"]["tests"] < 1 \
            or any(row["junit_summary"][name] != 0
                   for name in ("failures", "errors", "skipped")) \
            or row["junit_summary"]["unique_test_ids"] != \
                row["junit_summary"]["tests"] \
            or row["junit_summary"]["tests"] != len(row["collected_node_ids"]):
        raise ValueError("release-test JUnit semantics")
    expected_sources = {}
    for relative in TEST_INVENTORY:
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError("release-test source identity")
        expected_sources[relative] = sha256_file(source)
    if row.get("test_source_sha256") != expected_sources:
        raise ValueError("release-test source drift")
    if row.get("runtime_environment") != runtime_environment():
        raise ValueError("release-test Python environment drift")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--work-root", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT,
                                     text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                    text=True).strip()
    if head != args.expected_commit or branch != EXPECTED_BRANCH or dirty:
        raise SystemExit("RWWPO2_RELEASE_TEST_VERIFY_NO_GO:checkout")
    try:
        row = verify_release_test_receipt(
            args.receipt, receipt_sha256=args.receipt_sha256,
            expected_commit=args.expected_commit, manifest_path=args.manifest,
            manifest_sha256=args.manifest_sha256, work_root=args.work_root,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            ET.ParseError) as error:
        raise SystemExit("RWWPO2_RELEASE_TEST_VERIFY_NO_GO:" + str(error)) from error
    print(json.dumps({
        "status": "PASS", "decision": "RWWPO2_RELEASE_TEST_VERIFY_PASS",
        "git_commit": head, "receipt_report_sha256": row["report_sha256"],
        "tests": row["junit_summary"]["tests"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

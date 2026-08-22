"""Read-only authority validation for CORAL's shared Original/S128 evidence."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from recurrent.research.cosi import canonical_sha256, sha256_file
from recurrent.research.gate_a_execution import validate_jsonl_chain
from recurrent.research.stable_eval_identity import validate_resolved_manifest


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"CORAL_AUTHORITY_NO_GO: JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"CORAL_AUTHORITY_NO_GO: non-empty JSONL required: {path}")
    failures = validate_jsonl_chain(rows)
    if failures:
        raise ValueError(f"CORAL_AUTHORITY_NO_GO: ledger chain {path}: {failures}")
    return rows


def _bound_path(spec: Mapping[str, Any], field: str) -> Path:
    value = spec.get(field)
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"CORAL_AUTHORITY_NO_GO: absolute authority path {field}")
    path = Path(value).resolve()
    if not path.is_file():
        raise ValueError(f"CORAL_AUTHORITY_NO_GO: missing authority file {field}: {path}")
    return path


def _exact_file(spec: Mapping[str, Any], field: str, sha_field: str) -> Path:
    path = _bound_path(spec, field)
    expected = spec.get(sha_field)
    if not _sha(expected) or sha256_file(path) != expected:
        raise ValueError(f"CORAL_AUTHORITY_NO_GO: {field} SHA mismatch")
    return path


def validate_original_training_authority(
    spec: Mapping[str, Any], *, resolved_path: Path, expected_resolved_sha256: str,
) -> dict[str, Any]:
    """Authenticate B and return its P0 frozen trainer evidence."""
    if not _sha(expected_resolved_sha256) \
            or Path(spec.get("resolved", "")).resolve() != resolved_path.resolve() \
            or sha256_file(resolved_path) != expected_resolved_sha256:
        raise ValueError("CORAL_AUTHORITY_NO_GO: Original training resolved binding")
    final_path = _exact_file(spec, "final_report", "final_sha256")
    ledger_path = _exact_file(spec, "ledger", "ledger_sha256")
    final = _json(final_path)
    ledger = _jsonl(ledger_path)
    if final.get("status") != "PASS" or final.get("decision") != "ORIGINAL_T25_PASS" \
            or final.get("git_commit") != spec.get("git_commit") \
            or ledger[-1].get("record_sha256") != spec.get("ledger_tail") \
            or ledger[-1].get("report_sha256") != spec.get("final_sha256"):
        raise ValueError("CORAL_AUTHORITY_NO_GO: Original training final/ledger identity")
    resolved = _json(resolved_path)
    p0_path = (final_path.parent / "p0_preflight.json").resolve()
    if not p0_path.is_file() or sha256_file(p0_path) != final.get("p0_certificate_sha256"):
        raise ValueError("CORAL_AUTHORITY_NO_GO: Original P0 binding")
    p0 = _json(p0_path)
    if p0.get("status") != "PASS" or p0.get("decision") != "T25_P0_PASS" \
            or Path(str(p0.get("evidence", {}).get("resolved_manifest_path", ""))).resolve() \
            != resolved_path.resolve() \
            or p0.get("evidence", {}).get("resolved_manifest_sha256") \
            != canonical_sha256(resolved):
        raise ValueError("CORAL_AUTHORITY_NO_GO: Original P0 status/resolved binding")
    return {
        "resolved": resolved, "final": final, "p0": p0,
        "resolved_sha256": sha256_file(resolved_path),
        "p0_path": str(p0_path), "p0_sha256": sha256_file(p0_path),
        "final_sha256": sha256_file(final_path), "ledger_sha256": sha256_file(ledger_path),
        "ledger_tail": ledger[-1]["record_sha256"],
    }


def validate_stable_s128_authority(
    spec: Mapping[str, Any], *, resolved_path: Path, expected_resolved_sha256: str,
) -> dict[str, Any]:
    """Authenticate C and return the normalized fixed-S128 identity manifest."""
    if spec.get("resolved_sha256") != expected_resolved_sha256 \
            or Path(spec.get("resolved", "")).resolve() != resolved_path.resolve() \
            or not _sha(expected_resolved_sha256) \
            or sha256_file(resolved_path) != expected_resolved_sha256:
        raise ValueError("CORAL_AUTHORITY_NO_GO: stable-S128 resolved binding")
    final_path = _exact_file(spec, "final_report", "final_sha256")
    ledger_path = _exact_file(spec, "ledger", "ledger_sha256")
    final = _json(final_path)
    ledger = _jsonl(ledger_path)
    resolved = validate_resolved_manifest(_json(resolved_path))
    if resolved.get("eval_manifest_hash") != spec.get("eval_manifest_hash") \
            or final.get("status") != "PASS" \
            or final.get("decision") != "I_RECURRENT_IDENTITY_CANARY_PASS" \
            or ledger[-1].get("record_sha256") != spec.get("ledger_tail") \
            or ledger[-1].get("decision") != "I_RECURRENT_IDENTITY_CANARY_PASS" \
            or ledger[-1].get("artifact_sha256") != spec.get("final_sha256"):
        raise ValueError("CORAL_AUTHORITY_NO_GO: stable-S128 final/ledger identity")
    return {
        "resolved": resolved, "final": final,
        "resolved_sha256": sha256_file(resolved_path),
        "final_sha256": sha256_file(final_path), "ledger_sha256": sha256_file(ledger_path),
        "ledger_tail": ledger[-1]["record_sha256"],
        "stable_inventory_sha256": canonical_sha256([
            str(row["example_id"]) for row in resolved["identity_payload"]["rows"]
        ]),
    }


def validate_curve_authority(
    spec: Mapping[str, Any], *, stable_resolved: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate A through its ledger and externally frozen row digests."""
    final_path = _bound_path(spec, "final_report")
    p0_path = _bound_path(spec, "p0")
    resolved_path = _bound_path(spec, "resolved")
    ledger_path = _bound_path(spec, "ledger")
    final = _json(final_path)
    p0 = _json(p0_path)
    resolved = validate_resolved_manifest(_json(resolved_path))
    ledger = _jsonl(ledger_path)
    expected_digests = spec.get("canonical_metric_row_digests")
    if not isinstance(expected_digests, dict) or any(not _sha(value) for value in expected_digests.values()):
        raise ValueError("CORAL_AUTHORITY_NO_GO: curve canonical digest contract")
    if final.get("status") != "PASS" or final.get("decision") != "ORIGINAL_S128_CURVE_PASS" \
            or final.get("failures") != [] \
            or final.get("evidence", {}).get("metric_rows_sha256") != expected_digests \
            or final.get("evidence", {}).get("eval_manifest_hash") != stable_resolved.get("eval_manifest_hash"):
        raise ValueError("CORAL_AUTHORITY_NO_GO: curve final report identity")
    if p0.get("status") != "PASS" or p0.get("decision") != "ORIGINAL_S128_CURVE_P0_PASS" \
            or p0.get("evidence", {}).get("resolved_manifest_sha256") != sha256_file(resolved_path):
        raise ValueError("CORAL_AUTHORITY_NO_GO: curve P0/resolved identity")
    if resolved.get("identity_payload") != stable_resolved.get("identity_payload"):
        raise ValueError("CORAL_AUTHORITY_NO_GO: curve/stable S128 identity drift")
    tail = ledger[-1]
    if tail.get("status") != "PASS" or tail.get("decision") != "ORIGINAL_S128_CURVE_PASS" \
            or Path(str(tail.get("artifact", ""))).resolve() != final_path \
            or tail.get("artifact_sha256") != sha256_file(final_path):
        raise ValueError("CORAL_AUTHORITY_NO_GO: curve ledger/final binding")
    return {
        "final": final, "p0": p0, "resolved": resolved,
        "final_sha256": sha256_file(final_path), "p0_sha256": sha256_file(p0_path),
        "resolved_sha256": sha256_file(resolved_path), "ledger_sha256": sha256_file(ledger_path),
        "ledger_tail": tail["record_sha256"],
    }

#!/usr/bin/env python3
"""Fail-closed outcome-blind adjudicator for the single mechanism router."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import yaml

SPEC = Path(__file__).resolve().parents[1] / "docs/single_mechanism_extension_router_spec_20260819.yaml"
def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
def _fact_state(facts, key):
    value = facts.get(key)
    if value is True: return "true"
    if value is False: return "false"
    if value is None or value in {"pending", "missing", "invalid"}: return "pending"
    raise ValueError(f"fact_must_be_boolean_or_pending:{key}:{value!r}")
def candidate_status(rule, facts):
    required = {key: _fact_state(facts, key) for key in rule.get("require", [])}
    contraindicated = {key: _fact_state(facts, key) for key in rule.get("contraindicate", [])}
    if any(value == "true" for value in contraindicated.values()): status, reason = "false", "contraindication_present"
    elif any(value == "false" for value in required.values()): status, reason = "false", "required_fact_false"
    elif any(value == "pending" for value in required.values()) or any(value == "pending" for value in contraindicated.values()):
        status, reason = "pending", "required_or_contraindication_fact_pending"
    else: status, reason = "true", "all_required_true_and_all_contraindications_false"
    return {"status": status, "reason": reason, "required": required, "contraindicated": contraindicated}
def adjudicate(manifest, spec):
    if manifest.get("candidate_extension_outcome_accessed") is not False: raise ValueError("candidate_extension_outcome_access_must_be_false")
    if any(key in manifest for key in spec["forbidden_router_inputs"]): raise ValueError("forbidden_candidate_outcome_input")
    if not manifest.get("router_manifest_id") or not manifest.get("evidence_manifest_sha256"): raise ValueError("frozen_manifest_identity_and_hash_required")
    facts = manifest.get("facts")
    if not isinstance(facts, dict): raise ValueError("facts_mapping_required")
    results = {name: candidate_status(spec["eligibility"][name], facts) for name in spec["candidates"]}
    eligible = [name for name, row in results.items() if row["status"] == "true"]
    pending = [name for name, row in results.items() if row["status"] == "pending"]
    if len(eligible) == 1: decision, selected = "SELECT_ONE", eligible[0]
    elif len(eligible) > 1: decision, selected = "AMBIGUOUS_NO_EXTENSION", None
    elif pending: decision, selected = "PENDING_NO_EXTENSION", None
    else: decision, selected = "NO_MECHANISM_EXTENSION", None
    frozen = {"router_schema": spec["schema_version"], "router_manifest_id": manifest["router_manifest_id"],
      "evidence_manifest_sha256": manifest["evidence_manifest_sha256"], "facts": facts,
      "candidate_extension_outcome_accessed": False}
    return {"decision": decision, "selected": selected, "eligible": eligible, "pending": pending,
      "candidate_status": results, "decision_input_sha256": _canonical_sha(frozen), "training_authorized": False,
      "step400_authorized": False, "C256_authorized": False}
def self_test(spec):
    keys = {key for rule in spec["eligibility"].values() for field in ("require", "contraindicate") for key in rule.get(field, [])}
    base = {key: False for key in keys}; manifest = {"router_manifest_id": "test", "evidence_manifest_sha256": "0"*64,
      "candidate_extension_outcome_accessed": False, "facts": base}
    assert adjudicate(manifest, spec)["decision"] == "NO_MECHANISM_EXTENSION"
    cwi = dict(base); cwi.update({key: True for key in spec["eligibility"]["CWI"]["require"]}); manifest["facts"] = cwi
    assert adjudicate(manifest, spec)["selected"] == "CWI"
    ambiguous = dict(cwi); ambiguous.update({key: True for key in spec["eligibility"]["PORTABILITY"]["require"]}); manifest["facts"] = ambiguous
    assert adjudicate(manifest, spec)["decision"] == "AMBIGUOUS_NO_EXTENSION"
    pending = dict(base); pending.update({key: True for key in spec["eligibility"]["CWI"]["require"]}); pending[spec["eligibility"]["CWI"]["require"][0]] = None; manifest["facts"] = pending
    assert adjudicate(manifest, spec)["decision"] == "PENDING_NO_EXTENSION"
    manifest["candidate_extension_outcome_accessed"] = True
    try: adjudicate(manifest, spec)
    except ValueError as exc: assert "outcome_access" in str(exc)
    else: raise AssertionError("outcome_access_not_rejected")
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("manifest", nargs="?"); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    spec = yaml.safe_load(SPEC.read_text())
    if args.self_test: self_test(spec); print("single_mechanism_extension_router_self_test=ok"); return
    if not args.manifest: parser.error("manifest required unless --self-test")
    print(json.dumps(adjudicate(json.loads(Path(args.manifest).read_text()), spec), indent=2, ensure_ascii=False, sort_keys=True))
if __name__ == "__main__": main()

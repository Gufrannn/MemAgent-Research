"""Qualification and metrics for the orthogonal closed-loop composition gap."""
from __future__ import annotations

import math
import re
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")
CONTRACT_KEYS = (
    "initial_manifest_hash", "checkpoint_hash", "writer_reader_contract_hash",
    "horizon_contract_hash", "endpoint_definition_hash", "missing_rule_hash",
)


def _unidentified(reason: str) -> dict[str, Any]:
    return {"status": "COMPOSITION_GAP_NOT_IDENTIFIED", "reason": reason,
            "outcomes_read": False, "feedback_claim_authorized": False,
            "actionability_gate": False, "training_authorized": False}


def audit_composition_gap(value: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on transport qualification without changing actionability."""
    if value.get("schema_version") != "closed-loop-composition-gap-v1":
        return _unidentified("schema_version_mismatch")
    required = {
        "gap_direction": "V_splice_minus_V_direct_GC",
        "complete_per_example_bijection": True,
        "splice_algorithm_frozen_before_outcome": True,
        "source_rows_frozen_before_outcome": True,
        "composition_gap_actionability_gate": False,
        "composition_gap_rescues_terminal_IUT": False,
        "feedback_claim_authorized": False,
        "optimizer_steps": 0,
        "new_rollouts": False,
    }
    wrong = {key: (value.get(key), expected) for key, expected in required.items()
             if value.get(key) != expected}
    if wrong:
        return _unidentified(f"qualification_contract_failed:{wrong}")
    direct = value.get("direct_contract")
    splice = value.get("splice_contract")
    if not isinstance(direct, dict) or not isinstance(splice, dict):
        return _unidentified("direct_or_splice_contract_missing")
    for key in CONTRACT_KEYS:
        if not SHA256.fullmatch(str(direct.get(key, ""))) or not SHA256.fullmatch(str(splice.get(key, ""))):
            return _unidentified(f"invalid_contract_hash:{key}")
        if direct[key] != splice[key]:
            return _unidentified(f"contract_mismatch:{key}")
    if not SHA256.fullmatch(str(value.get("splice_algorithm_hash", ""))):
        return _unidentified("splice_algorithm_hash_missing")
    frozen_sources = value.get("prefrozen_source_row_hashes")
    rows = value.get("rows")
    if (not isinstance(frozen_sources, list) or not frozen_sources or
            len(frozen_sources) != len(set(frozen_sources)) or
            any(not SHA256.fullmatch(str(item)) for item in frozen_sources) or
            not isinstance(rows, list) or len(rows) != len(frozen_sources)):
        return _unidentified("prefrozen_source_rows_or_row_count_invalid")
    seen_ids = set(); seen_sources = set(); seen_splice = set()
    for row in rows:
        stable_id = row.get("stable_example_id")
        source_hash = row.get("direct_source_row_hash")
        splice_hash = row.get("splice_row_hash")
        if stable_id is None or str(stable_id) in seen_ids:
            return _unidentified("stable_example_bijection_failure")
        if (not SHA256.fullmatch(str(source_hash or "")) or source_hash in seen_sources or
                not SHA256.fullmatch(str(splice_hash or "")) or splice_hash in seen_splice):
            return _unidentified("source_or_splice_row_bijection_failure")
        seen_ids.add(str(stable_id)); seen_sources.add(source_hash); seen_splice.add(splice_hash)
    if seen_sources != set(frozen_sources):
        return _unidentified("source_rows_do_not_equal_prefrozen_manifest")

    sesoi = value.get("composition_transport_SESOI")
    if not isinstance(sesoi, (int, float)) or not math.isfinite(float(sesoi)) or float(sesoi) < 0:
        return _unidentified("composition_transport_SESOI_invalid")
    gaps = []
    direct_contrasts = []
    splice_contrasts = []
    for row in rows:
        numeric = (row.get("direct_gc_terminal"), row.get("splice_terminal"),
                   row.get("best_control_terminal"))
        if not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in numeric):
            return _unidentified("nonfinite_terminal_value")
        direct_value, splice_value, control_value = map(float, numeric)
        gaps.append(splice_value - direct_value)
        direct_contrasts.append(direct_value - control_value)
        splice_contrasts.append(splice_value - control_value)
    mean_gap = sum(gaps) / len(gaps)
    mae = sum(abs(gap) for gap in gaps) / len(gaps)
    direct_mean = sum(direct_contrasts) / len(direct_contrasts)
    splice_mean = sum(splice_contrasts) / len(splice_contrasts)
    reversal = direct_mean * splice_mean < 0
    large_error = mae >= float(sesoi) and mae > 0
    return {"status": "COMPOSITION_GAP_QUALIFIED", "outcomes_read": True,
            "gap_direction": "V_splice_minus_V_direct_GC",
            "signed_mean_composition_gap": mean_gap,
            "composition_gap_MAE_error_mass": mae,
            "direct_GC_minus_best_control_mean": direct_mean,
            "splice_minus_best_control_mean": splice_mean,
            "direction_reversal": reversal,
            "large_composition_error": large_error,
            "myopic_nontransport": reversal or large_error,
            "zero_mean_does_not_imply_zero_error_mass": True,
            "claim_scope": "single_step_to_closed_loop_composition_transport_only",
            "feedback_claim_authorized": False,
            "actionability_gate": False,
            "training_authorized": False}

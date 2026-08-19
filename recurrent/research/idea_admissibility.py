"""Fail-closed evidence adjudication and append-only experiment ledger."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PENDING = "PENDING_EVIDENCE_NO_SELECTION"
NO_METHOD = "NO_METHOD"
ELIGIBLE = "ELIGIBLE_FOR_2STEP_SMOKE"
NON_ORIGINAL_ARMS = {
    "cerc_native_credit", "ncr_certified_routing", "generic_qa_aux",
    "generic_frozen_judge_tournament", "information_matched_raw_judge",
    "uniform_tie_rescue", "typed_boundary_prompt_control", "target_aligned_repair",
}
NCR_GATES = (
    "shape_a_candidate_free_t0", "exact_linked_same_write_key", "exact_tie_coverage",
    "frozen_readout", "writer_only", "non_tie_bitwise_unchanged", "gradient_safety",
    "beats_qa_only", "beats_generic_qa", "beats_generic_judge",
    "beats_information_matched_raw_judge", "beats_generic_tie_rescue",
    "himpo_non_equivalence", "himpo_like_baseline_matched",
    "memory_r2_non_equivalence", "memory_r2_like_baseline_matched",
    "exact_noop_v2_qualified",
)


def validate_shape_a_contract(contract: dict[str, Any]) -> None:
    expected = {
        "independent_unit": "stable_example_id", "max_independent_n": 128,
        "primary_representation": "paired_tau", "stacked_role": "implementation_consistency_audit_only",
        "count_paired_and_stacked_as_one": True, "select_more_significant_representation": False,
        "b_raw": "tau~P2_raw_T0", "b_struct": "tau~P2_raw_T0+D_star",
        "d_star_dimensions": 1, "outer_grouped_folds": 4, "model_capacity": "low_capacity_linear",
    }
    wrong = {key: (contract.get(key), value) for key, value in expected.items() if contract.get(key) != value}
    if wrong:
        raise ValueError(f"{NO_METHOD}: invalid frozen Shape A contract: {wrong}")
    if contract.get("arms_increase_independent_n") or contract.get("turns_increase_independent_n") \
            or contract.get("seeds_increase_independent_n") or contract.get("tokens_increase_independent_n"):
        raise ValueError(f"{NO_METHOD}: repeated arms/turns/seeds/tokens cannot increase independent n")
    harm_events = int(contract.get("harm_events", 0))
    if harm_events < 20 and (contract.get("multivariable_logistic_primary") or contract.get("auroc_primary")):
        raise ValueError(f"{NO_METHOD}: harm_events={harm_events}<20 forbids multivariable logistic/AUROC primary")
    firewall = contract.get("inference_firewall", {})
    required = {
        "fold_level_t_or_wilcoxon": False, "repeated_cv_is_independent_replication": False,
        "restricted_d_permutation_label": "artifact_sensitivity_not_exact_crt",
        "central_evidence_rule": "A_AND_B_AND_C", "allow_choose_a_b_or_c": False,
        "arm_x_d_in_maxT": False, "algebra_audit_outputs_p_values": False,
        "algebra_audit_authorizes_claim": False,
    }
    wrong = {key: (firewall.get(key), value) for key, value in required.items() if firewall.get(key) != value}
    false_positives = firewall.get("false_positive_counterexamples", [])
    expected_fp = ["pure_difficulty", "p2_redundancy", "role_prevalence", "single_outlier",
                   "heteroskedasticity_only", "fold_accident", "regularization_suppression", "coarse_permutation_artifact"]
    if wrong or false_positives != expected_fp:
        raise ValueError(f"{NO_METHOD}: invalid inference dependence firewall: wrong={wrong}, counterexamples={false_positives}")
    claim = str(contract.get("central_claim", ""))
    fixed_claim = ("preregistered relational compression of the same audit transcript predicts commit-vs-discard "
                   "effect beyond frozen direction-blind marginal summaries and matched pairing shams.")
    forbidden = ("same-information adds information", "structural information gain", "I(tau;D|R,M)>0")
    if claim != fixed_claim or any(term.lower() in claim.lower() for term in forbidden):
        raise ValueError(f"{NO_METHOD}: forbidden information-gain claim or non-frozen central wording")
    provenance = contract.get("d_input_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{NO_METHOD}: UNKNOWN_PROVENANCE_NO_GO: D_input_provenance is required before outcomes")
    required_prov = ("cells", "roles", "targets", "masks", "normalization", "baseline_retained_fields",
                     "baseline_discarded_fields", "deterministically_reconstructable", "d_access_tier",
                     "baseline_access_tier", "query_budget", "token_budget", "gpu_budget")
    missing = [key for key in required_prov if key not in provenance]
    if missing: raise ValueError(f"{NO_METHOD}: UNKNOWN_PROVENANCE_NO_GO: missing {missing}")
    if provenance["d_access_tier"] != provenance["baseline_access_tier"]:
        raise ValueError(f"{NO_METHOD}: A_ACCESS_MISMATCH_NO_GO")
    baseline_full = bool(provenance.get("baseline_contains_full_transcript_metadata"))
    deterministic = bool(provenance["deterministically_reconstructable"])
    expected_semantic = "F_DETERMINISTIC_INDUCTIVE_BIAS" if baseline_full and deterministic else "M_RELATIONAL_COMPRESSION"
    if provenance.get("semantic_class") != expected_semantic:
        raise ValueError(f"{NO_METHOD}: semantic_class must be {expected_semantic}")
    audit = contract.get("outcome_free_representation_audit", {})
    for key in ("residual_variance_ratio", "condition_number", "vif", "role_overlap"):
        if key not in audit: raise ValueError(f"{NO_METHOD}: missing outcome-free representation audit {key}")
    if audit.get("interpretation") != "model_class_representation_audit_not_mutual_information":
        raise ValueError(f"{NO_METHOD}: outcome-free diagnostics are not mutual information")
    sensitivity = contract.get("cpu_sensitivity", {})
    allowed = {"available", "not_available_existing_outputs_insufficient"}
    if sensitivity.get("full_transcript_frozen_random_projection") not in allowed or sensitivity.get("matched_pairing_sham") not in allowed:
        raise ValueError(f"{NO_METHOD}: invalid CPU sensitivity status")
    if sensitivity.get("expand_rollout", True): raise ValueError(f"{NO_METHOD}: CPU sensitivity cannot expand rollout")
    pairing = contract.get("semantic_pairing_null", {})
    pairing_expected = {"k": 2000, "same_response_tensor": True, "same_pipeline": True,
        "exact_crt_p_value": False, "second_primary": False, "c_layer_falsification_only": True,
        "learned_pairing_or_gnn_rescue_on_same_b128": False}
    wrong = {key: (pairing.get(key), value) for key, value in pairing_expected.items() if pairing.get(key) != value}
    for key in ("seed", "allowed_edges_hash", "generator_sha", "folds_hash", "manifest_hash"):
        if not pairing.get(key): wrong[key] = (pairing.get(key), "frozen before B128")
    if wrong: raise ValueError(f"{NO_METHOD}: invalid semantic pairing null: {wrong}")
    measurement = contract.get("dstar_measurement", {})
    if measurement.get("status") != "MEASUREMENT_RELIABLE" or not measurement.get("independent_semantic_replicas") \
            or not measurement.get("isomorphic_replica_contract") or measurement.get("deterministic_rerun"):
        raise ValueError(f"{NO_METHOD}: D* measurement not identified/reliable")
    if not measurement.get("audit_hash"):
        raise ValueError(f"{NO_METHOD}: D* measurement audit hash required")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_ledger(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{PENDING}: evidence ledger is required: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{PENDING}: evidence ledger is empty")
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Decision:
    status: str
    training_authorized: bool
    selected_arm: str | None
    ledger_hash: str
    reasons: tuple[str, ...]


def adjudicate(rows: list[dict[str, Any]], ledger_hash: str) -> Decision:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        for required in ("event_id", "timestamp", "candidate", "status"):
            if not row.get(required):
                return Decision(NO_METHOD, False, None, ledger_hash, (f"missing {required}",))
        latest[str(row["candidate"])] = row
    if any(row["status"] == "pending" for row in latest.values()):
        return Decision(PENDING, False, None, ledger_hash, ("pending evidence",))
    eligible = [name for name, row in latest.items() if row["status"] == "eligible"]
    if eligible != ["ncr_certified_routing"]:
        return Decision(NO_METHOD, False, None, ledger_hash, (f"eligible candidates={eligible!r}; expected NCR only",))
    row = latest[eligible[0]]
    try:
        validate_shape_a_contract(row.get("shape_a_contract", {}))
    except ValueError as exc:
        return Decision(NO_METHOD, False, None, ledger_hash, (str(exc),))
    gates = row.get("gates", {})
    failed = [gate for gate in NCR_GATES if gates.get(gate) is not True]
    if failed:
        return Decision(NO_METHOD, False, None, ledger_hash, tuple(f"failed gate: {x}" for x in failed))
    if row.get("shape_a", {}).get("t0_formula") != "P2_raw^T0 vs P2_raw^T0+D_pre^audit":
        return Decision(NO_METHOD, False, None, ledger_hash, ("invalid Shape A T0",))
    if row.get("shape_a", {}).get("t1_leaks_into_t0", True):
        return Decision(NO_METHOD, False, None, ledger_hash, ("T1 candidate point gate leaks into T0",))
    return Decision(ELIGIBLE, True, eligible[0], ledger_hash, ())


def require_arm(arm: str, ledger_path: str | Path | None, *, diagnostic_only: bool = False) -> Decision:
    if arm == "qa_only_original":
        return Decision("ORIGINAL_AUTHORIZED", True, arm, "", ())
    if arm not in NON_ORIGINAL_ARMS:
        raise ValueError(f"unknown IDEA_ARM={arm}")
    if not ledger_path:
        raise ValueError(f"{PENDING}: non-Original arm requires IDEA_EVIDENCE_LEDGER")
    rows, digest = load_ledger(ledger_path)
    decision = adjudicate(rows, digest)
    if arm == "typed_boundary_prompt_control":
        if diagnostic_only:
            return Decision("DIAGNOSTIC_ONLY", False, None, digest, ("MemTX collision: prompt diagnostic only",))
        raise ValueError(f"{NO_METHOD}: typed boundary is not training-authorized")
    if arm in {"cerc_native_credit", "target_aligned_repair"}:
        raise ValueError(f"{NO_METHOD}: {arm} is control/concept code, not an authorized method launcher")
    if arm != "ncr_certified_routing":
        if decision.status != ELIGIBLE:
            raise ValueError(f"{decision.status}: {decision.reasons}")
        return Decision("AUTHORIZED_NCR_BASELINE", True, arm, digest, ())
    if not decision.training_authorized:
        raise ValueError(f"{decision.status}: {decision.reasons}")
    return decision


def append_run_ledger(path: str | Path, record: dict[str, Any]) -> None:
    required = ("run_id", "arm", "lambda", "stratum", "readout_hash", "manifest_hash", "seed_schedule")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"run ledger missing required fields: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    fd = os.open(path, flags, 0o644)
    try:
        os.write(fd, (json.dumps(record, sort_keys=True) + "\n").encode())
    finally:
        os.close(fd)

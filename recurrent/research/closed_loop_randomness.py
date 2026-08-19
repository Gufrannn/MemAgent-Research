"""Closed-loop v6 randomness-estimand and addressable-request audit."""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any

SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUEST_ROLES = {"writer", "certificate", "terminal_reader"}


def address_key(*, experiment: str, mode: str, policy_or_crn: str, example: str,
                replicate: int, turn: int, component: str, request_role: str) -> str:
    return (f"H({experiment}|{mode}|{policy_or_crn}|{example}|{replicate}|"
            f"{turn}|{component}|{request_role})")


def _invalid(reason: str, *, status: str = "RANDOMNESS_ESTIMAND_INVALID") -> dict[str, Any]:
    return {"status": status, "reason": reason,
            "policy_mean_identified": False, "seed_or_replicate_increases_n": False,
            "training_authorized": False}


def _invalid_s(reason: str) -> dict[str, Any]:
    return _invalid(reason, status="STOCHASTIC_POLICY_MEAN_INVALID")


def _validate_addresses(value: dict[str, Any], replicate_keys: set[tuple[str, str, int]]) -> str | None:
    if value.get("sequential_prng_position_is_trajectory_identity") is not False:
        return "sequential_PRNG_position_identity_forbidden"
    requests = value.get("request_ledger")
    if not isinstance(requests, list) or not requests:
        return "addressable_request_ledger_missing"
    seen = set(); turn_roles_by_replicate = defaultdict(set)
    required = ("experiment", "mode", "policy_or_crn", "example", "replicate", "turn",
                "component", "request_role", "address_key", "address_hash")
    for index, row in enumerate(requests):
        if any(key not in row for key in required): return f"request={index}_field_missing"
        if row["mode"] != "S" or row["request_role"] not in REQUEST_ROLES:
            return f"request={index}_mode_or_role_invalid"
        if not isinstance(row["replicate"], int) or not isinstance(row["turn"], int) or row["turn"] < 1:
            return f"request={index}_replicate_or_turn_invalid"
        key = (str(row["policy_or_crn"]), str(row["example"]), row["replicate"])
        if key not in replicate_keys: return f"request={index}_not_in_prefrozen_replicates"
        expected = address_key(experiment=str(row["experiment"]), mode="S",
                               policy_or_crn=key[0], example=key[1], replicate=key[2],
                               turn=row["turn"], component=str(row["component"]),
                               request_role=str(row["request_role"]))
        if row["address_key"] != expected or row["address_hash"] != hashlib.sha256(expected.encode()).hexdigest():
            return f"request={index}_address_key_or_hash_mismatch"
        if expected in seen: return f"request={index}_duplicate_address"
        seen.add(expected); turn_roles_by_replicate[key].add((row["turn"], row["request_role"]))
    horizon = value.get("horizon")
    if not isinstance(horizon, int) or horizon < 1: return "horizon_invalid"
    expected_turn_roles = {(turn, role) for turn in range(1, horizon + 1)
                           for role in REQUEST_ROLES}
    for key in replicate_keys:
        if turn_roles_by_replicate[key] != expected_turn_roles:
            return f"turn_x_request_role_addressing_incomplete:{key}"
    return None


def audit_randomness_estimand(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != "closed-loop-randomness-estimand-v1":
        return _invalid("schema_version_mismatch")
    if value.get("mode_frozen_before_outcome") is not True:
        return _invalid("primary_mode_not_frozen_before_outcome")
    mode = value.get("primary_mode")
    if mode == "D":
        required = {"estimand": "temperature0_deterministic_protocol_value", "temperature": 0,
                    "deterministic_protocol_frozen": True, "optimizer_steps": 0, "new_rollouts": False}
        wrong = {key: (value.get(key), expected) for key, expected in required.items()
                 if value.get(key) != expected}
        if wrong or not SHA256.fullmatch(str(value.get("deterministic_protocol_hash", ""))):
            return _invalid(f"deterministic_protocol_contract_failed:{wrong}")
        sensitivity = value.get("stochastic_sensitivity_requested")
        if not isinstance(sensitivity, bool): return _invalid("stochastic_sensitivity_requested_missing")
        status = "DETERMINISTIC_PROTOCOL_PRIMARY"
        if sensitivity:
            direct = value.get("deterministic_gc_minus_best_control")
            stochastic = value.get("stochastic_gc_minus_best_control")
            if not all(isinstance(item, (int, float)) and math.isfinite(float(item))
                       for item in (direct, stochastic)):
                return _invalid("stochastic_sensitivity_contrasts_invalid")
            if float(direct) * float(stochastic) < 0:
                status = "STOCHASTIC_NONTRANSPORT"
        return {"status": status, "primary_mode": "D",
                "estimand": "temperature0_deterministic_protocol_value",
                "deterministic_terminal_IUT_preserved": True,
                "stochastic_sensitivity_is_actionability_gate": False,
                "seed_or_replicate_increases_n": False, "training_authorized": False}
    if mode == "F":
        required = {"estimand": "single_frozen_seed_realization", "single_seed_screening_only": True,
                    "confirmatory_or_policy_mean_claim_authorized": False,
                    "optimizer_steps": 0, "new_rollouts": False}
        wrong = {key: (value.get(key), expected) for key, expected in required.items()
                 if value.get(key) != expected}
        if wrong or not isinstance(value.get("frozen_seed"), int):
            return _invalid(f"single_seed_contract_failed:{wrong}")
        return {"status": "SINGLE_FROZEN_SEED_REALIZED_SCREENING_ONLY", "primary_mode": "F",
                "estimand": "single_frozen_seed_realization", "actionability_claim_authorized": False,
                "policy_mean_identified": False, "seed_or_replicate_increases_n": False,
                "training_authorized": False}
    if mode != "S": return _invalid("primary_mode_must_be_D_S_or_F")
    required = {"estimand": "seed_marginal_stochastic_policy_value",
                "policy_specific_seed_namespaces": True, "seed_namespaces_independent": True,
                "seed_namespaces_nonoverlapping": True,
                "within_policy_example_replicate_mean_before_example_comparison": True,
                "seed_or_replicate_increases_scientific_n": False,
                "optimizer_steps": 0, "new_rollouts": False}
    wrong = {key: (value.get(key), expected) for key, expected in required.items()
             if value.get(key) != expected}
    if wrong: return _invalid_s(f"stochastic_policy_contract_failed:{wrong}")
    policies = value.get("policies"); examples = value.get("examples"); k = value.get("K")
    namespaces = value.get("policy_seed_namespaces")
    if (not isinstance(policies, list) or not policies or len(policies) != len(set(policies)) or
            not isinstance(examples, list) or not examples or len(examples) != len(set(examples)) or
            not isinstance(k, int) or k < 2 or not isinstance(namespaces, dict) or set(namespaces) != set(policies)):
        return _invalid_s("policy_example_K_or_namespace_manifest_invalid")
    seed_sets = []
    for policy in policies:
        seeds = namespaces[policy]
        if not isinstance(seeds, list) or len(seeds) != k or len(seeds) != len(set(seeds)):
            return _invalid_s(f"policy_namespace_invalid:{policy}")
        seed_sets.append(set(seeds))
    if any(seed_sets[i] & seed_sets[j] for i in range(len(seed_sets)) for j in range(i + 1, len(seed_sets))):
        return _invalid_s("policy_seed_namespaces_overlap")
    records = value.get("replicates")
    if not isinstance(records, list): return _invalid_s("replicate_ledger_missing")
    values = defaultdict(list); replicate_keys = set(); seen = set()
    for row in records:
        key = (str(row.get("policy")), str(row.get("stable_example_id")), row.get("replicate"))
        if (key[0] not in policies or key[1] not in {str(item) for item in examples} or
                not isinstance(key[2], int) or not 0 <= key[2] < k or key in seen):
            return _invalid_s("replicate_key_invalid_or_duplicate")
        seed = row.get("seed"); endpoint = row.get("official_endpoint_value")
        if seed != namespaces[key[0]][key[2]] or not isinstance(endpoint, (int, float)) or not math.isfinite(float(endpoint)):
            return _invalid_s("replicate_seed_or_endpoint_invalid")
        seen.add(key); replicate_keys.add(key); values[key[:2]].append(float(endpoint))
    expected = {(str(policy), str(example), replicate) for policy in policies for example in examples
                for replicate in range(k)}
    if seen != expected: return _invalid_s("replicate_ledger_not_complete_policy_x_example_x_K")
    address_error = _validate_addresses(value, replicate_keys)
    if address_error: return _invalid_s(address_error)
    crn = value.get("crn_sensitivity", {"requested": False})
    if not isinstance(crn, dict) or not isinstance(crn.get("requested"), bool):
        return _invalid_s("CRN_sensitivity_declaration_missing")
    if crn["requested"]:
        required_crn = {"corrected_per_trajectory_seeds": True, "bci_status": "PASS_COUPLED",
                        "role": "coupling_sensitivity_only", "primary_estimand": False}
        wrong_crn = {key: (crn.get(key), expected) for key, expected in required_crn.items()
                     if crn.get(key) != expected}
        crn_seeds = crn.get("seed_namespace")
        if (wrong_crn or not isinstance(crn_seeds, list) or not crn_seeds or
                set(crn_seeds) & set().union(*seed_sets)):
            return _invalid_s(f"CRN_sensitivity_contract_failed:{wrong_crn}")
    means = [{"policy": policy, "stable_example_id": str(example),
              "replicate_mean_endpoint": sum(values[(str(policy), str(example))]) / k}
             for policy in policies for example in examples]
    return {"status": "SEED_MARGINAL_STOCHASTIC_POLICY_VALUE_QUALIFIED", "primary_mode": "S",
            "estimand": "seed_marginal_stochastic_policy_value", "K": k,
            "policy_example_replicate_means": means,
            "independent_unit": "stable_example_id", "seed_or_replicate_increases_n": False,
            "addressable_randomness_key": "H(experiment,mode,policy_or_CRN,example,replicate,turn,component,request_role)",
            "sequential_PRNG_position_identity_authorized": False,
            "training_authorized": False}

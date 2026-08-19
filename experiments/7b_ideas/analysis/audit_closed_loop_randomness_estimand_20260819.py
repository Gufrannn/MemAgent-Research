#!/usr/bin/env python3
"""CLI/self-test for closed-loop v6 randomness-estimand qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from recurrent.research.closed_loop_randomness import address_key, audit_randomness_estimand  # noqa: E402


def _s_manifest():
    policies = ["GC", "GF"]; examples = ["e0", "e1"]; k = 2
    namespaces = {"GC": [10, 11], "GF": [20, 21]}
    replicates = []; requests = []
    for policy in policies:
        for example in examples:
            for replicate in range(k):
                replicates.append({"policy": policy, "stable_example_id": example, "replicate": replicate,
                                   "seed": namespaces[policy][replicate],
                                   "official_endpoint_value": float(replicate)})
                for turn in (1, 2):
                    for role in ("writer", "certificate", "terminal_reader"):
                        key = address_key(experiment="audit16", mode="S", policy_or_crn=policy,
                                          example=example, replicate=replicate, turn=turn,
                                          component=role, request_role=role)
                        requests.append({"experiment": "audit16", "mode": "S", "policy_or_crn": policy,
                          "example": example, "replicate": replicate, "turn": turn, "component": role,
                          "request_role": role, "address_key": key,
                          "address_hash": hashlib.sha256(key.encode()).hexdigest()})
    return {"schema_version": "closed-loop-randomness-estimand-v1", "primary_mode": "S",
      "mode_frozen_before_outcome": True, "estimand": "seed_marginal_stochastic_policy_value",
      "policy_specific_seed_namespaces": True, "seed_namespaces_independent": True,
      "seed_namespaces_nonoverlapping": True,
      "within_policy_example_replicate_mean_before_example_comparison": True,
      "seed_or_replicate_increases_scientific_n": False, "optimizer_steps": 0, "new_rollouts": False,
      "policies": policies, "examples": examples, "K": k, "policy_seed_namespaces": namespaces,
      "replicates": replicates, "horizon": 2, "sequential_prng_position_is_trajectory_identity": False,
      "request_ledger": requests, "crn_sensitivity": {"requested": False}}


def self_test():
    qualified = audit_randomness_estimand(_s_manifest())
    assert qualified["status"] == "SEED_MARGINAL_STOCHASTIC_POLICY_VALUE_QUALIFIED"
    bad = _s_manifest(); bad["policy_seed_namespaces"]["GF"][0] = 10
    assert audit_randomness_estimand(bad)["status"] == "STOCHASTIC_POLICY_MEAN_INVALID"
    bad = _s_manifest(); bad["request_ledger"][0]["address_key"] = "sequential-position-0"
    assert audit_randomness_estimand(bad)["status"] == "STOCHASTIC_POLICY_MEAN_INVALID"
    bad = _s_manifest(); bad["request_ledger"] = [row for row in bad["request_ledger"]
      if not (row["policy_or_crn"] == "GC" and row["example"] == "e0" and
              row["replicate"] == 0 and row["turn"] == 2 and row["request_role"] == "terminal_reader")]
    assert audit_randomness_estimand(bad)["status"] == "STOCHASTIC_POLICY_MEAN_INVALID"
    deterministic = {"schema_version": "closed-loop-randomness-estimand-v1", "primary_mode": "D",
      "mode_frozen_before_outcome": True, "estimand": "temperature0_deterministic_protocol_value",
      "temperature": 0, "deterministic_protocol_frozen": True, "deterministic_protocol_hash": "a" * 64,
      "stochastic_sensitivity_requested": True, "deterministic_gc_minus_best_control": .2,
      "stochastic_gc_minus_best_control": -.1, "optimizer_steps": 0, "new_rollouts": False}
    assert audit_randomness_estimand(deterministic)["status"] == "STOCHASTIC_NONTRANSPORT"
    frozen = {"schema_version": "closed-loop-randomness-estimand-v1", "primary_mode": "F",
      "mode_frozen_before_outcome": True, "estimand": "single_frozen_seed_realization",
      "single_seed_screening_only": True, "confirmatory_or_policy_mean_claim_authorized": False,
      "frozen_seed": 7, "optimizer_steps": 0, "new_rollouts": False}
    assert audit_randomness_estimand(frozen)["status"] == "SINGLE_FROZEN_SEED_REALIZED_SCREENING_ONLY"
    print("closed_loop_randomness_estimand_v6_self_test=ok")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); return
    if not args.manifest: parser.error("--manifest required")
    print(json.dumps(audit_randomness_estimand(json.loads(args.manifest.read_text())), indent=2, sort_keys=True))


if __name__ == "__main__": main()

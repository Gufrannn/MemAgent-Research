"""Deterministic outcome-blind semantic pairing null manifest and SPE decision."""

from __future__ import annotations
import hashlib, json, random

K = 2000
PRESERVED = ("response_values", "missingness", "pair_count", "direction_count", "max_mad_nonlinearity",
             "one_dimensional_output", "access_tier", "query_budget", "token_budget", "gpu_budget")


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate(*, roles: list[str], targets: list[str], allowed_edges: list[tuple[str, str]], seed: int,
             folds_hash: str, generator_sha: str, k: int = K) -> dict:
    if k != K: raise ValueError("semantic pairing null requires frozen K=2000")
    identity = set(zip(roles, targets)); allowed = sorted(set(map(tuple, allowed_edges)) - identity)
    if not allowed: raise ValueError("no valid non-identity allowed edges")
    if len(generator_sha) != 64 or len(folds_hash) != 64: raise ValueError("generator SHA and folds hash must be SHA-256")
    rng = random.Random(seed); mappings = []
    for index in range(k):
        edge = allowed[rng.randrange(len(allowed))]
        mappings.append({"index": index, "role": edge[0], "target": edge[1], "mapping_hash": _hash([seed, index, edge])})
    config = {"k": k, "seed": seed, "allowed_edges": allowed, "generator_sha": generator_sha,
              "folds_hash": folds_hash, "preserved": PRESERVED, "outcome_conditioned": False,
              "identity_allowed": False, "invalid_allowed": False, "cross_permission_allowed": False,
              "same_pipeline_as_real_d": True}
    return {"config": config, "config_hash": _hash(config), "mappings": mappings}


def validate(manifest: dict, *, expected_hash: str) -> None:
    config = manifest.get("config", {})
    if manifest.get("config_hash") != expected_hash or _hash(config) != expected_hash:
        raise ValueError("semantic pairing manifest/hash mismatch")
    if config.get("k") != K or tuple(config.get("preserved", ())) != PRESERVED:
        raise ValueError("semantic pairing preservation contract mismatch")
    if any(config.get(key) for key in ("outcome_conditioned", "identity_allowed", "invalid_allowed", "cross_permission_allowed")):
        raise ValueError("forbidden semantic pairing mapping")
    if not config.get("same_pipeline_as_real_d") or len(manifest.get("mappings", [])) != K:
        raise ValueError("all shams must use the real-D pipeline")


def spe_decision(*, delta_real: float, delta_sham: list[float], sesoi: float,
                 leave_role_stable: bool, checkpoint_stable: bool) -> dict:
    if len(delta_sham) != K: raise ValueError("SPE requires K=2000 sham deltas")
    q95 = sorted(float(x) for x in delta_sham)[int(.95 * (K - 1))]
    spe = float(delta_real) - q95
    semantic = delta_real > sesoi and spe > 0 and leave_role_stable and checkpoint_stable
    generic_only = delta_real > 0 and spe <= 0
    return {"SPE": spe, "q95_sham": q95,
            "classification": "obligation_semantic_relational_compression" if semantic else
              "generic_relational_compression_obligation_shapeA_NO_GO" if generic_only else "NO_GO",
            "c_layer_falsification_only": True, "exact_crt_p_value": False, "second_primary": False,
            "model_rescue_on_same_B128_allowed": False}

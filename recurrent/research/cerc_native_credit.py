"""CERC native-credit exposure control; never creates targets or pair rewards."""

from __future__ import annotations
from collections import defaultdict


def validate_native_credit(rows: list[dict]) -> dict:
    by_prompt_variant: dict[tuple[str, str], list[float]] = defaultdict(list)
    variants_by_prompt: dict[str, set[str]] = defaultdict(set)
    prompt_rewards: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["uid"]), str(row["variant_id"]))
        by_prompt_variant[key].append(float(row["qa_reward"]))
        variants_by_prompt[key[0]].add(key[1])
        prompt_rewards[key[0]].append(float(row["qa_reward"]))
    native = {key: max(vals) != min(vals) for key, vals in by_prompt_variant.items()}
    illusion = any(max(prompt_rewards[u]) != min(prompt_rewards[u]) and not any(native[(u, v)] for v in variants)
                   for u, variants in variants_by_prompt.items())
    if illusion:
        raise ValueError("MIXED_GROUP_ILLUSION: cross-variant differences with within-variant ties")
    if not any(native.values()):
        raise ValueError("NO_METHOD: no same-prompt/same-variant native QA variance")
    return {"control_only": True, "native_variant_groups": sum(native.values()), "adds_reward": False}

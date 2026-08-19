"""Prompt-only five-arm diagnostic. It never emits reward or training labels."""

from __future__ import annotations
import hashlib
import random

ARMS = ("original_free_text", "boundary_only", "random_label", "equal_length_style", "typed_known_open")


def build_prompts(text: str, *, token_budget: int, seed: int) -> dict[str, dict]:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    rng = random.Random(seed)
    labels = ["[KNOWN]", "[OPEN]"]
    rng.shuffle(labels)
    variants = {
        "original_free_text": text,
        "boundary_only": f"[BOUNDARY] {text}",
        "random_label": f"{labels[0]} {text}",
        "equal_length_style": f"[STYLE___] {text}",
        "typed_known_open": f"[KNOWN] {text} [OPEN]",
    }
    return {name: {"prompt": value, "token_budget": token_budget, "label_permutation": labels,
                   "permutation_hash": hashlib.sha256(f"{seed}:{labels}".encode()).hexdigest(),
                   "diagnostic_only": True} for name, value in variants.items()}

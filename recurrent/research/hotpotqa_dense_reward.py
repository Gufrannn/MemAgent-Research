"""Dense HotpotQA training reward; evaluation remains independent EM/F1."""

from __future__ import annotations

import collections
import re
import string
from typing import Iterable

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_EXPLICIT_ANSWER = re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)\s*(.+)", re.IGNORECASE)


def _last_boxed_answer(text: str) -> str | None:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    cursor, depth = start + len(r"\boxed{"), 1
    for index in range(cursor, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[cursor:index].strip()
    return None


def extract_answer(solution_str: str) -> tuple[str, bool, str]:
    text = str(solution_str or "").strip()
    boxed = _last_boxed_answer(text)
    if boxed is not None:
        return boxed, True, "boxed"
    tail = text[-800:]
    explicit = list(_EXPLICIT_ANSWER.finditer(tail))
    if explicit:
        return explicit[-1].group(1).splitlines()[0].strip(" $`*_"), False, "explicit"
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return (lines[-1] if lines else tail).strip(" $`*_"), False, "last_line"


def normalize_answer(text: str) -> str:
    lowered = str(text or "").lower()
    no_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    return _WHITESPACE.sub(" ", _ARTICLES.sub(" ", no_punctuation)).strip()


def token_f1(prediction: str, reference: str) -> tuple[float, float]:
    pred, ref = normalize_answer(prediction), normalize_answer(reference)
    exact = float(pred == ref)
    pred_tokens, ref_tokens = pred.split(), ref.split()
    if not pred_tokens or not ref_tokens:
        return exact, exact
    common = sum((collections.Counter(pred_tokens) & collections.Counter(ref_tokens)).values())
    if common == 0:
        return 0.0, exact
    precision, recall = common / len(pred_tokens), common / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall), exact


def _references(ground_truth: str | Iterable[str]) -> list[str]:
    return [ground_truth] if isinstance(ground_truth, str) else [str(x) for x in ground_truth]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | Iterable[str],
    *,
    f1_weight: float = 0.95,
    grounded_box_bonus: float = 0.05,
    **_: object,
) -> dict[str, float | str]:
    del data_source
    if f1_weight < 0 or grounded_box_bonus < 0 or f1_weight + grounded_box_bonus > 1.0 + 1e-12:
        raise ValueError("reward weights must be non-negative and sum to at most one")
    prediction, is_boxed, route = extract_answer(solution_str)
    best_f1, best_exact = max(
        (token_f1(prediction, ref) for ref in _references(ground_truth)),
        default=(0.0, 0.0),
    )
    bonus = grounded_box_bonus if is_boxed and best_f1 > 0 else 0.0
    return {
        "score": min(1.0, max(0.0, f1_weight * best_f1 + bonus)),
        "dense_token_f1": best_f1,
        "dense_exact_match": best_exact,
        "dense_boxed": float(is_boxed),
        "dense_grounded_box_bonus": bonus,
        "dense_extraction_route": route,
    }

"""Independent fixed-S128 HotpotQA metrics for I/T performance reports.

These metrics are intentionally recomputed from terminal text and parquet
ground truth.  Training reward fields written by the rollout are never trusted
as evaluation measurements.
"""

from __future__ import annotations

import collections
import re
import string
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_EXPLICIT_ANSWER = re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)\s*(.+)", re.IGNORECASE)


def normalize_answer(value: object) -> str:
    lowered = str(value or "").lower()
    unpunctuated = "".join(char for char in lowered if char not in string.punctuation)
    return _WHITESPACE.sub(" ", _ARTICLES.sub(" ", unpunctuated)).strip()


def extract_last_boxed(text: object) -> tuple[str, bool]:
    """Extract the last balanced ``\\boxed{...}`` and report whether it exists.

    A balanced but empty box is still a successful extraction route.  This is
    deliberately distinct from format success: the corrected-project scorer
    stops at that empty prediction instead of falling back to an earlier
    explicit-answer line.
    """
    value = str(text or "")
    start = value.rfind(r"\boxed{")
    if start < 0:
        return "", False
    cursor = start + len(r"\boxed{")
    depth = 1
    for index in range(cursor, len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                answer = value[cursor:index].strip()
                return answer, True
    return "", False


def extract_terminal_answer(text: object) -> tuple[str, str, bool]:
    """Freeze the corrected-project boxed -> explicit -> last-line route."""
    value = str(text or "").strip()
    boxed, boxed_found = extract_last_boxed(value)
    if boxed_found:
        return boxed, "boxed", bool(boxed)
    tail = value[-800:]
    explicit = list(_EXPLICIT_ANSWER.finditer(tail))
    if explicit:
        return explicit[-1].group(1).splitlines()[0].strip(" $`*_"), "explicit", False
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return (lines[-1] if lines else tail).strip(" $`*_"), "last_line", False


def _references(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        return [str(item) for item in value]
    return [str(value)] if value is not None else []


def _one_reference(prediction: str, reference: str) -> dict[str, float]:
    pred = normalize_answer(prediction)
    gold = normalize_answer(reference)
    if not pred or not gold:
        return {
            "exact_match": 0.0, "token_f1": 0.0,
            "precision": 0.0, "recall": 0.0, "sub_exact_match": 0.0,
        }
    exact = float(pred == gold)
    pred_tokens, gold_tokens = pred.split(), gold.split()
    common = sum(
        (collections.Counter(pred_tokens) & collections.Counter(gold_tokens)).values()
    )
    if common == 0:
        precision = recall = f1 = 0.0
    else:
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        f1 = 2.0 * precision * recall / (precision + recall)
    # Historical diagnostic only.  Explicit non-empty guards avoid the Python
    # truth that "" is a substring of every answer.
    sub_em = float(bool(pred and gold) and (pred in gold or gold in pred))
    return {
        "exact_match": exact, "token_f1": f1,
        "precision": precision, "recall": recall,
        "sub_exact_match": sub_em,
    }


def score_terminal_output(output: object, ground_truth: object) -> dict[str, Any]:
    prediction, route, format_success = extract_terminal_answer(output)
    references = _references(ground_truth)
    candidates = [_one_reference(prediction, reference) for reference in references]
    if not candidates:
        candidates = [_one_reference("", "")]
    # Precision/recall decompose the same reference choice that realizes F1;
    # they are not independently maximized across aliases.
    selected = max(
        candidates,
        key=lambda item: (
            item["token_f1"], item["exact_match"],
            item["precision"], item["recall"],
        ),
    )
    return {
        "prediction": prediction,
        "extraction_route": route,
        "format_success": float(format_success),
        "exact_match": max(item["exact_match"] for item in candidates),
        "token_f1": max(item["token_f1"] for item in candidates),
        "precision": selected["precision"],
        "recall": selected["recall"],
        "sub_exact_match": max(item["sub_exact_match"] for item in candidates),
    }


def summarize_hotpot_metrics(
    rows: Sequence[Mapping[str, Any]], *, expected_denominator: int
) -> dict[str, float | int]:
    """Macro metrics over an exact, predeclared denominator."""
    denominator = int(expected_denominator)
    if denominator < 1 or len(rows) != denominator:
        raise ValueError(
            f"metric denominator must be exactly {denominator}, got {len(rows)}"
        )
    required = (
        "exact_match", "token_f1", "precision", "recall",
        "format_success", "sub_exact_match",
    )
    for index, row in enumerate(rows):
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"metric row {index} is missing {missing}")
    return {
        "denominator": denominator,
        "normalized_exact_match": sum(float(row["exact_match"]) for row in rows)
        / denominator,
        "token_f1": sum(float(row["token_f1"]) for row in rows) / denominator,
        "precision": sum(float(row["precision"]) for row in rows) / denominator,
        "recall": sum(float(row["recall"]) for row in rows) / denominator,
        "format_success": sum(float(row["format_success"]) for row in rows)
        / denominator,
        "historical_sub_exact_match_diagnostic": sum(
            float(row["sub_exact_match"]) for row in rows
        ) / denominator,
    }


def summarize_fixed_s128(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if len(rows) != 128:
        raise ValueError(f"fixed-S128 metric denominator must be exactly 128, got {len(rows)}")
    required = ("exact_match", "token_f1", "format_success", "sub_exact_match")
    for index, row in enumerate(rows):
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"metric row {index} is missing {missing}")
    # Preserve the certified S128 aggregate surface exactly. Precision/recall
    # are available on rows but are not retroactively added to old reports.
    return {
        "denominator": 128,
        "normalized_exact_match": sum(float(row["exact_match"]) for row in rows) / 128,
        "token_f1": sum(float(row["token_f1"]) for row in rows) / 128,
        "format_success": sum(float(row["format_success"]) for row in rows) / 128,
        "historical_sub_exact_match_diagnostic": sum(
            float(row["sub_exact_match"]) for row in rows
        ) / 128,
    }


def paired_descriptive_summary(
    i_rows: Sequence[Mapping[str, Any]], t25_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(i_rows) != 128 or len(t25_rows) != 128:
        raise ValueError("paired S128 contrast requires 128 rows from each interface")
    i_by_key = {str(row["stable_key"]): row for row in i_rows}
    t_by_key = {str(row["stable_key"]): row for row in t25_rows}
    if len(i_by_key) != 128 or set(i_by_key) != set(t_by_key):
        raise ValueError("I/T stable-key inventories are not the same 128 examples")
    metrics = ("exact_match", "token_f1", "format_success", "sub_exact_match")
    result: dict[str, Any] = {
        "denominator": 128,
        "estimand": "T25 minus I on these same curated fixed 128 examples",
        "causal": False,
        "population_inference": False,
    }
    for metric in metrics:
        deltas = [
            float(t_by_key[key][metric]) - float(i_by_key[key][metric])
            for key in sorted(i_by_key)
        ]
        result[metric] = {
            "mean_difference": sum(deltas) / 128,
            "improved": sum(delta > 0 for delta in deltas),
            "unchanged": sum(delta == 0 for delta in deltas),
            "worsened": sum(delta < 0 for delta in deltas),
        }
    return result

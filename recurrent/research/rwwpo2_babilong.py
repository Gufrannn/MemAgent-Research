"""Frozen BABILong adapter and metrics for the RWWPO-2 evaluation program.

The adapter is deliberately independent from the training reward.  It maps the
official ``input/question/target`` rows into the four-column MemAgent validation
schema, freezes stable evaluation identities, and recomputes all metrics from
terminal text.  No benchmark outcome is consumed while selecting the
development or confirmation membership.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from recurrent.research.s128_hotpot_metrics import (
    score_terminal_output,
    summarize_hotpot_metrics,
)
from recurrent.research.stable_eval_identity import (
    canonical_sha256,
    sha256_text,
)


SCHEMA_VERSION = "rwwpo2-babilong-adapter-v1"
SOURCE_DATASET_ID = "RMT-team/babilong"
SOURCE_REVISION = "e3a924b6686759422257925a695cbbb4b2684936"
SOURCE_ROWS_PER_CELL = 100
LENGTHS = ("32k", "128k")
TASK_DEPTH = {"qa1": 1, "qa2": 2, "qa3": 3}
CHUNK_SIZE = 5000
MAX_CHUNKS = {"32k": 8, "128k": 32}
GENERATION_SEED = 602214076
SELECTION_NAMESPACE = "rwwpo2-babilong-pilot-v1"

# The wording is frozen from the public lm-evaluation-harness BABILong task
# definitions, followed by one MemAgent-specific formatting instruction.  The
# target sentence remains unchanged, so official substring accuracy is still
# reconstructible from the raw terminal output.
TASK_DESCRIPTIONS = {
    "qa1": (
        "I will give you context with the facts about positions of different "
        "persons hidden in some random text and a question. You need to answer "
        "the question based only on the information from the facts. If a person "
        "was in different locations, use the latest location to answer the "
        "question. The answer sentence must have the form: The most recent "
        "location of 'person' is 'location'."
    ),
    "qa2": (
        "I will give you context with the facts about locations and actions of "
        "different persons hidden in some random text and a question. You need "
        "to answer the question based only on the information from the facts. "
        "If a person got an item in the first location and travelled to the "
        "second location the item is also in the second location. If a person "
        "dropped an item in the first location and moved to the second location "
        "the item remains in the first location. The answer sentence must have "
        "the form: The 'item' is in 'location'."
    ),
    "qa3": (
        "I give you context with the facts about locations and actions of "
        "different persons hidden in some random text and a question. You need "
        "to answer the question based only on the information from the facts. "
        "If a person got an item in the first location and travelled to the "
        "second location the item is also in the second location. If a person "
        "dropped an item in the first location and moved to the second location "
        "the item remains in the first location. The answer sentence must have "
        "the form: Before the 'location_1' the 'item' was in the 'location_2'."
    ),
}

# These indices are the first eight of a SHA-256 rank over [namespace, seed,
# length, task, source_index].  They were fixed before any BABILong model output
# was generated.  Every other source index belongs to the sealed confirmation
# complement.
DEVELOPMENT_SOURCE_INDICES = {
    "32k": {
        "qa1": (54, 40, 16, 89, 9, 90, 7, 36),
        "qa2": (21, 27, 69, 58, 96, 90, 14, 4),
        "qa3": (20, 70, 55, 36, 6, 57, 99, 80),
    },
    "128k": {
        "qa1": (19, 83, 43, 21, 20, 82, 24, 15),
        "qa2": (25, 81, 27, 21, 93, 95, 32, 5),
        "qa3": (97, 24, 72, 98, 9, 1, 0, 22),
    },
}

_CONTROL = re.compile(r"[\x00-\x1f]")


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"BABILong {field} must be non-empty text")
    return value.strip()


def selection_rank(length: str, task: str, source_index: int) -> str:
    """Return the outcome-blind membership rank used by the frozen split."""
    return hashlib.sha256(json.dumps(
        [SELECTION_NAMESPACE, 2026, length, task, int(source_index)],
        separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()


def expected_development_indices(length: str, task: str) -> tuple[int, ...]:
    if length not in LENGTHS or task not in TASK_DEPTH:
        raise ValueError(f"unsupported BABILong cell: {length}/{task}")
    ranked = sorted(
        range(SOURCE_ROWS_PER_CELL),
        key=lambda index: selection_rank(length, task, index),
    )
    return tuple(ranked[:8])


def validate_frozen_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed if the checked-in preregistration drifts from executable constants."""
    expected = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": SOURCE_DATASET_ID,
        "dataset_revision": SOURCE_REVISION,
        "source_rows_per_cell": SOURCE_ROWS_PER_CELL,
        "lengths": list(LENGTHS),
        "task_depth": TASK_DEPTH,
        "chunk_size": CHUNK_SIZE,
        "max_chunks": MAX_CHUNKS,
        "generation_seed": GENERATION_SEED,
        "selection_namespace": SELECTION_NAMESPACE,
        "development_source_indices": {
            length: {task: list(indices) for task, indices in tasks.items()}
            for length, tasks in DEVELOPMENT_SOURCE_INDICES.items()
        },
        "primary_metric": "official_case_insensitive_target_substring_accuracy",
        "key_secondary": "strict_normalized_exact_match",
        "program": "RWWPO-2",
        "scientific_role": "adaptive_development_pilot_then_sealed_confirmation",
        "partition_sizes_per_cell": {"development": 8, "confirmation": 92},
        "development_is_adaptive": True,
        "confirmation_forbidden_until_r400_training_complete": True,
        "secondary_metrics": ["macro_token_f1", "macro_precision", "macro_recall"],
        "safety_metric": "boxed_format_success",
        "decode": {
            "n": 1, "do_sample": False, "temperature": 0.0,
            "top_p": 1.0, "top_k": -1,
        },
        "evaluation_order": [
            "B-R20-development", "D-R20-development",
            "E-R20-development-when-available", "B-D-E-R50-development",
            "B-D-E-R400-sealed-confirmation",
        ],
        "claim_limits": {
            "qa_depth": (
                "registered reasoning-depth proxy, not direct causal "
                "credit-location evidence"
            ),
            "development": "descriptive and adaptive only",
            "r50": "mechanism/pilot evidence, not final performance",
            "r400_confirmation": (
                "formal performance only after the 92-per-cell complement is sealed"
            ),
        },
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"frozen BABILong contract differs at {field}")
    for length in LENGTHS:
        for task in TASK_DEPTH:
            if DEVELOPMENT_SOURCE_INDICES[length][task] != expected_development_indices(
                length, task
            ):
                raise ValueError(f"development selector drift at {length}/{task}")


def partition_indices(length: str, task: str, partition: str) -> tuple[int, ...]:
    development = DEVELOPMENT_SOURCE_INDICES[length][task]
    if partition == "development":
        return development
    if partition == "confirmation":
        excluded = set(development)
        return tuple(index for index in range(SOURCE_ROWS_PER_CELL) if index not in excluded)
    raise ValueError(f"unsupported BABILong partition: {partition}")


def semantic_index(length: str, task: str, source_index: int) -> int:
    length_code = {"32k": 32, "128k": 128}[length]
    return length_code * 10_000 + TASK_DEPTH[task] * 1_000 + int(source_index)


def adapt_source_row(
    source: Mapping[str, Any], *, length: str, task: str, source_index: int,
    partition: str, source_order_index: int,
    context_token_length: Callable[[str], int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map one official row to MemAgent schema plus stable identity evidence."""
    if length not in LENGTHS or task not in TASK_DEPTH:
        raise ValueError(f"unsupported BABILong cell: {length}/{task}")
    if source_index < 0 or source_index >= SOURCE_ROWS_PER_CELL:
        raise ValueError("BABILong source index is outside the frozen 0..99 domain")
    context = _require_text(source.get("input"), "input")
    question = _require_text(source.get("question"), "question")
    target = _require_text(source.get("target"), "target")
    context_tokens = int(context_token_length(context))
    if context_tokens < 1:
        raise ValueError("BABILong context token count must be positive")
    capacity = CHUNK_SIZE * MAX_CHUNKS[length]
    if context_tokens > capacity:
        raise ValueError(
            f"BABILong {length}/{task}/{source_index} has {context_tokens} Qwen tokens "
            f"but frozen capacity is {capacity}; truncation is forbidden"
        )
    prompt_text = (
        TASK_DESCRIPTIONS[task]
        + "\nReturn exactly one final line as \\boxed{<answer sentence>}. "
        + "Do not put explanations inside or after the box.\n\nQuestion: "
        + question
    )
    stable_source = {
        "dataset_id": SOURCE_DATASET_ID,
        "dataset_revision": SOURCE_REVISION,
        "length": length,
        "task": task,
        "depth": TASK_DEPTH[task],
        "source_index": int(source_index),
        "input": context,
        "question": question,
        "target": target,
    }
    source_identity = canonical_sha256(stable_source)
    index = semantic_index(length, task, source_index)
    row = {
        "prompt": [{"role": "user", "content": prompt_text}],
        "context": context,
        "reward_model": {"ground_truth": [target]},
        "extra_info": {
            "index": index,
            "babilong_length": length,
            "babilong_task": task,
            "babilong_depth": TASK_DEPTH[task],
            "babilong_source_index": int(source_index),
            "babilong_partition": partition,
            "babilong_source_identity": source_identity,
        },
    }
    identity = {
        "example_id": str(index),
        "semantic_dataset_index": index,
        "source_order_index": int(source_order_index),
        "raw_row_position": int(source_order_index),
        "production_effective_position": int(source_order_index),
        "context_token_count": context_tokens,
        "source_question_hash": sha256_text(prompt_text),
        "source_context_hash": sha256_text(context),
        "ground_truth_hash": canonical_sha256([target]),
        "babilong_length": length,
        "babilong_task": task,
        "babilong_depth": TASK_DEPTH[task],
        "babilong_source_index": int(source_index),
        "babilong_source_identity": source_identity,
    }
    return row, identity


def adapt_partition(
    sources: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]], *,
    partition: str, context_token_length: Callable[[str], int],
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Adapt both lengths while preserving depth-major, selector-rank order."""
    result = {}
    for length in LENGTHS:
        rows: list[dict[str, Any]] = []
        identities: list[dict[str, Any]] = []
        for task in TASK_DEPTH:
            source_rows = list(sources.get((length, task), ()))
            if len(source_rows) != SOURCE_ROWS_PER_CELL:
                raise ValueError(
                    f"BABILong {length}/{task} must contain exactly "
                    f"{SOURCE_ROWS_PER_CELL} rows, got {len(source_rows)}"
                )
            for source_index in partition_indices(length, task, partition):
                row, identity = adapt_source_row(
                    source_rows[source_index], length=length, task=task,
                    source_index=source_index, partition=partition,
                    source_order_index=len(rows),
                    context_token_length=context_token_length,
                )
                rows.append(row)
                identities.append(identity)
        result[length] = (rows, identities)
    return result


def official_substring_accuracy(output: object, target: object) -> float:
    """Reproduce lm-eval BABILong's case-insensitive target substring metric."""
    prediction = _CONTROL.sub("\n", str(output or "")).strip()
    gold = _require_text(target, "metric target")
    return float(gold.lower() in prediction.lower())


def score_babilong_output(output: object, target: str) -> dict[str, Any]:
    row = score_terminal_output(output, [target])
    return {**row, "official_accuracy": official_substring_accuracy(output, target)}


def summarize_babilong_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("BABILong metric rows must be non-empty")
    base = summarize_hotpot_metrics(rows, expected_denominator=len(rows))
    base["official_accuracy"] = sum(float(row["official_accuracy"]) for row in rows) / len(rows)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        length = str(row["babilong_length"])
        task = str(row["babilong_task"])
        depth = int(row["babilong_depth"])
        if length not in LENGTHS or TASK_DEPTH.get(task) != depth:
            raise ValueError("BABILong metric row has invalid cell identity")
        grouped[f"{length}/{task}/depth{depth}"].append(row)
    by_cell = {}
    for cell, cell_rows in sorted(grouped.items()):
        aggregate = summarize_hotpot_metrics(cell_rows, expected_denominator=len(cell_rows))
        aggregate["official_accuracy"] = sum(
            float(row["official_accuracy"]) for row in cell_rows
        ) / len(cell_rows)
        by_cell[cell] = aggregate
    return {"overall": base, "by_cell": by_cell}


def paired_descriptive_difference(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], *,
    left_name: str, right_name: str,
) -> dict[str, Any]:
    """Pair two interfaces by source identity; never infer a population effect."""
    left_by = {str(row["babilong_source_identity"]): row for row in left}
    right_by = {str(row["babilong_source_identity"]): row for row in right}
    if len(left_by) != len(left) or set(left_by) != set(right_by):
        raise ValueError("BABILong paired interfaces do not share one exact source inventory")
    metrics = (
        "official_accuracy", "token_f1", "exact_match", "format_success",
    )
    result = {
        "estimand": f"{left_name} minus {right_name} on the same adaptive development rows",
        "denominator": len(left_by),
        "causal": False,
        "population_inference": False,
    }
    for metric in metrics:
        differences = [
            float(left_by[key][metric]) - float(right_by[key][metric])
            for key in sorted(left_by)
        ]
        result[metric] = {
            "mean_difference": sum(differences) / len(differences),
            "improved": sum(value > 0 for value in differences),
            "unchanged": sum(value == 0 for value in differences),
            "worsened": sum(value < 0 for value in differences),
        }
    return result

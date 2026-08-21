"""Auditable paired-effect candidate construction and writer-credit routing.

The input to this module is a *validated* same-materialized-candidate
COMMIT/RETAIN capture.  No score is accepted from a JSON field.  Features are
recomputed solely from pre-branch state/candidate evidence, paired outcomes
are recomputed separately, and every held-out score comes from a stable-example
grouped cross-fit model which excludes that example.

This module does not select a method, attach to the trainer, or authorize GPU
training.  The four-pair GPU4-5 capture is permanently a pipeline pilot; only
the exact disjoint capture32 preregistration can enter scientific readiness.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from recurrent.research.commit_retain_capture import (
    SHARED_ARM,
    STABLE_FIELDS,
    _writer_generation,
    build_state_blob,
    canonical_json,
    canonical_sha256,
    require_sha256,
    stable_capture_ids,
    validate_pair_record,
)
from recurrent.research.stable_eval_identity import (
    MANIFEST_ROW_FIELDS,
    evaluation_trajectory_seed,
    validate_resolved_manifest,
)
from recurrent.research.serialization_credit_pilots import center_truncate_token_ids
from recurrent.research.trajectory_seeding import derive_turn_request_seeds


CANDIDATE_ID = "paired_effect_credit"
BUNDLE_SCHEMA = "memagent.paired-effect.crossfit-bundle.v1"
PREBRANCH_SCHEMA = "memagent.paired-effect.prebranch-candidate.v1"
CAPTURE32_PREREG_SCHEMA = "memagent.paired-effect.capture32-preregistration.v1"
TRAINER_INTEGRATION_AUTHORIZED = False
TARGET_NAME = "token_f1_commit_minus_retain"
FOLD_RULE = "sorted_stable_example_id_round_robin_v1"
CAPTURE4_PILOT_POSITIONS = (15, 47, 79, 111)
CAPTURE32_SELECTED_POSITIONS = tuple(range(1, 128, 4))
CAPTURE32_COUNT = 32
CAPTURE32_FOLD_COUNT = 4
CAPTURE32_FOLD_SIZE = 8
CAPTURE32_CHUNK_SIZE = 5000
S128_AUTHORITY_SCHEMA = "memagent.paired-effect.s128-authority.v1"
S128_AUTHORITY_REL = "manifests/h20/qwen25_7b_paired_effect_s128_authority.json"
EXPECTED_S128_AUTHORITY_FILE_SHA256 = (
    "8c7c34cf884972325f5f42c0541f2f8b12ff2c60eb81aeb71b5b59555c899396"
)
EXPECTED_S128_AUTHORITY_SHA256 = (
    "3f612d3717a3934bb7d5c9db054b43b175369d6c76e7af796c73a55132cd5c83"
)
EXPECTED_S128_MODEL_FILE_MANIFEST_SHA256 = (
    "0b5381a2d40dfcad3d72be1f9cfc335433c6b7e3012042f1b1cc768447139fc7"
)
EXPECTED_CAPTURE32_FULL_RANKING_SHA256 = (
    "a971ab01d123b898921f332890c2e26ce939282946f1669a71c54dd168e3e78b"
)
EXPECTED_CAPTURE32_SELECTED_INVENTORY_SHA256 = (
    "0434147f4b6d7878b31662d70b7fce0ee263b989e2106028d7f6ac8b3bb97d87"
)
FEATURE_SCHEMA = (
    "log1p_old_state_tokens",
    "log1p_candidate_state_tokens",
    "candidate_minus_old_length_ratio",
    "candidate_old_token_jaccard",
    "old_tokens_retained_fraction",
    "candidate_tokens_from_old_fraction",
    "candidate_unique_token_fraction",
    "log1p_candidate_prompt_tokens",
    "log1p_candidate_chunk_tokens",
    "intervention_horizon_fraction",
    "candidate_relation_marker_density",
    "candidate_negation_marker_density",
    "candidate_source_marker_density",
    "candidate_numeric_token_density",
)

_WORDS = re.compile(r"[A-Za-z0-9_'-]+")
_RELATION_MARKERS = {
    "after", "author", "because", "before", "born", "capital", "caused",
    "during", "founded", "located", "member", "part", "therefore",
}
_NEGATION_MARKERS = {"no", "not", "never", "unknown", "uncertain", "without"}
_SOURCE_MARKERS = {"according", "document", "evidence", "passage", "source"}

_CAPTURE32_ROW_FIELDS = {
    "selection_slot", "stratum_index", "prompt_length_stratum_start",
    "prompt_length_stratum_end", "prompt_length_sorted_position", "crossfit_fold",
    "example_id", "semantic_dataset_index", "source_order_index", "raw_row_position",
    "production_effective_position", "context_token_count", "source_question_hash",
    "source_context_hash", "ground_truth_hash", "writer_turn0_prompt_token_length",
    "writer_turn0_prompt_token_sha256", "trajectory_seed", "writer_turn0_request_seed",
    "intervention_writer_turn", "total_writer_turns", "stable_example_id",
    "stable_root_id", "stable_write_id", "question_token_ids_sha256",
    "context_token_ids_sha256", "row_sha256",
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"PAIRED_EFFECT_NO_GO: {field} must be finite")
    return float(value)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(1.0, float(denominator))


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"PAIRED_EFFECT_NO_GO: {field} must be SHA-256")
    return value


def _exact_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"PAIRED_EFFECT_NO_GO: {field} must be an integer >= {minimum}")
    return value


def _capture32_fold_assignments(stable_ids: Sequence[str]) -> dict[str, int]:
    assignments = stable_fold_assignments(stable_ids, fold_count=CAPTURE32_FOLD_COUNT)
    sizes = [sum(value == fold for value in assignments.values())
             for fold in range(CAPTURE32_FOLD_COUNT)]
    if len(assignments) != CAPTURE32_COUNT or sizes != [CAPTURE32_FOLD_SIZE] * 4:
        raise ValueError(
            f"PAIRED_EFFECT_NO_GO: capture32 fold sizes {sizes} != {[CAPTURE32_FOLD_SIZE] * 4}"
        )
    return assignments


def validate_s128_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the complete Stable-I S128 identity payload."""
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "source_manifest", "source_manifest_sha256",
        "eval_manifest_hash", "model_file_manifest_sha256",
        "identity_payload", "authority_sha256",
    }:
        raise ValueError("PAIRED_EFFECT_NO_GO: S128 authority fields drifted")
    unsigned = {key: child for key, child in value.items() if key != "authority_sha256"}
    if value.get("schema") != S128_AUTHORITY_SCHEMA \
            or value.get("source_manifest") \
            != "manifests/h20/qwen25_7b_stable_i4x2_seed2026.json" \
            or value.get("source_manifest_sha256") \
            != "6ec24e46954c64d8cb5802e0d92779996b623666be940fb64e21c256e487b128" \
            or value.get("eval_manifest_hash") \
            != "351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a" \
            or value.get("model_file_manifest_sha256") \
            != EXPECTED_S128_MODEL_FILE_MANIFEST_SHA256 \
            or value.get("authority_sha256") != EXPECTED_S128_AUTHORITY_SHA256 \
            or value.get("authority_sha256") != canonical_sha256(unsigned):
        raise ValueError("PAIRED_EFFECT_NO_GO: S128 authority binding drifted")
    checked = validate_resolved_manifest({
        "identity_payload": value.get("identity_payload"),
        "eval_manifest_hash": value.get("eval_manifest_hash"),
    })
    if len(checked["identity_payload"]["rows"]) != 128:
        raise ValueError("PAIRED_EFFECT_NO_GO: S128 authority is not exact 128")
    return json.loads(canonical_json(value))


def validate_capture32_authority_binding(
    preregistration: Mapping[str, Any], authority: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind every preregistered identity to the frozen Stable-I S128 payload.

    The preregistration's own hashes are necessary but not sufficient: an
    attacker can otherwise rewrite both a row and its local digests.  This
    check starts from the separately frozen, complete 128-row authority and
    independently recreates the stable IDs used by the ranking and selected
    inventory.
    """
    prereg = validate_capture32_preregistration(preregistration)
    checked_authority = validate_s128_authority(authority)
    if prereg["source"]["eval_manifest_hash"] != checked_authority["eval_manifest_hash"]:
        raise ValueError("PAIRED_EFFECT_NO_GO: S128 authority eval hash differs")
    authority_rows = checked_authority["identity_payload"]["rows"]
    by_source_order = {int(row["source_order_index"]): row for row in authority_rows}
    ranking = prereg["selection"]["full_population_ranking"]
    for position, source_order in enumerate(ranking["source_order_indices"]):
        row = by_source_order.get(int(source_order))
        if row is None or int(row["raw_row_position"]) != int(
            ranking["raw_row_positions"][position]
        ):
            raise ValueError("PAIRED_EFFECT_NO_GO: ranking differs from S128 authority")
        trajectory_seed = evaluation_trajectory_seed(
            base_seed=prereg["source"]["base_seed"],
            eval_manifest_hash=prereg["source"]["eval_manifest_hash"],
            example_id=str(row["example_id"]),
            source_order_index=int(row["source_order_index"]),
            replica_id=0,
        )
        total_turns = (
            int(row["context_token_count"]) + CAPTURE32_CHUNK_SIZE - 1
        ) // CAPTURE32_CHUNK_SIZE
        intervention_turn = max(1, (total_turns - 1) // 2)
        identity = {
            field: prereg["source"]["eval_manifest_hash"]
            if field == "eval_manifest_hash"
            else row[field]
            for field in STABLE_FIELDS
        }
        stable_ids = stable_capture_ids(
            identity,
            trajectory_seed=trajectory_seed,
            writer_turn=intervention_turn,
        )
        if ranking["stable_example_ids"][position] != stable_ids["stable_example_id"]:
            raise ValueError("PAIRED_EFFECT_NO_GO: ranking stable ID differs from S128 authority")

    for selected in prereg["selected_inventory"]:
        authority_row = by_source_order.get(int(selected["source_order_index"]))
        if authority_row is None or any(
            selected[field] != authority_row[field] for field in MANIFEST_ROW_FIELDS
        ):
            raise ValueError("PAIRED_EFFECT_NO_GO: selected row differs from S128 authority")
    return prereg


def recompute_capture32_source_evidence(
    *,
    parquet_rows: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any],
    tokenizer: Any,
    writer_prompt_builder: Any,
    no_memory_text: str = "No previous memory",
    max_context_tokens: int = 40000,
    chunk_size: int = CAPTURE32_CHUNK_SIZE,
    base_seed: int = 2026,
) -> dict[str, Any]:
    """Rebuild the outcome-blind 128-row ranking and exact capture32 rows.

    This is intentionally executed from raw parquet and the frozen tokenizer
    before a completed capture can be accepted.  No stored prompt length,
    prompt hash, stable ID, fold, or selection position is trusted.
    """
    checked_authority = validate_s128_authority(authority)
    stable_rows = checked_authority["identity_payload"]["rows"]
    if len(parquet_rows) != 128 or len(stable_rows) != 128:
        raise ValueError("PAIRED_EFFECT_NO_GO: source replay requires exact S128")
    if chunk_size != CAPTURE32_CHUNK_SIZE or max_context_tokens != 40000 \
            or base_seed != 2026 or no_memory_text != "No previous memory":
        raise ValueError("PAIRED_EFFECT_NO_GO: source replay configuration drifted")
    stable_by_raw = {int(row["raw_row_position"]): row for row in stable_rows}
    no_memory_ids = list(tokenizer.encode(no_memory_text, add_special_tokens=False))
    candidates: list[dict[str, Any]] = []
    for raw_position, source_row in enumerate(parquet_rows):
        stable = stable_by_raw.get(raw_position)
        if stable is None:
            raise ValueError(f"PAIRED_EFFECT_NO_GO: missing authority raw row {raw_position}")
        prompt = source_row.get("prompt")
        if not isinstance(prompt, list) or len(prompt) != 1 \
                or not isinstance(prompt[0], Mapping) \
                or prompt[0].get("role") != "user" \
                or not isinstance(prompt[0].get("content"), str):
            raise ValueError("PAIRED_EFFECT_NO_GO: S128 prompt structure drifted")
        question = str(prompt[0]["content"])
        context = str(source_row.get("context"))
        reward_model = source_row.get("reward_model")
        if not isinstance(reward_model, Mapping) or "ground_truth" not in reward_model:
            raise ValueError("PAIRED_EFFECT_NO_GO: S128 ground truth structure drifted")
        if hashlib.sha256(question.encode("utf-8")).hexdigest() \
                != stable["source_question_hash"] \
                or hashlib.sha256(context.encode("utf-8")).hexdigest() \
                != stable["source_context_hash"] \
                or canonical_sha256(reward_model["ground_truth"]) \
                != stable["ground_truth_hash"]:
            raise ValueError(f"PAIRED_EFFECT_NO_GO: S128 content hash drift at row {raw_position}")
        question_ids = list(tokenizer.encode(question, add_special_tokens=False))
        raw_context_ids = list(tokenizer.encode(context, add_special_tokens=False))
        if len(raw_context_ids) != int(stable["context_token_count"]):
            raise ValueError(f"PAIRED_EFFECT_NO_GO: tokenizer/context drift at row {raw_position}")
        context_ids = center_truncate_token_ids(raw_context_ids, max_context_tokens)
        first_chunk = context_ids[:chunk_size]
        writer_prompt_ids = list(
            writer_prompt_builder(question_ids, no_memory_ids, first_chunk)
        )
        trajectory_seed = evaluation_trajectory_seed(
            base_seed=base_seed,
            eval_manifest_hash=checked_authority["eval_manifest_hash"],
            example_id=str(stable["example_id"]),
            source_order_index=int(stable["source_order_index"]),
            replica_id=0,
        )
        total_turns = (len(context_ids) + chunk_size - 1) // chunk_size
        intervention_turn = max(1, (total_turns - 1) // 2)
        if total_turns < 3 or intervention_turn >= total_turns - 1:
            raise ValueError(f"PAIRED_EFFECT_NO_GO: row {raw_position} lacks prefix/future")
        identity = {
            field: checked_authority["eval_manifest_hash"]
            if field == "eval_manifest_hash"
            else stable[field]
            for field in STABLE_FIELDS
        }
        stable_ids = stable_capture_ids(
            identity,
            trajectory_seed=trajectory_seed,
            writer_turn=intervention_turn,
        )
        candidates.append({
            **dict(stable),
            **stable_ids,
            "writer_turn0_prompt_token_length": len(writer_prompt_ids),
            "writer_turn0_prompt_token_sha256": canonical_sha256(writer_prompt_ids),
            "trajectory_seed": trajectory_seed,
            "writer_turn0_request_seed": derive_turn_request_seeds(
                [trajectory_seed], [0], 0
            )[0],
            "intervention_writer_turn": intervention_turn,
            "total_writer_turns": total_turns,
            "question_token_ids_sha256": canonical_sha256(question_ids),
            "context_token_ids_sha256": canonical_sha256(context_ids),
        })
    ordered = sorted(
        candidates,
        key=lambda row: (
            int(row["writer_turn0_prompt_token_length"]),
            int(row["source_order_index"]),
        ),
    )
    ranking = {
        "source_order_indices": [int(row["source_order_index"]) for row in ordered],
        "raw_row_positions": [int(row["raw_row_position"]) for row in ordered],
        "writer_turn0_prompt_token_lengths": [
            int(row["writer_turn0_prompt_token_length"]) for row in ordered
        ],
        "writer_turn0_prompt_token_sha256": [
            row["writer_turn0_prompt_token_sha256"] for row in ordered
        ],
        "stable_example_ids": [row["stable_example_id"] for row in ordered],
    }
    selected_base = [ordered[position] for position in CAPTURE32_SELECTED_POSITIONS]
    assignments = _capture32_fold_assignments(
        [row["stable_example_id"] for row in selected_base]
    )
    selected: list[dict[str, Any]] = []
    for slot, (position, base) in enumerate(zip(CAPTURE32_SELECTED_POSITIONS, selected_base)):
        row = {
            "selection_slot": slot,
            "stratum_index": slot,
            "prompt_length_stratum_start": slot * 4,
            "prompt_length_stratum_end": slot * 4 + 3,
            "prompt_length_sorted_position": position,
            "crossfit_fold": assignments[base["stable_example_id"]],
            **{field: base[field] for field in _CAPTURE32_ROW_FIELDS if field not in {
                "selection_slot", "stratum_index", "prompt_length_stratum_start",
                "prompt_length_stratum_end", "prompt_length_sorted_position",
                "crossfit_fold", "row_sha256",
            }},
        }
        row["row_sha256"] = canonical_sha256(row)
        selected.append(row)
    return {
        "full_population_ranking": ranking,
        "full_population_ranking_sha256": canonical_sha256(ranking),
        "selected_inventory": selected,
        "selected_inventory_sha256": canonical_sha256(selected),
    }


def validate_capture32_preregistration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the literal, pre-outcome S128 capture32 cohort and folds."""
    if not isinstance(value, Mapping):
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 preregistration must be an object")
    allowed = {
        "schema", "frozen_at", "candidate_id", "source", "selection", "folds",
        "scorer", "admissibility", "attrition", "selected_inventory", "inventory",
        "claim_boundary", "preregistration_sha256",
    }
    if set(value) != allowed:
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 preregistration fields drifted")
    unsigned = {key: child for key, child in value.items()
                if key != "preregistration_sha256"}
    if value.get("preregistration_sha256") != canonical_sha256(unsigned):
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 preregistration digest mismatch")
    if value.get("schema") != CAPTURE32_PREREG_SCHEMA \
            or value.get("candidate_id") != CANDIDATE_ID \
            or value.get("frozen_at") != "2026-08-21T00:00:00+08:00":
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 preregistration identity drifted")

    source = value.get("source")
    expected_source = {
        "dataset_role": "existing_project_fixed_s128",
        "validation_sha256": "54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6",
        "population_count": 128,
        "eval_manifest_hash": "351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a",
        "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "base_model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "model_file_manifest_sha256": EXPECTED_S128_MODEL_FILE_MANIFEST_SHA256,
        "tokenizer_manifest_sha256": "1567e178abe4f245846c6bd59e7e6f3b7e842fde92200ddfc74851559a402023",
        "s128_authority": S128_AUTHORITY_REL,
        "s128_authority_file_sha256": EXPECTED_S128_AUTHORITY_FILE_SHA256,
        "s128_authority_sha256": EXPECTED_S128_AUTHORITY_SHA256,
        "base_seed": 2026,
    }
    if source != expected_source:
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 source S128/model contract drifted")

    selection = value.get("selection")
    selection_keys = {
        "kind", "population_count", "strata_count", "rows_per_stratum",
        "selected_offset_within_stratum_zero_based", "order_keys",
        "allowed_selection_inputs", "forbidden_selection_inputs",
        "selected_sorted_positions", "prior_observed_pilot_positions",
        "prior_observed_stable_example_ids", "prior_observed_inventory_sha256",
        "prior_pilot_excluded", "replacement_forbidden", "full_population_ranking",
        "full_population_ranking_sha256",
    }
    if not isinstance(selection, Mapping) or set(selection) != selection_keys:
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 selection contract drifted")
    if any((
        selection["kind"] != "outcome_blind_writer_turn0_prompt_length_strata_v2",
        selection["population_count"] != 128,
        selection["strata_count"] != CAPTURE32_COUNT,
        selection["rows_per_stratum"] != 4,
        selection["selected_offset_within_stratum_zero_based"] != 1,
        selection["order_keys"] != [
            "writer_turn0_prompt_token_length ASC", "source_order_index ASC"
        ],
        selection["allowed_selection_inputs"] != [
            "writer_turn0_prompt_token_length", "source_order_index"
        ],
        selection["selected_sorted_positions"] != list(CAPTURE32_SELECTED_POSITIONS),
        selection["prior_observed_pilot_positions"] != list(CAPTURE4_PILOT_POSITIONS),
        selection["prior_pilot_excluded"] is not True,
        selection["replacement_forbidden"] is not True,
    )):
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 outcome-blind strata drifted")
    forbidden = set(selection["forbidden_selection_inputs"])
    if not {
        "arm_outcome", "reader_answer", "reward", "token_f1", "exact_match",
        "candidate_output", "actual_cost", "existing_score", "runtime_uuid",
        "pair_id", "ground_truth", "ground_truth_hash",
    }.issubset(forbidden):
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 selection leakage firewall drifted")
    pilot_ids = selection["prior_observed_stable_example_ids"]
    if not isinstance(pilot_ids, list) or len(pilot_ids) != 4 \
            or len(set(pilot_ids)) != 4 or any(
                _sha256(item, "prior pilot stable ID") != item for item in pilot_ids
            ) or selection["prior_observed_inventory_sha256"] != canonical_sha256(pilot_ids):
        raise ValueError("PAIRED_EFFECT_NO_GO: prior capture4 exclusion inventory drifted")

    ranking = selection["full_population_ranking"]
    ranking_fields = {
        "source_order_indices", "raw_row_positions",
        "writer_turn0_prompt_token_lengths", "writer_turn0_prompt_token_sha256",
        "stable_example_ids",
    }
    if not isinstance(ranking, Mapping) or set(ranking) != ranking_fields \
            or any(not isinstance(ranking[field], list) or len(ranking[field]) != 128
                   for field in ranking_fields):
        raise ValueError("PAIRED_EFFECT_NO_GO: full S128 prompt ranking is incomplete")
    if selection["full_population_ranking_sha256"] \
            != EXPECTED_CAPTURE32_FULL_RANKING_SHA256 \
            or selection["full_population_ranking_sha256"] != canonical_sha256(ranking):
        raise ValueError("PAIRED_EFFECT_NO_GO: full S128 prompt ranking digest mismatch")
    source_orders = ranking["source_order_indices"]
    raw_positions = ranking["raw_row_positions"]
    prompt_lengths = ranking["writer_turn0_prompt_token_lengths"]
    ranking_ids = ranking["stable_example_ids"]
    if sorted(source_orders) != list(range(128)) or sorted(raw_positions) != list(range(128)):
        raise ValueError("PAIRED_EFFECT_NO_GO: full S128 ranking lost/duplicated rows")
    if any(_exact_int(item, "ranking prompt length", minimum=1) != item
           for item in prompt_lengths) or any(
               _sha256(item, "ranking prompt hash") != item
               for item in ranking["writer_turn0_prompt_token_sha256"]
           ) or any(_sha256(item, "ranking stable ID") != item for item in ranking_ids):
        raise ValueError("PAIRED_EFFECT_NO_GO: full S128 ranking values are invalid")
    if len(set(ranking_ids)) != 128 or list(zip(prompt_lengths, source_orders)) != sorted(
        zip(prompt_lengths, source_orders)
    ):
        raise ValueError("PAIRED_EFFECT_NO_GO: full S128 ranking order/identity drifted")

    rows = value.get("selected_inventory")
    if not isinstance(rows, list) or len(rows) != CAPTURE32_COUNT:
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 inventory must contain exactly 32 rows")
    stable_examples: list[str] = []
    stable_roots: list[str] = []
    stable_writes: list[str] = []
    for slot, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _CAPTURE32_ROW_FIELDS:
            raise ValueError(f"PAIRED_EFFECT_NO_GO: capture32 row {slot} fields drifted")
        row_unsigned = {key: child for key, child in row.items() if key != "row_sha256"}
        if row["row_sha256"] != canonical_sha256(row_unsigned):
            raise ValueError(f"PAIRED_EFFECT_NO_GO: capture32 row {slot} digest mismatch")
        expected_position = CAPTURE32_SELECTED_POSITIONS[slot]
        integer_expectations = {
            "selection_slot": slot,
            "stratum_index": slot,
            "prompt_length_stratum_start": slot * 4,
            "prompt_length_stratum_end": slot * 4 + 3,
            "prompt_length_sorted_position": expected_position,
        }
        if any(row.get(field) != expected for field, expected in integer_expectations.items()):
            raise ValueError(f"PAIRED_EFFECT_NO_GO: capture32 row {slot} stratum drifted")
        for field in (
            "semantic_dataset_index", "source_order_index", "raw_row_position",
            "production_effective_position", "context_token_count",
            "writer_turn0_prompt_token_length", "trajectory_seed",
            "writer_turn0_request_seed", "intervention_writer_turn",
            "total_writer_turns", "crossfit_fold",
        ):
            _exact_int(row[field], f"capture32 row {slot}.{field}")
        if row["example_id"] != str(row["semantic_dataset_index"]) \
                or row["production_effective_position"] != row["source_order_index"]:
            raise ValueError(f"PAIRED_EFFECT_NO_GO: capture32 row {slot} source identity drifted")
        for field in (
            "source_question_hash", "source_context_hash", "ground_truth_hash",
            "writer_turn0_prompt_token_sha256", "stable_example_id", "stable_root_id",
            "stable_write_id", "question_token_ids_sha256", "context_token_ids_sha256",
            "row_sha256",
        ):
            _sha256(row[field], f"capture32 row {slot}.{field}")
        if row["source_order_index"] != source_orders[expected_position] \
                or row["raw_row_position"] != raw_positions[expected_position] \
                or row["writer_turn0_prompt_token_length"] != prompt_lengths[expected_position] \
                or row["writer_turn0_prompt_token_sha256"] != ranking[
                    "writer_turn0_prompt_token_sha256"
                ][expected_position] \
                or row["stable_example_id"] != ranking_ids[expected_position]:
            raise ValueError(f"PAIRED_EFFECT_NO_GO: capture32 row {slot} differs from S128 ranking")
        identity = {
            field: source["eval_manifest_hash"] if field == "eval_manifest_hash" else row[field]
            for field in STABLE_FIELDS
        }
        expected_ids = stable_capture_ids(
            identity,
            trajectory_seed=row["trajectory_seed"],
            writer_turn=row["intervention_writer_turn"],
        )
        if any(row[field] != expected_ids[field] for field in expected_ids):
            raise ValueError(f"PAIRED_EFFECT_NO_GO: capture32 row {slot} stable IDs do not reproduce")
        expected_request = derive_turn_request_seeds(
            [row["trajectory_seed"]], [0], 0
        )[0]
        expected_trajectory_seed = evaluation_trajectory_seed(
            base_seed=source["base_seed"],
            eval_manifest_hash=source["eval_manifest_hash"],
            example_id=row["example_id"],
            source_order_index=row["source_order_index"],
            replica_id=0,
        )
        expected_total_turns = (
            row["context_token_count"] + CAPTURE32_CHUNK_SIZE - 1
        ) // CAPTURE32_CHUNK_SIZE
        expected_intervention_turn = max(1, (expected_total_turns - 1) // 2)
        if row["trajectory_seed"] != expected_trajectory_seed:
            raise ValueError(f"PAIRED_EFFECT_NO_GO: capture32 row {slot} trajectory seed drifted")
        if row["writer_turn0_request_seed"] != expected_request:
            raise ValueError(f"PAIRED_EFFECT_NO_GO: capture32 row {slot} turn-0 seed drifted")
        if row["total_writer_turns"] != expected_total_turns \
                or row["intervention_writer_turn"] != expected_intervention_turn \
                or expected_total_turns < 3 \
                or expected_intervention_turn >= expected_total_turns - 1:
            raise ValueError(f"PAIRED_EFFECT_NO_GO: capture32 row {slot} lacks prefix/future")
        stable_examples.append(row["stable_example_id"])
        stable_roots.append(row["stable_root_id"])
        stable_writes.append(row["stable_write_id"])
    if any(len(set(items)) != CAPTURE32_COUNT for items in (
        stable_examples, stable_roots, stable_writes
    )) or set(stable_examples).intersection(pilot_ids):
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 identities duplicate or reuse capture4")

    assignments = _capture32_fold_assignments(stable_examples)
    if any(row["crossfit_fold"] != assignments[row["stable_example_id"]] for row in rows):
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 row fold mapping drifted")
    membership = {
        str(fold): sorted(stable_id for stable_id, assigned in assignments.items()
                          if assigned == fold)
        for fold in range(CAPTURE32_FOLD_COUNT)
    }
    folds = value.get("folds")
    if folds != {
        "fold_count": 4,
        "assignment_rule": FOLD_RULE,
        "expected_fold_sizes": [8, 8, 8, 8],
        "membership": membership,
        "membership_sha256": canonical_sha256(membership),
        "frozen_before_first_generate": True,
    }:
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 4x8 fold contract drifted")
    inventory = value.get("inventory")
    if inventory != {
        "selected_inventory_sha256": EXPECTED_CAPTURE32_SELECTED_INVENTORY_SHA256,
        "stable_example_ids_sha256": canonical_sha256(sorted(stable_examples)),
        "stable_root_ids_sha256": canonical_sha256(sorted(stable_roots)),
        "stable_write_ids_sha256": canonical_sha256(sorted(stable_writes)),
    } or inventory["selected_inventory_sha256"] != canonical_sha256(rows):
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 inventory hashes drifted")

    expected_scorer = {
        "kind": "standardized_ridge",
        "ridge": 1.0,
        "fold_count": 4,
        "fold_rule": FOLD_RULE,
        "feature_schema": list(FEATURE_SCHEMA),
        "feature_schema_sha256": canonical_sha256(list(FEATURE_SCHEMA)),
        "standardization": "fit_fold_population_mean_and_std_ddof0",
        "baseline": "fit_fold_target_mean",
        "score_clipping": "none",
        "missing_or_nonfinite_input": "FAIL",
        "hyperparameter_tuning_on_capture32": "forbidden",
        "outcome_hidden_for_scored_row": True,
        "deployment_model_role": "diagnostic_full_capture_fit_not_authorized",
    }
    if value.get("scorer") != expected_scorer:
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 scorer preregistration drifted")

    expected_admissibility = {
        "required_unique_stable_examples": 32,
        "nontrivial_effect_epsilon": 0.01,
        "minimum_nontrivial_effect_examples": 8,
        "effect_bin_precision": 0.000001,
        "minimum_distinct_effect_bins": 3,
        "minimum_mean_absolute_effect": 0.02,
        "minimum_target_variance": 0.0001,
        "minimum_crossfit_mse_improvement_fraction": 0.05,
        "minimum_crossfit_pearson_correlation": 0.2,
        "minimum_folds_with_positive_mse_improvement": 3,
        "minimum_heldout_examples_per_fold": 8,
        "minimum_fit_examples_per_fold": 24,
    }
    if value.get("admissibility") != expected_admissibility:
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 admissibility thresholds drifted")
    expected_attrition = {
        "commitment_point": "capture32_p0_pass_before_first_generate",
        "expected_pairs": 32,
        "required_arms_per_pair": ["COMMIT", "RETAIN"],
        "partial_after_commitment": "FAIL",
        "missing_before_commitment": "PENDING",
        "replacement": "forbidden",
        "stitching_runs": "forbidden",
        "restart_policy": "new_run_id_and_full_32_only",
        "capture4_may_fill_missing": False,
    }
    if value.get("attrition") != expected_attrition:
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 attrition policy drifted")
    claims = value.get("claim_boundary")
    if not isinstance(claims, Mapping) or any(claims.get(field) is not expected for field, expected in {
        "development_admissibility_only": True,
        "capture4_is_pipeline_pilot_only": True,
        "trainer_attached": False,
        "method_selected": False,
        "training_authorized": False,
        "paper_performance_result": False,
        "causal_effect_claim": False,
    }.items()) or claims.get("s128_label_reuse_warning") != (
        "These 32 outcomes are development evidence; using them to fit/select a method "
        "precludes treating those same rows as untouched confirmatory evaluation."
    ):
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 claim boundary drifted")
    return json.loads(canonical_json(value))


def order_and_validate_capture32_pairs(
    pairs: Sequence[Mapping[str, Any]], preregistration: Mapping[str, Any], *,
    decoder: Any = None,
) -> list[dict[str, Any]]:
    """Require the exact 32 preregistered pairs, then return frozen cohort order."""
    prereg = validate_capture32_preregistration(preregistration)
    if len(pairs) != CAPTURE32_COUNT:
        raise ValueError(
            f"PAIRED_EFFECT_NO_GO: capture32 attrition {len(pairs)} != {CAPTURE32_COUNT}"
        )
    expected_rows = prereg["selected_inventory"]
    expected_by_write = {row["stable_write_id"]: row for row in expected_rows}
    observed: dict[str, dict[str, Any]] = {}
    for raw_pair in pairs:
        pair = validate_pair_record(raw_pair, decoder=decoder)
        write_id = pair["stable_write_id"]
        expected = expected_by_write.get(write_id)
        if expected is None or write_id in observed:
            raise ValueError("PAIRED_EFFECT_NO_GO: capture32 has replacement/duplicate pair")
        for field in (*STABLE_FIELDS, "stable_example_id", "stable_root_id", "stable_write_id",
                      "trajectory_seed", "intervention_writer_turn", "total_writer_turns"):
            frozen = prereg["source"]["eval_manifest_hash"] if field == "eval_manifest_hash" \
                else expected[field]
            if pair[field] != frozen:
                raise ValueError(f"PAIRED_EFFECT_NO_GO: capture32 pair differs from {field}")
        if pair["question_token_ids_sha256"] != expected["question_token_ids_sha256"]:
            raise ValueError("PAIRED_EFFECT_NO_GO: capture32 pair question tokens drifted")
        reconstructed_context = [
            token
            for generation in [*pair["prefix_turns"], pair["candidate"]]
            for token in generation["prompt"]["chunk_token_ids"]
        ] + [
            token
            for chunk in pair["shared_contract"]["future_chunks"]
            for token in chunk["token_ids"]
        ]
        if canonical_sha256(reconstructed_context) != expected["context_token_ids_sha256"]:
            raise ValueError("PAIRED_EFFECT_NO_GO: capture32 pair context tokens drifted")
        if pair["prefix_turns"][0]["prompt"]["token_ids_sha256"] != expected[
            "writer_turn0_prompt_token_sha256"
        ]:
            raise ValueError("PAIRED_EFFECT_NO_GO: capture32 pair turn-0 prompt drifted")
        observed[write_id] = pair
    if set(observed) != set(expected_by_write):
        raise ValueError("PAIRED_EFFECT_NO_GO: capture32 frozen inventory is incomplete")
    return [observed[row["stable_write_id"]] for row in expected_rows]


def build_prebranch_candidate_payload(
    pair: Mapping[str, Any], *, validate: bool = True
) -> dict[str, Any]:
    """Project a validated pair to the exact outcome-free inference input."""
    checked = validate_pair_record(pair) if validate else dict(pair)
    unsigned = {
        "schema": PREBRANCH_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "stable_identity": {field: checked[field] for field in STABLE_FIELDS},
        "stable_example_id": checked["stable_example_id"],
        "stable_root_id": checked["stable_root_id"],
        "stable_write_id": checked["stable_write_id"],
        "trajectory_seed": checked["trajectory_seed"],
        "intervention_writer_turn": checked["intervention_writer_turn"],
        "total_writer_turns": checked["total_writer_turns"],
        "question_token_ids_sha256": checked["question_token_ids_sha256"],
        "old_state": checked["old_state"],
        "candidate_generation_count": checked["candidate_generation_count"],
        "candidate_materialized_before_arm_start": checked[
            "candidate_materialized_before_arm_start"
        ],
        "candidate": checked["candidate"],
        "shared_contract_sha256": checked["shared_contract_sha256"],
    }
    return {**unsigned, "prebranch_payload_sha256": canonical_sha256(unsigned)}


def validate_prebranch_candidate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an online scorer input without requiring any future outcome."""
    if not isinstance(payload, Mapping):
        raise ValueError("PAIRED_EFFECT_NO_GO: prebranch payload must be an object")
    allowed = {
        "schema", "candidate_id", "stable_identity", "stable_example_id", "stable_root_id",
        "stable_write_id", "trajectory_seed", "intervention_writer_turn",
        "total_writer_turns", "question_token_ids_sha256", "old_state",
        "candidate_generation_count", "candidate_materialized_before_arm_start",
        "candidate", "shared_contract_sha256", "prebranch_payload_sha256",
    }
    if set(payload) != allowed:
        raise ValueError("PAIRED_EFFECT_NO_GO: prebranch payload has extra/missing fields")
    unsigned = {key: value for key, value in payload.items()
                if key != "prebranch_payload_sha256"}
    if payload.get("prebranch_payload_sha256") != canonical_sha256(unsigned):
        raise ValueError("PAIRED_EFFECT_NO_GO: prebranch payload digest mismatch")
    if payload.get("schema") != PREBRANCH_SCHEMA or payload.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("PAIRED_EFFECT_NO_GO: prebranch payload identity mismatch")
    for field in ("stable_example_id", "stable_root_id", "stable_write_id",
                  "question_token_ids_sha256", "shared_contract_sha256"):
        require_sha256(payload.get(field), f"prebranch.{field}")
    trajectory_seed = payload.get("trajectory_seed")
    intervention = payload.get("intervention_writer_turn")
    total = payload.get("total_writer_turns")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (
        trajectory_seed, intervention, total
    )) or trajectory_seed < 0 or intervention < 1 or total <= intervention:
        raise ValueError("PAIRED_EFFECT_NO_GO: prebranch schedule is invalid")
    if payload.get("candidate_generation_count") != 1 \
            or payload.get("candidate_materialized_before_arm_start") is not True:
        raise ValueError("PAIRED_EFFECT_NO_GO: candidate was not materialized exactly once")
    stable = stable_capture_ids(
        payload.get("stable_identity", {}),
        trajectory_seed=int(trajectory_seed),
        writer_turn=int(intervention),
    )
    if any(payload.get(field) != stable[field] for field in (
        "stable_example_id", "stable_root_id", "stable_write_id"
    )):
        raise ValueError("PAIRED_EFFECT_NO_GO: stable identity does not reproduce")
    old_state = build_state_blob(payload.get("old_state", {}))
    candidate = _writer_generation(
        payload.get("candidate", {}),
        field="prebranch.candidate",
        stable_write_id_value=str(payload["stable_write_id"]),
        expected_phase="candidate_writer",
        expected_arm=SHARED_ARM,
        expected_turn=int(intervention),
    )
    expected_seed = derive_turn_request_seeds(
        [int(trajectory_seed)], [0], int(intervention)
    )[0]
    if candidate["request_seed"] != expected_seed:
        raise ValueError("PAIRED_EFFECT_NO_GO: candidate request seed drifted")
    loaded = candidate["prompt"]["loaded_state_receipt"]
    if loaded["source_role"] != "old_state" \
            or canonical_json(loaded["state"]) != canonical_json(old_state):
        raise ValueError("PAIRED_EFFECT_NO_GO: candidate did not load exact old state")
    if candidate["state_after"]["bytes_sha256"] == old_state["bytes_sha256"]:
        raise ValueError("PAIRED_EFFECT_NO_GO: candidate is identical to old state")
    rebuilt_unsigned = {
        **unsigned,
        "old_state": old_state,
        "candidate": candidate,
    }
    rebuilt = {
        **rebuilt_unsigned,
        "prebranch_payload_sha256": canonical_sha256(rebuilt_unsigned),
    }
    if canonical_json(rebuilt) != canonical_json(dict(payload)):
        raise ValueError("PAIRED_EFFECT_NO_GO: prebranch payload is non-canonical")
    return rebuilt


def extract_prebranch_features(
    payload: Mapping[str, Any], *, validate: bool = True
) -> dict[str, Any]:
    """Extract the model vector from a payload with no outcome-bearing fields."""
    checked = validate_prebranch_candidate_payload(payload) if validate else dict(payload)
    old_ids = [int(value) for value in checked["old_state"]["token_ids"]]
    candidate = checked["candidate"]
    candidate_ids = [int(value) for value in candidate["state_after"]["token_ids"]]
    old_set, candidate_set = set(old_ids), set(candidate_ids)
    union = old_set | candidate_set
    intersection = old_set & candidate_set
    words = _WORDS.findall(str(candidate["output_text"]))
    lower = [word.lower() for word in words]
    candidate_count, old_count = len(candidate_ids), len(old_ids)
    prompt_count = len(candidate["prompt"]["token_ids"])
    chunk_count = len(candidate["prompt"]["chunk_token_ids"])
    intervention = int(checked["intervention_writer_turn"])
    total_turns = int(checked["total_writer_turns"])
    features = [
        math.log1p(old_count),
        math.log1p(candidate_count),
        (candidate_count - old_count) / max(1.0, float(old_count)),
        _ratio(len(intersection), len(union)),
        _ratio(len(intersection), len(old_set)),
        _ratio(len(intersection), len(candidate_set)),
        _ratio(len(candidate_set), candidate_count),
        math.log1p(prompt_count),
        math.log1p(chunk_count),
        intervention / max(1.0, float(total_turns - 1)),
        _ratio(sum(word in _RELATION_MARKERS for word in lower), len(words)),
        _ratio(sum(word in _NEGATION_MARKERS for word in lower), len(words)),
        _ratio(sum(word in _SOURCE_MARKERS for word in lower), len(words)),
        _ratio(sum(any(char.isdigit() for char in word) for word in words), len(words)),
    ]
    if len(features) != len(FEATURE_SCHEMA) or any(not math.isfinite(value) for value in features):
        raise ValueError("PAIRED_EFFECT_NO_GO: outcome-hidden feature extraction failed")
    return {
        "stable_example_id": checked["stable_example_id"],
        "stable_write_id": checked["stable_write_id"],
        "feature_schema": list(FEATURE_SCHEMA),
        "feature_schema_sha256": canonical_sha256(list(FEATURE_SCHEMA)),
        "feature_input_sha256": checked["prebranch_payload_sha256"],
        "feature_vector": features,
        "feature_vector_sha256": canonical_sha256(features),
        "outcome_hidden_for_scored_row": True,
        "forbidden_input_fields_absent": [
            "arms", "ground_truth", "terminal_answer", "reward", "outcome",
        ],
    }


def extract_outcome_hidden_features(
    pair: Mapping[str, Any], *, validate: bool = True
) -> dict[str, Any]:
    checked = validate_pair_record(pair) if validate else dict(pair)
    payload = build_prebranch_candidate_payload(checked, validate=False)
    result = extract_prebranch_features(payload)
    return {
        **result,
        "pair_id": checked["pair_id"],
    }


def paired_outcome(pair: Mapping[str, Any], *, validate: bool = True) -> dict[str, Any]:
    checked = validate_pair_record(pair) if validate else dict(pair)
    commit = checked["arms"]["COMMIT"]["final_reader"]["outcome"]
    retain = checked["arms"]["RETAIN"]["final_reader"]["outcome"]
    commit_f1 = _finite_number(commit.get("token_f1"), "COMMIT token_f1")
    retain_f1 = _finite_number(retain.get("token_f1"), "RETAIN token_f1")
    commit_em = _finite_number(commit.get("exact_match"), "COMMIT exact_match")
    retain_em = _finite_number(retain.get("exact_match"), "RETAIN exact_match")
    return {
        "stable_example_id": checked["stable_example_id"],
        "stable_write_id": checked["stable_write_id"],
        "pair_id": checked["pair_id"],
        "target_name": TARGET_NAME,
        "commit_token_f1": commit_f1,
        "retain_token_f1": retain_f1,
        "paired_effect_target": commit_f1 - retain_f1,
        "commit_exact_match": commit_em,
        "retain_exact_match": retain_em,
        "paired_exact_match_difference": commit_em - retain_em,
    }


def observations_from_pairs(
    pairs: Sequence[Mapping[str, Any]], *, validate: bool = True
) -> list[dict[str, Any]]:
    if not pairs:
        raise ValueError("PAIRED_EFFECT_PENDING: no authenticated capture pairs")
    result: list[dict[str, Any]] = []
    seen_examples: set[str] = set()
    seen_writes: set[str] = set()
    seen_pairs: set[str] = set()
    for pair in pairs:
        checked = validate_pair_record(pair) if validate else dict(pair)
        feature = extract_outcome_hidden_features(checked, validate=False)
        outcome = paired_outcome(checked, validate=False)
        identities = (
            str(checked["stable_example_id"]), str(checked["stable_write_id"]),
            str(checked["pair_id"]),
        )
        if identities[0] in seen_examples or identities[1] in seen_writes \
                or identities[2] in seen_pairs:
            raise ValueError("PAIRED_EFFECT_NO_GO: duplicate stable capture identity")
        seen_examples.add(identities[0])
        seen_writes.add(identities[1])
        seen_pairs.add(identities[2])
        result.append({**feature, **outcome})
    return result


def stable_fold_assignments(
    stable_example_ids: Sequence[str], *, fold_count: int
) -> dict[str, int]:
    unique = sorted(set(stable_example_ids))
    if fold_count < 2 or len(unique) < fold_count or len(unique) != len(stable_example_ids):
        raise ValueError(
            "PAIRED_EFFECT_NO_GO: stable-example crossfit needs unique examples and populated folds"
        )
    return {stable_id: index % fold_count for index, stable_id in enumerate(unique)}


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [list(map(float, row)) + [float(value)]
                 for row, value in zip(matrix, vector)]
    if size == 0 or any(len(row) != size + 1 for row in augmented):
        raise ValueError("PAIRED_EFFECT_NO_GO: malformed ridge system")
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("PAIRED_EFFECT_NO_GO: singular ridge system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * reference
                    for value, reference in zip(augmented[row], augmented[column])
                ]
    solution = [augmented[index][-1] for index in range(size)]
    if any(not math.isfinite(value) for value in solution):
        raise ValueError("PAIRED_EFFECT_NO_GO: non-finite ridge solution")
    return solution


def _fit_ridge(
    feature_rows: Sequence[Sequence[float]], targets: Sequence[float], *, ridge: float
) -> dict[str, Any]:
    if not feature_rows or len(feature_rows) != len(targets) \
            or isinstance(ridge, bool) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("PAIRED_EFFECT_NO_GO: invalid ridge fit input")
    width = len(feature_rows[0])
    if width != len(FEATURE_SCHEMA) or any(len(row) != width for row in feature_rows):
        raise ValueError("PAIRED_EFFECT_NO_GO: feature width drift")
    means = [sum(float(row[column]) for row in feature_rows) / len(feature_rows)
             for column in range(width)]
    scales: list[float] = []
    for column, mean in enumerate(means):
        variance = sum((float(row[column]) - mean) ** 2 for row in feature_rows) / len(feature_rows)
        scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
    design = [[1.0] + [
        (float(value) - means[column]) / scales[column]
        for column, value in enumerate(row)
    ] for row in feature_rows]
    dimension = width + 1
    gram = [[sum(row[left] * row[right] for row in design)
             for right in range(dimension)] for left in range(dimension)]
    # A tiny intercept penalty avoids a degenerate matrix while preserving the
    # usual effectively-unpenalized intercept semantics.
    gram[0][0] += ridge * 1e-9
    for index in range(1, dimension):
        gram[index][index] += ridge
    rhs = [sum(row[column] * float(target) for row, target in zip(design, targets))
           for column in range(dimension)]
    coefficients = _solve_linear(gram, rhs)
    return {
        "intercept": coefficients[0],
        "weights": coefficients[1:],
        "feature_means": means,
        "feature_scales": scales,
    }


def _predict(model: Mapping[str, Any], features: Sequence[float]) -> float:
    transformed = [
        (float(value) - float(model["feature_means"][index]))
        / float(model["feature_scales"][index])
        for index, value in enumerate(features)
    ]
    score = float(model["intercept"]) + sum(
        float(weight) * value for weight, value in zip(model["weights"], transformed)
    )
    if not math.isfinite(score):
        raise ValueError("PAIRED_EFFECT_NO_GO: non-finite held-out score")
    return score


def build_crossfit_bundle_from_observations(
    observations: Sequence[Mapping[str, Any]], *, fold_count: int, ridge: float,
    expected_fold_assignments: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("PAIRED_EFFECT_PENDING: no paired observations")
    # Canonicalize before every fit/hash. Ledger serialization order is not a
    # scientific input and must not alter floating-point accumulation, fold
    # models, held-out scores, or the evidence digest.
    ordered_observations = sorted(
        [dict(row) for row in observations],
        key=lambda row: str(row["stable_example_id"]),
    )
    stable_ids = [str(row["stable_example_id"]) for row in ordered_observations]
    computed_assignments = stable_fold_assignments(stable_ids, fold_count=fold_count)
    if expected_fold_assignments is None:
        assignments = computed_assignments
    else:
        assignments = {str(key): int(value) for key, value in expected_fold_assignments.items()}
        if assignments != computed_assignments:
            raise ValueError("PAIRED_EFFECT_NO_GO: crossfit fold assignment differs from preregistration")
    fold_membership = {
        str(fold): sorted(stable_id for stable_id, assigned in assignments.items()
                          if assigned == fold)
        for fold in range(fold_count)
    }
    models: dict[str, Any] = {}
    score_rows: list[dict[str, Any]] = []
    for score_fold in range(fold_count):
        fit = [row for row in ordered_observations
               if assignments[str(row["stable_example_id"])] != score_fold]
        heldout = [row for row in ordered_observations
                   if assignments[str(row["stable_example_id"])] == score_fold]
        fit_ids = sorted(str(row["stable_example_id"]) for row in fit)
        heldout_ids = {str(row["stable_example_id"]) for row in heldout}
        if not fit or not heldout or heldout_ids.intersection(fit_ids):
            raise ValueError("PAIRED_EFFECT_NO_GO: crossfit membership leakage/attrition")
        model = _fit_ridge(
            [row["feature_vector"] for row in fit],
            [float(row["paired_effect_target"]) for row in fit],
            ridge=ridge,
        )
        unsigned_model = {
            "score_fold": score_fold,
            "fit_stable_example_ids": fit_ids,
            "fit_membership_sha256": canonical_sha256(fit_ids),
            **model,
        }
        model_sha = canonical_sha256(unsigned_model)
        models[str(score_fold)] = {**unsigned_model, "model_sha256": model_sha}
        for row in heldout:
            score_rows.append({
                "record_type": "paired_effect_heldout_score",
                "stable_example_id": row["stable_example_id"],
                "stable_write_id": row["stable_write_id"],
                "pair_id": row["pair_id"],
                "score_fold": score_fold,
                "fit_membership_sha256": unsigned_model["fit_membership_sha256"],
                "model_sha256": model_sha,
                "feature_input_sha256": row["feature_input_sha256"],
                "feature_vector_sha256": row["feature_vector_sha256"],
                "paired_effect_score": _predict(model, row["feature_vector"]),
                "outcome_hidden_for_scored_row": True,
            })
    score_rows.sort(key=lambda row: str(row["stable_example_id"]))
    all_fit_ids = sorted(stable_ids)
    deployment_fit = _fit_ridge(
        [row["feature_vector"] for row in ordered_observations],
        [float(row["paired_effect_target"]) for row in ordered_observations],
        ridge=ridge,
    )
    unsigned_deployment_model = {
        "model_role": "diagnostic_full_capture_fit",
        "fit_stable_example_ids": all_fit_ids,
        "fit_membership_sha256": canonical_sha256(all_fit_ids),
        **deployment_fit,
        "outcome_fields_accepted_at_inference": False,
        "deployment_use_authorized": False,
    }
    deployment_model = {
        **unsigned_deployment_model,
        "model_sha256": canonical_sha256(unsigned_deployment_model),
    }
    unsigned = {
        "schema": BUNDLE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "target_name": TARGET_NAME,
        "feature_schema": list(FEATURE_SCHEMA),
        "feature_schema_sha256": canonical_sha256(list(FEATURE_SCHEMA)),
        "fold_rule": FOLD_RULE,
        "fold_count": fold_count,
        "fold_assignments": dict(sorted(assignments.items())),
        "fold_membership_sha256": canonical_sha256(fold_membership),
        "ridge": float(ridge),
        "capture_pair_ids": sorted(str(row["pair_id"]) for row in observations),
        "capture_observations_sha256": canonical_sha256(ordered_observations),
        "models": models,
        "scores": score_rows,
        "deployment_model": deployment_model,
        "stable_example_grouped_crossfit": True,
        "outcome_hidden_at_scoring": True,
        "training_authorized": False,
        "method_selected": False,
    }
    return {**unsigned, "bundle_sha256": canonical_sha256(unsigned)}


def build_crossfit_bundle(
    pairs: Sequence[Mapping[str, Any]], *, fold_count: int, ridge: float,
    capture32_preregistration: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_assignments: dict[str, int] | None = None
    if capture32_preregistration is not None:
        checked = validate_capture32_preregistration(capture32_preregistration)
        pairs = order_and_validate_capture32_pairs(pairs, checked)
        expected_assignments = {
            row["stable_example_id"]: row["crossfit_fold"]
            for row in checked["selected_inventory"]
        }
    observations = observations_from_pairs(pairs)
    return (
        build_crossfit_bundle_from_observations(
            observations, fold_count=fold_count, ridge=ridge,
            expected_fold_assignments=expected_assignments,
        ),
        observations,
    )


def validate_crossfit_bundle(
    bundle: Mapping[str, Any], observations: Sequence[Mapping[str, Any]], *,
    expected_fold_assignments: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if bundle.get("bundle_sha256") != canonical_sha256(unsigned):
        raise ValueError("PAIRED_EFFECT_NO_GO: crossfit bundle digest mismatch")
    expected = build_crossfit_bundle_from_observations(
        observations,
        fold_count=int(bundle.get("fold_count", 0)),
        ridge=_finite_number(bundle.get("ridge"), "ridge"),
        expected_fold_assignments=expected_fold_assignments,
    )
    if canonical_json(bundle) != canonical_json(expected):
        raise ValueError("PAIRED_EFFECT_NO_GO: crossfit bundle does not reproduce")
    for row in expected["scores"]:
        model = expected["models"][str(row["score_fold"])]
        if row["stable_example_id"] in model["fit_stable_example_ids"]:
            raise AssertionError("PAIRED_EFFECT_NO_GO: scored example entered fit membership")
    return expected


def _compute_centered_trajectory_bonuses(
    *,
    scores: Sequence[float],
    qa_rewards: Sequence[float],
    stable_group_ids: Sequence[str],
    eligible: Sequence[bool],
    exact_correct: Sequence[bool],
    lambda_: float,
) -> tuple[list[float], list[bool]]:
    count = len(scores)
    if count < 1 or not all(len(values) == count for values in (
        qa_rewards, stable_group_ids, eligible, exact_correct
    )):
        raise ValueError("PAIRED_EFFECT_NO_GO: trajectory metadata length mismatch")
    if isinstance(lambda_, bool) or not math.isfinite(float(lambda_)) or lambda_ <= 0:
        raise ValueError("PAIRED_EFFECT_NO_GO: lambda must be finite and positive")
    result = [0.0] * count
    routed = [False] * count
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in stable_group_ids):
        raise ValueError("PAIRED_EFFECT_NO_GO: routing group IDs are not stable SHA identities")
    for stable_group_id in dict.fromkeys(stable_group_ids):
        indices = [index for index, value in enumerate(stable_group_ids)
                   if value == stable_group_id]
        if len(indices) < 2:
            continue
        rewards = [_finite_number(qa_rewards[index], "qa_reward") for index in indices]
        if any(value != rewards[0] for value in rewards[1:]):
            continue
        if all(bool(exact_correct[index]) for index in indices):
            # A perfectly solved group carries no useful credit assignment
            # ambiguity.  It is protected, not an exceptional batch.
            continue
        if not all(bool(eligible[index]) for index in indices):
            continue
        local = [_finite_number(scores[index], "heldout_score") for index in indices]
        mean = sum(local) / len(local)
        for index, value in zip(indices, local):
            result[index] = float(lambda_) * (value - mean)
            routed[index] = True
    return result, routed


def audit_writer_credit_routing(
    *,
    trajectory_qa_advantage: Any,
    qa_reward: Any,
    stable_group_ids: Sequence[str],
    diagnostic_scores: Any,
    sample_index: Any,
    response_mask: Any,
    final_mask: Any,
    eligible: Any,
    exact_correct: Any,
    lambda_: float,
) -> tuple[Any, list[dict[str, Any]]]:
    """Offline tensor audit for the proposed writer-only routing semantics.

    This function is intentionally not a trainer entrypoint and accepts no
    deployment bundle.  A later integration must join authenticated score
    receipts by stable identity; this diagnostic cannot authorize that step.
    """
    import torch

    if TRAINER_INTEGRATION_AUTHORIZED:
        raise AssertionError("paired-effect diagnostic unexpectedly authorized trainer use")
    trajectory_tensors = {
        "trajectory_qa_advantage": trajectory_qa_advantage,
        "qa_reward": qa_reward,
        "diagnostic_scores": diagnostic_scores,
        "eligible": eligible,
        "exact_correct": exact_correct,
    }
    if any(not isinstance(value, torch.Tensor) for value in trajectory_tensors.values()):
        raise ValueError("PAIRED_EFFECT_NO_GO: trajectory inputs must be torch tensors")
    if any(value.ndim != 1 for value in trajectory_tensors.values()):
        raise ValueError("PAIRED_EFFECT_NO_GO: trajectory tensors must be one-dimensional")
    if any(not torch.is_floating_point(trajectory_tensors[name]) for name in (
        "trajectory_qa_advantage", "qa_reward", "diagnostic_scores"
    )):
        raise ValueError("PAIRED_EFFECT_NO_GO: trajectory values/scores must be floating point")
    count = len(diagnostic_scores)
    if count < 1 or len(stable_group_ids) != count \
            or any(len(value) != count for value in trajectory_tensors.values()):
        raise ValueError("PAIRED_EFFECT_NO_GO: trajectory metadata length mismatch")
    if eligible.dtype != torch.bool or exact_correct.dtype != torch.bool:
        raise ValueError("PAIRED_EFFECT_NO_GO: routing eligibility masks must be bool")
    row_tensors = {
        "sample_index": sample_index,
        "response_mask": response_mask,
        "final_mask": final_mask,
    }
    if any(not isinstance(value, torch.Tensor) for value in row_tensors.values()):
        raise ValueError("PAIRED_EFFECT_NO_GO: actor row inputs must be torch tensors")
    if sample_index.ndim != 1 or response_mask.ndim != 2 \
            or final_mask.ndim != 1 or len(sample_index) != len(response_mask) \
            or len(final_mask) != len(sample_index):
        raise ValueError("PAIRED_EFFECT_NO_GO: actor row tensors are misaligned")
    if final_mask.dtype != torch.bool:
        raise ValueError("PAIRED_EFFECT_NO_GO: final mask must be bool")
    if sample_index.dtype == torch.bool or sample_index.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64
    ):
        raise ValueError("PAIRED_EFFECT_NO_GO: sample_index must be integer")
    if set(int(value) for value in sample_index.detach().cpu()) != set(range(count)):
        raise ValueError("PAIRED_EFFECT_NO_GO: sample_index coverage mismatch")
    if not bool(torch.all((response_mask == 0) | (response_mask == 1))):
        raise ValueError("PAIRED_EFFECT_NO_GO: response mask must be binary")
    bonuses, routed_values = _compute_centered_trajectory_bonuses(
        scores=[float(value) for value in diagnostic_scores.detach().cpu()],
        qa_rewards=[float(value) for value in qa_reward.detach().cpu()],
        stable_group_ids=stable_group_ids,
        eligible=[bool(value) for value in eligible.detach().cpu()],
        exact_correct=[bool(value) for value in exact_correct.detach().cpu()],
        lambda_=lambda_,
    )
    device = trajectory_qa_advantage.device
    sample_index = sample_index.to(device)
    response_mask = response_mask.to(device)
    final_mask = final_mask.to(device=device, dtype=torch.bool)
    bonus = torch.tensor(bonuses, dtype=trajectory_qa_advantage.dtype, device=device)
    routed = torch.tensor(routed_values, dtype=torch.bool, device=device)
    final_counts = torch.zeros(count, dtype=torch.long, device=device)
    final_counts.scatter_add_(0, sample_index, final_mask.to(torch.long))
    if not bool(torch.all(final_counts == 1)):
        raise ValueError("PAIRED_EFFECT_NO_GO: each trajectory needs exactly one final row")
    writer_mask = response_mask * (~final_mask).unsqueeze(-1)
    token_counts = torch.zeros(count, dtype=torch.long, device=device)
    token_counts.scatter_add_(0, sample_index, writer_mask.sum(-1).to(torch.long))
    if bool(torch.any(routed & (token_counts < 1))):
        raise ValueError("PAIRED_EFFECT_NO_GO: routed trajectory has no writer token")
    per_token = bonus / token_counts.clamp_min(1).to(bonus.dtype)
    token_bonus = per_token[sample_index].unsqueeze(-1) * writer_mask
    base = trajectory_qa_advantage[sample_index].unsqueeze(-1).expand_as(response_mask) * response_mask
    # Never add floating zero to protected cells: this preserves even a signed
    # zero bit pattern on final/non-target rows.
    result = base.clone()
    changed = token_bonus != 0
    result[changed] = base[changed] + token_bonus[changed]
    if not torch.equal(result[final_mask], base[final_mask]):
        raise AssertionError("PAIRED_EFFECT_NO_GO: final rows changed")
    if not torch.equal(result[~routed[sample_index]], base[~routed[sample_index]]):
        raise AssertionError("PAIRED_EFFECT_NO_GO: non-target rows changed")
    audit: list[dict[str, Any]] = []
    for index in range(count):
        local = sample_index == index
        delivered = float(token_bonus[local].sum().detach().cpu())
        expected = float(bonus[index].detach().cpu())
        if abs(delivered - expected) > 1e-6 * max(1.0, abs(expected)):
            raise AssertionError("PAIRED_EFFECT_NO_GO: trajectory-total normalization failed")
        audit.append({
            "trajectory_index": index,
            "routed": bool(routed[index].item()),
            "centered_trajectory_bonus": expected,
            "delivered_writer_token_bonus": delivered,
            "writer_token_count": int(token_counts[index].item()),
            "normalization": "trajectory_total_over_all_valid_writer_tokens",
            "writer_only": True,
            "final_answer_bonus": 0.0,
            "diagnostic_only": True,
            "trainer_integration_authorized": False,
        })
    return result, audit

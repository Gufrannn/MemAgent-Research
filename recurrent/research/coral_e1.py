"""Trusted, fixed contracts for CORAL occupancy-response measurements."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable


SKETCH_SPEC = {
    "schema": "memagent.coral.fixed-count-sketch.v3",
    "independent_bases": 4,
    "buckets_per_basis": 256,
    "output_dimensions": 1024,
    "norm_scale": "divide_concatenated_sketch_by_sqrt_independent_bases",
    "world_size": 2,
    "parameter_order": "fsdp_local_named_parameters_iteration",
    "element_index": "rank_local_flat_index_with_parameter_ordinal",
    "parameter_multiplier": 1442695040888963407,
    "rank_multiplier": 3202034522624059733,
    "basis_multiplier": 3935559000370003845,
    "mixer_multiplier_1": 6364136223846793005,
    "mixer_multiplier_2": 2691343689449507681,
    "mixer_shifts": [30, 27, 31],
    "sign_bit": 17,
}


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SKETCH_BASIS_SHA256 = canonical_sha256(SKETCH_SPEC)

ORACLE_REPORT_FIELDS = {
    "schema", "status", "decision", "world_size", "backend",
    "basis_sha256", "full_gradient_elements", "padded_shard_elements",
    "full_gradient_max_abs_error", "sketch_max_abs_error",
    "sketch_assembly_aperture",
    "projection_relative_norm_error", "projection_error_aperture",
    "collision_calibration_elements", "collision_calibration_buckets_per_basis",
    "collision_calibration_exact_norm", "collision_calibration_projected_norm",
    "collision_calibration_relative_norm_error",
    "collision_calibration_error_aperture", "two_rank_local_denominators",
    "denominator_gradient_closure_max_abs_error",
    "denominator_gradient_closure_aperture", "report_sha256",
}


def _finite_number(value, field: str, low: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"CORAL_E1_NO_GO: oracle {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < low:
        raise ValueError(f"CORAL_E1_NO_GO: oracle {field} outside range")
    return result


def validate_fsdp_sketch_oracle_report(report) -> None:
    """Strictly validate the complete two-rank v3 oracle contract."""
    if not isinstance(report, dict) or set(report) != ORACLE_REPORT_FIELDS:
        raise ValueError("CORAL_E1_NO_GO: FSDP sketch oracle fields")
    unsigned = {key: value for key, value in report.items()
                if key != "report_sha256"}
    if report["schema"] != "memagent.coral.e1-fsdp-sketch-oracle.v3" \
            or report["status"] != "PASS" \
            or report["decision"] != "CORAL_E1_SKETCH_ORACLE_PASS" \
            or type(report["world_size"]) is not int \
            or report["world_size"] != 2 \
            or report["backend"] != "nccl" \
            or report["basis_sha256"] != SKETCH_BASIS_SHA256 \
            or report["report_sha256"] != canonical_sha256(unsigned):
        raise ValueError("CORAL_E1_NO_GO: FSDP sketch oracle identity")
    integer_contract = {
        "full_gradient_elements": 221,
        "padded_shard_elements": 222,
        "collision_calibration_elements": 1_000_003,
        "collision_calibration_buckets_per_basis": 256,
    }
    if any(type(report[field]) is not int or report[field] != expected
           for field, expected in integer_contract.items()):
        raise ValueError("CORAL_E1_NO_GO: FSDP sketch oracle dimensions")
    denominators = report["two_rank_local_denominators"]
    if not isinstance(denominators, list) \
            or any(type(value) is not int for value in denominators) \
            or denominators != [5, 7] \
            or report["projection_error_aperture"] != 0.10 \
            or report["collision_calibration_error_aperture"] != 0.10 \
            or report["sketch_assembly_aperture"] != 1e-7 \
            or report["denominator_gradient_closure_aperture"] != 1e-6:
        raise ValueError("CORAL_E1_NO_GO: FSDP sketch oracle aperture drift")
    full_error = _finite_number(
        report["full_gradient_max_abs_error"], "full_gradient_max_abs_error",
    )
    assembly_error = _finite_number(
        report["sketch_max_abs_error"], "sketch_max_abs_error",
    )
    projection_error = _finite_number(
        report["projection_relative_norm_error"], "projection_relative_norm_error",
    )
    dense_exact = _finite_number(
        report["collision_calibration_exact_norm"],
        "collision_calibration_exact_norm", low=1e-30,
    )
    dense_projected = _finite_number(
        report["collision_calibration_projected_norm"],
        "collision_calibration_projected_norm", low=1e-30,
    )
    dense_error = _finite_number(
        report["collision_calibration_relative_norm_error"],
        "collision_calibration_relative_norm_error",
    )
    denominator_error = _finite_number(
        report["denominator_gradient_closure_max_abs_error"],
        "denominator_gradient_closure_max_abs_error",
    )
    if full_error > 2e-5 or assembly_error > report["sketch_assembly_aperture"] \
            or projection_error > report["projection_error_aperture"] \
            or dense_error > report["collision_calibration_error_aperture"] \
            or denominator_error > report["denominator_gradient_closure_aperture"]:
        raise ValueError("CORAL_E1_NO_GO: FSDP sketch oracle error aperture")
    # Recompute the reported dense relative error instead of trusting both the
    # projected/exact norms and their derived summary independently.
    recomputed_dense_error = abs(dense_projected - dense_exact) / dense_exact
    if abs(recomputed_dense_error - dense_error) > 1e-12:
        raise ValueError("CORAL_E1_NO_GO: FSDP sketch oracle derived error")


def sketch_bucket_and_sign(index: int, ordinal: int, rank: int, basis: int):
    """Pure reference for the nonlinear basis-separated coordinate hash."""
    mask = (1 << 63) - 1
    mixed = (
        int(index)
        + int(ordinal) * SKETCH_SPEC["parameter_multiplier"]
        + int(rank) * SKETCH_SPEC["rank_multiplier"]
        + int(basis) * SKETCH_SPEC["basis_multiplier"]
    ) & mask
    mixed = ((mixed ^ (mixed >> 30)) * SKETCH_SPEC["mixer_multiplier_1"]) & mask
    mixed = ((mixed ^ (mixed >> 27)) * SKETCH_SPEC["mixer_multiplier_2"]) & mask
    mixed = (mixed ^ (mixed >> 31)) & mask
    return (
        mixed % SKETCH_SPEC["buckets_per_basis"],
        1 if ((mixed >> SKETCH_SPEC["sign_bit"]) & 1) == 0 else -1,
    )


def fixed_count_sketch(named_gradients: Iterable[tuple[str, object]], rank: int,
                       world_size: int):
    """Return four fixed 256-bucket linear sketches and squared norm.

    This runs after the actual actor loss backward. FSDP owns disjoint local
    gradient shards; the worker wrapper all-reduces both outputs, making the
    result four basis-separated fixed linear sketches of the complete
    distributed gradient.  Concatenation is divided by sqrt(4), so its
    squared norm is the mean of the four projected squared norms.
    """
    import torch

    if world_size != SKETCH_SPEC["world_size"] or rank < 0 or rank >= world_size:
        raise ValueError("CORAL_E1_NO_GO: fixed sketch requires canonical two-rank FSDP")
    device = torch.device("cuda", torch.cuda.current_device())
    sketch = torch.zeros(SKETCH_SPEC["output_dimensions"], device=device, dtype=torch.float64)
    squared_norm = torch.zeros(1, device=device, dtype=torch.float64)
    gradient_count = 0
    modulus_mask = (1 << 63) - 1
    for ordinal, (_, parameter) in enumerate(named_gradients):
        gradient = parameter.grad
        if gradient is None:
            continue
        flat = gradient.detach().reshape(-1)
        gradient_count += 1
        # Work in bounded chunks: no 64 x number-of-parameters allocation.
        for start in range(0, flat.numel(), 1_000_000):
            values = flat[start:start + 1_000_000].to(
                device=device, dtype=torch.float64, non_blocking=False,
            )
            squared_norm.add_(torch.dot(values, values))
            indices = torch.arange(
                start, start + values.numel(), device=values.device, dtype=torch.int64,
            )
            for basis in range(SKETCH_SPEC["independent_bases"]):
                mixed = (
                    indices
                    + ordinal * SKETCH_SPEC["parameter_multiplier"]
                    + rank * SKETCH_SPEC["rank_multiplier"]
                    + basis * SKETCH_SPEC["basis_multiplier"]
                ) & modulus_mask
                mixed = ((mixed ^ (mixed >> 30))
                         * SKETCH_SPEC["mixer_multiplier_1"]) & modulus_mask
                mixed = ((mixed ^ (mixed >> 27))
                         * SKETCH_SPEC["mixer_multiplier_2"]) & modulus_mask
                mixed = (mixed ^ (mixed >> 31)) & modulus_mask
                buckets = torch.remainder(
                    mixed, SKETCH_SPEC["buckets_per_basis"],
                ) + basis * SKETCH_SPEC["buckets_per_basis"]
                signs = torch.where(
                    ((mixed >> SKETCH_SPEC["sign_bit"]) & 1) == 0,
                    torch.ones_like(values), -torch.ones_like(values),
                )
                sketch.scatter_add_(0, buckets, values * signs)
    if gradient_count == 0:
        raise ValueError("CORAL_E1_NO_GO: actual backward produced no gradients")
    sketch.div_(math.sqrt(SKETCH_SPEC["independent_bases"]))
    return sketch, squared_norm

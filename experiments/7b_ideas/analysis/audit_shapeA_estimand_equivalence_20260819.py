#!/usr/bin/env python3
"""Fail-closed algebra audit for Shape A; emits no p-values or claim authorization."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

REQUIRED = ("stable_example_id", "y_factual", "y_noop", "d_star")


def audit(rows: list[dict]) -> dict:
    if not 2 <= len(rows) <= 128:
        raise ValueError(f"independent stable_example_id n must be in [2,128], got {len(rows)}")
    ids = [str(row.get("stable_example_id", "")) for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("stable_example_id must be present and unique")
    missing = [(i, key) for i, row in enumerate(rows) for key in REQUIRED if key not in row]
    if missing: raise ValueError(f"missing required fields: {missing[:5]}")
    d = np.asarray([row["d_star"] for row in rows], dtype=float)
    yf = np.asarray([row["y_factual"] for row in rows], dtype=float)
    yn = np.asarray([row["y_noop"] for row in rows], dtype=float)
    if not np.isfinite(np.r_[d, yf, yn]).all(): raise ValueError("non-finite audit value")
    paired_x = np.column_stack([np.ones(len(rows)), d])
    if np.linalg.matrix_rank(paired_x) != 2: raise ValueError("paired design matrix is rank deficient")
    paired = np.linalg.lstsq(paired_x, yf - yn, rcond=None)[0][1]
    # Pair fixed effects plus factual-arm main and factual-arm×D. With two
    # balanced rows per pair, the interaction is algebraically the tau slope.
    n = len(rows); y = np.r_[yn, yf]; arm = np.r_[np.zeros(n), np.ones(n)]
    pair_fe = np.vstack([np.eye(n), np.eye(n)])[:, 1:]
    stacked_x = np.column_stack([np.ones(2*n), pair_fe, arm, arm * np.r_[d, d]])
    if np.linalg.matrix_rank(stacked_x) != stacked_x.shape[1]:
        raise ValueError("pair-FE stacked design matrix is rank deficient")
    stacked = np.linalg.lstsq(stacked_x, y, rcond=None)[0][-1]
    if not np.isclose(paired, stacked, rtol=1e-9, atol=1e-10):
        raise ValueError(f"estimand implementation mismatch: paired={paired}, stacked={stacked}")
    return {"audit_only": True, "claim_authorized": False, "p_values_emitted": False,
            "independent_n": n, "paired_slope": float(paired), "stacked_pair_fe_interaction": float(stacked),
            "algebraically_equivalent": True}


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--input", type=Path, required=True); args = p.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    print(json.dumps(audit(rows), sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())

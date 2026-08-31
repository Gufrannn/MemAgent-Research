#!/usr/bin/env python3
"""Summarize P22 scale-sanity runs across model size and retrieval pressure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fget(obj: dict[str, Any], path: list[str], default: Any = math.nan) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def find_file(run_dir: Path, relative_options: list[str]) -> Path | None:
    for rel in relative_options:
        p = run_dir / rel
        if p.exists():
            return p
    return None


def summarize_run(run_dir: Path, label: str) -> list[dict[str, Any]]:
    matrix_path = find_file(
        run_dir,
        [
            "matrix_with_expand_q/longmemeval_operation_value_summary.csv",
            "matrix/longmemeval_operation_value_summary.csv",
        ],
    )
    reward_mdp = find_file(
        run_dir,
        [
            "mdp_evidence_analysis_grow_q/p14_mdp_evidence_reward_summary.json",
            "mdp_evidence_analysis_with_expand_q_v2/p14_mdp_evidence_reward_summary.json",
            "mdp_evidence_analysis_with_expand_q/p14_mdp_evidence_reward_summary.json",
            "mdp_evidence_analysis/p14_mdp_evidence_reward_summary.json",
        ],
    )
    utility_mdp = find_file(
        run_dir,
        [
            "mdp_evidence_analysis_grow_q/p14_mdp_evidence_utility_summary.json",
            "mdp_evidence_analysis_with_expand_q_v2/p14_mdp_evidence_utility_summary.json",
            "mdp_evidence_analysis_with_expand_q/p14_mdp_evidence_utility_summary.json",
            "mdp_evidence_analysis/p14_mdp_evidence_utility_summary.json",
        ],
    )
    if matrix_path is None:
        raise FileNotFoundError(f"No matrix summary found under {run_dir}")

    matrix_rows = [row for row in read_csv(matrix_path) if row.get("group") == "ALL"]
    by_op = {row["operation"]: row for row in matrix_rows}
    reward = read_json(reward_mdp) if reward_mdp else {}
    utility = read_json(utility_mdp) if utility_mdp else {}
    rows = []
    for operation in sorted(by_op):
        row = by_op[operation]
        rows.append(
            {
                "run_label": label,
                "run_dir": str(run_dir),
                "operation": operation,
                "n": row.get("n"),
                "mean_reward": row.get("mean_reward"),
                "mean_cost": row.get("mean_cost"),
                "mean_utility": row.get("mean_utility"),
                "delta_reward_vs_stop": row.get("mean_delta_reward_vs_baseline"),
                "delta_utility_vs_stop": row.get("mean_delta_utility_vs_baseline"),
                "evidence_recall": row.get("mean_evidence_session_recall"),
                "all_evidence_present": row.get("mean_all_evidence_present"),
                "trace_coverage": row.get("trace_coverage"),
                "reward_mean_abs_path_shift": fget(reward, ["path_dependent_marginal_value", "any_abs_path_shift", "mean"]),
                "reward_p_abs_path_shift_gt_0.05": fget(
                    reward,
                    ["bootstrap_uncertainty", "proportion_abs_path_shift_gt_0.05_ci", "proportion"],
                ),
                "reward_gain_b2_minus_b1_mean": fget(reward, ["budget_redesign", "gain_B2_minus_B1", "mean"]),
                "reward_gain_b2_minus_b1_n_positive": fget(reward, ["budget_redesign", "gain_B2_minus_B1", "n_positive"]),
                "reward_composition_gain_gt_0.1_prop": fget(
                    reward,
                    ["bootstrap_uncertainty", "proportion_composition_gain_gt_0.10_ci", "proportion"],
                ),
                "utility_mean_abs_path_shift": fget(utility, ["path_dependent_marginal_value", "any_abs_path_shift", "mean"]),
                "utility_gain_b2_minus_b1_mean": fget(utility, ["budget_redesign", "gain_B2_minus_B1", "mean"]),
            }
        )
    return rows


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, Path(path)
    path = Path(spec)
    return path.name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="label=/path/to/run_dir, repeated")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for spec in args.run:
        label, run_dir = parse_run_spec(spec)
        rows.extend(summarize_run(run_dir, label))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "n_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

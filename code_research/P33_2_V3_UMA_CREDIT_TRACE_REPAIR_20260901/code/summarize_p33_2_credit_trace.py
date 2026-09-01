#!/usr/bin/env python3
"""Summarize P33.2 UMA credit-trace JSONL into audit tables.

This is an analysis-only script.  It expects logs produced with
``UMA_CREDIT_TRACE=1`` and does not call any model, judge, trainer, or evaluator.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pvariance
from typing import Any


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def resolve_trace_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        expanded = [Path(p) for p in glob.glob(item)] or [Path(item)]
        for path in expanded:
            if path.is_dir():
                paths.extend(sorted(path.glob("*.jsonl")))
            else:
                paths.append(path)
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise FileNotFoundError(f"No trace JSONL files resolved from: {inputs}")
    return unique


def read_events(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event["_trace_file"] = str(path)
                    rows.append(event)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} on line {line_no}: {exc}") from exc
    return rows


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ols_r2(y: list[float], columns: list[list[float]]) -> float | None:
    if len(y) < 3:
        return None
    try:
        import numpy as np

        y_arr = np.asarray(y, dtype=float)
        x_arr = np.asarray([[1.0] + [col[i] for col in columns] for i in range(len(y))], dtype=float)
        beta, *_ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
        pred = x_arr @ beta
        ss_res = float(((y_arr - pred) ** 2).sum())
        ss_tot = float(((y_arr - y_arr.mean()) ** 2).sum())
        if ss_tot <= 0:
            return None
        return 1.0 - ss_res / ss_tot
    except Exception:
        return None


def summarize(trace_inputs: list[str], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_paths = resolve_trace_paths(trace_inputs)
    events = read_events(trace_paths)
    adv_rows = [e.get("payload", {}) for e in events if e.get("event") == "p33_2_grpo_advantage_row"]
    reward_rows = [e.get("payload", {}) for e in events if e.get("event") == "p33_2_reward_postprocess"]
    ray_completion_rows = [e.get("payload", {}) for e in events if e.get("event") == "p33_2_ray_task_completed"]
    ray_reorder_rows = [e.get("payload", {}) for e in events if e.get("event") == "p33_2_ray_reorder_precheck"]

    table_rows: list[dict[str, Any]] = []
    for row in adv_rows:
        table_rows.append({
            "uid": to_str(row.get("uid")),
            "trajectory_id": to_str(row.get("trajectory_id")),
            "trajectory_key": to_str(row.get("trajectory_key")),
            "trajectory_step": to_str(row.get("trajectory_step")),
            "sample_index": to_str(row.get("sample_index")),
            "rollout_n": to_str(row.get("rollout_n")),
            "validate": to_str(row.get("validate")),
            "batch_row_index": to_str(row.get("batch_row_index")),
            "original_index": to_str(row.get("original_index")),
            "is_final": to_str(row.get("is_final")),
            "conversation_index": to_str(row.get("conversation_index")),
            "memory_step_index": to_str(row.get("memory_step_index")),
            "final_query_index": to_str(row.get("final_query_index")),
            "data_source": to_str(row.get("data_source")),
            "agent_name": to_str(row.get("agent_name")),
            "grpo_group": to_str(row.get("grpo_group")),
            "row_index": to_str(row.get("row_index")),
            "qa_component": to_float(row.get("qa_outcome_component")),
            "tool_reward": to_float(row.get("tool_reward")),
            "total_reward": to_float(row.get("reward_score")),
            "raw_score": to_float(row.get("raw_score")),
            "grpo_group_mean": to_float(row.get("grpo_group_mean")),
            "grpo_group_std": to_float(row.get("grpo_group_std")),
            "grpo_group_n": to_str(row.get("grpo_group_n")),
            "advantage": to_float(row.get("advantage_scalar")),
            "advantage_unique_values_on_generated_tokens": to_str(row.get("advantage_unique_values_on_generated_tokens")),
            "response_mask_tokens": to_str(row.get("response_mask_tokens")),
            "num_tools": to_str(row.get("num_tools")),
            "tool_counts": json.dumps(row.get("tool_counts"), ensure_ascii=False, sort_keys=True),
        })

    identity_violations = {
        "missing_trajectory_key": sum(1 for r in table_rows if not r["trajectory_key"]),
        "missing_conversation_index": sum(1 for r in table_rows if not r["conversation_index"]),
        "memory_rows_missing_memory_step_index": sum(
            1
            for r in table_rows
            if r.get("is_final") in {"0", "False", "false"} and not r["memory_step_index"]
        ),
    }
    if any(identity_violations.values()):
        raise ValueError(
            "P33.2 identity contract violation: "
            + json.dumps(identity_violations, ensure_ascii=False, sort_keys=True)
        )

    by_traj_phase: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in table_rows:
        by_traj_phase[(row["trajectory_key"], row["is_final"])].append(row)
    for rows in by_traj_phase.values():
        rows.sort(key=lambda r: int(r["conversation_index"]) if str(r["conversation_index"]).isdigit() else 10**9)

    by_grpo_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table_rows:
        by_grpo_group[row["grpo_group"]].append(row)
    for rows in by_grpo_group.values():
        rows.sort(
            key=lambda r: (
                float("inf") if r["raw_score"] is None else -r["raw_score"],
                int(r["row_index"]) if str(r["row_index"]).isdigit() else 10**9,
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["reward_rank_within_grpo_group"] = rank

    memory_by_traj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table_rows:
        if row.get("is_final") in {"0", "False", "false"}:
            memory_by_traj[row["trajectory_key"]].append(row)
    for rows in memory_by_traj.values():
        adv_vals = [r["advantage"] for r in rows if r["advantage"] is not None]
        traj_mean_adv = mean(adv_vals) if adv_vals else None
        for row in rows:
            row["trajectory_mean_memory_advantage"] = traj_mean_adv
    for row in table_rows:
        row.setdefault("trajectory_mean_memory_advantage", None)
        row.setdefault("reward_rank_within_grpo_group", "")

    csv_path = output_dir / "p33_2_credit_trace_table.csv"
    fieldnames = [
        "uid", "trajectory_key", "trajectory_id", "trajectory_step", "sample_index", "rollout_n", "validate",
        "batch_row_index", "original_index", "is_final", "conversation_index", "memory_step_index",
        "final_query_index", "trajectory_mean_memory_advantage", "reward_rank_within_grpo_group",
        "data_source", "agent_name", "grpo_group", "row_index", "qa_component", "tool_reward",
        "total_reward", "raw_score", "grpo_group_mean", "grpo_group_std", "grpo_group_n", "advantage",
        "advantage_unique_values_on_generated_tokens", "response_mask_tokens", "num_tools", "tool_counts",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    memory_rows = [r for r in table_rows if r.get("is_final") in {"0", "False", "false"}]
    qa_vars = []
    adv_vars = []
    trajectory_distribution = []
    for (trajectory_key, is_final), rows in sorted(by_traj_phase.items()):
        if is_final not in {"0", "False", "false"}:
            continue
        qa_vals = [r["qa_component"] for r in rows if r["qa_component"] is not None]
        adv_vals = [r["advantage"] for r in rows if r["advantage"] is not None]
        qa_var = pvariance(qa_vals) if len(qa_vals) >= 2 else 0.0 if len(qa_vals) == 1 else None
        adv_var = pvariance(adv_vals) if len(adv_vals) >= 2 else 0.0 if len(adv_vals) == 1 else None
        if qa_var is not None:
            qa_vars.append(qa_var)
        if adv_var is not None:
            adv_vars.append(adv_var)
        trajectory_distribution.append({
            "trajectory_key": trajectory_key,
            "trajectory_id": rows[0].get("trajectory_id") if rows else "",
            "n_memory_rows": len(rows),
            "qa_component_variance": qa_var,
            "advantage_variance": adv_var,
            "trajectory_mean_memory_advantage": rows[0].get("trajectory_mean_memory_advantage") if rows else None,
            "tool_reward_values": [r["tool_reward"] for r in rows],
            "advantage_values": [r["advantage"] for r in rows],
        })

    paired = [
        (r["tool_reward"], r["advantage"], r["qa_component"])
        for r in memory_rows
        if r["tool_reward"] is not None and r["advantage"] is not None and r["qa_component"] is not None
    ]
    tool_vals = [x[0] for x in paired]
    adv_vals = [x[1] for x in paired]
    qa_vals = [x[2] for x in paired]
    r2_qa = ols_r2(adv_vals, [qa_vals]) if paired else None
    r2_tool = ols_r2(adv_vals, [tool_vals]) if paired else None
    r2_both = ols_r2(adv_vals, [qa_vals, tool_vals]) if paired else None

    ray_completion_positions = [r.get("completion_position") for r in ray_completion_rows]
    ray_submission_indices = [r.get("submission_index") for r in ray_completion_rows]
    ray_order_matches_submission = (
        ray_completion_positions == ray_submission_indices if ray_completion_rows else None
    )

    summary = {
        "status": "P33_2_TRACE_SUMMARY_COMPLETE",
        "trace_inputs": trace_inputs,
        "trace_paths": [str(p) for p in trace_paths],
        "identity_contract": {
            "status": "PASS",
            "violations": identity_violations,
            "required_fields": ["trajectory_key", "conversation_index"],
            "memory_required_fields": ["memory_step_index"],
            "trajectory_key_source": "p33_2 derived from official get_trajectory_info(step, sample_index, rollout_n, validate)",
            "chunk_index_source": "p33_2_conversation_index / p33_2_memory_step_index from tool_mem_agent_loop output construction",
        },
        "n_events": len(events),
        "n_reward_postprocess_rows": len(reward_rows),
        "n_grpo_advantage_rows": len(adv_rows),
        "n_memory_rows": len(memory_rows),
        "n_grpo_groups": len(set(r.get("grpo_group") for r in table_rows)),
        "is_final_counts": dict(Counter(r.get("is_final") for r in table_rows)),
        "within_trajectory_qa_variance_mean": mean(qa_vars) if qa_vars else None,
        "within_trajectory_advantage_variance_mean": mean(adv_vars) if adv_vars else None,
        "corr_tool_reward_advantage_memory": pearson(tool_vals, adv_vals),
        "corr_qa_component_advantage_memory": pearson(qa_vals, adv_vals),
        "advantage_variance_explained_r2": {
            "qa_component_only": r2_qa,
            "tool_reward_only": r2_tool,
            "qa_plus_tool": r2_both,
        },
        "ray_completion_order": {
            "n_completion_rows": len(ray_completion_rows),
            "n_reorder_precheck_rows": len(ray_reorder_rows),
            "completion_positions": ray_completion_positions,
            "submission_indices_in_completion_order": ray_submission_indices,
            "completion_order_matches_submission_order": ray_order_matches_submission,
            "reorder_precheck_rows": ray_reorder_rows,
            "status": "NO_RAY_COMPLETION_EVENTS" if not ray_completion_rows else "RECORDED",
        },
        "trajectory_distribution": trajectory_distribution,
        "outputs": {
            "table_csv": str(csv_path),
        },
        "interpretation_guardrails": [
            "This summarizes real trace logs only if the input JSONL came from an actual UMA rollout.",
            "It does not judge semantic memory correctness.",
            "It does not establish RL necessity.",
            "Intervention-based credit reference is still required before method design.",
        ],
    }
    summary_path = output_dir / "p33_2_credit_trace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = output_dir / "p33_2_credit_trace_summary.md"
    md_path.write_text(
        "\n".join([
            "# P33.2 UMA Credit Trace Summary",
            "",
            f"Trace inputs: `{trace_inputs}`",
            f"Resolved trace files: `{[str(p) for p in trace_paths]}`",
            f"Identity contract: `{summary['identity_contract']}`",
            f"Rows: reward_postprocess={len(reward_rows)}, grpo_advantage={len(adv_rows)}, memory={len(memory_rows)}",
            f"Mean within-trajectory QA variance: `{summary['within_trajectory_qa_variance_mean']}`",
            f"Mean within-trajectory advantage variance: `{summary['within_trajectory_advantage_variance_mean']}`",
            f"Corr(tool_reward, advantage) on memory rows: `{summary['corr_tool_reward_advantage_memory']}`",
            f"Corr(QA component, advantage) on memory rows: `{summary['corr_qa_component_advantage_memory']}`",
            f"R2 QA/tool/both: `{summary['advantage_variance_explained_r2']}`",
            f"Ray order status: `{summary['ray_completion_order']}`",
            "",
            "Boundary: instrumentation/anatomy only; no semantic correctness judgment and no RL necessity claim.",
        ]) + "\n",
        encoding="utf-8",
    )
    summary["outputs"]["summary_json"] = str(summary_path)
    summary["outputs"]["summary_md"] = str(md_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-jsonl", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = summarize(args.trace_jsonl, Path(args.output_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

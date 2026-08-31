#!/usr/bin/env python3
"""P25.5 mechanism audit for ANSWER-vs-SHRINK.

This is an offline, non-training analysis.  It tests whether SHRINK gains are
associated with integration-demand proxies and whether SHRINK losses are
associated with preservation-risk proxies.  Gold evidence labels and benchmark
question_type are used only after frozen generation for mechanism diagnosis.
They are not legal online controller features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_qid(qid: str) -> str:
    qid = str(qid)
    return qid if qid.startswith("longmemeval_") else f"longmemeval_{qid}"


def raw_qid(qid: str) -> str:
    qid = str(qid)
    return qid[len("longmemeval_") :] if qid.startswith("longmemeval_") else qid


def parse_float(value: Any) -> float:
    if value in {None, "", "nan", "NaN"}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_int(value: Any) -> int | None:
    if value in {None, "", "nan", "NaN"}:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def parse_date(text: Any) -> datetime | None:
    if text is None:
        return None
    candidates = re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", str(text))
    for candidate in candidates:
        try:
            return datetime.strptime(candidate.replace("-", "/"), "%Y/%m/%d")
        except ValueError:
            continue
    return None


def bin_count(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3plus"


def bin_span(span: float) -> str:
    if math.isnan(span):
        return "missing"
    if span <= 1:
        return "span1"
    if span <= 5:
        return "span2to5"
    if span <= 20:
        return "span6to20"
    return "span_gt20"


def bin_days(days: float) -> str:
    if math.isnan(days):
        return "missing"
    if days == 0:
        return "same_time"
    if days <= 1:
        return "within_day"
    if days <= 7:
        return "within_week"
    if days <= 31:
        return "within_month"
    return "over_month"


def load_raw_features(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("raw LongMemEval file must be a JSON list")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = normalize_qid(str(row.get("question_id") or ""))
        if not qid:
            continue
        session_ids = [str(x) for x in row.get("haystack_session_ids") or []]
        gold = [str(x) for x in row.get("answer_session_ids") or []]
        gold_positions = [session_ids.index(x) for x in gold if x in session_ids]
        dates = row.get("haystack_dates") or []
        gold_dates = [parse_date(dates[pos]) for pos in gold_positions if pos < len(dates)]
        gold_dates = [dt for dt in gold_dates if dt is not None]
        qdate = parse_date(row.get("question_date"))
        if gold_positions:
            span = max(gold_positions) - min(gold_positions) + 1
            density = len(gold_positions) / max(1, span)
        else:
            span = math.nan
            density = math.nan
        if gold_dates:
            temporal_spread = (max(gold_dates) - min(gold_dates)).days
            latest_age = (qdate - max(gold_dates)).days if qdate is not None else math.nan
        else:
            temporal_spread = math.nan
            latest_age = math.nan
        out[qid] = {
            "raw_qid": raw_qid(qid),
            "question_type": row.get("question_type") or "unknown",
            "gold_relevant_session_count": len(gold),
            "gold_present_in_haystack_count": len(gold_positions),
            "haystack_session_count": len(session_ids),
            "gold_position_span": span,
            "gold_position_density": density,
            "gold_temporal_spread_days": temporal_spread,
            "latest_gold_age_days": latest_age,
            "gold_relevant_count_bin": bin_count(len(gold)),
            "gold_position_span_bin": bin_span(span),
            "gold_temporal_spread_bin": bin_days(temporal_spread),
            "gold_latest_age_bin": bin_days(latest_age),
        }
    return out


def load_model_table(spec: str) -> tuple[str, Path, list[dict[str, Any]]]:
    if "=" not in spec:
        raise ValueError("--model-table must be label=path")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"Empty model table: {path}")
    return label, path, rows


def wtl(delta: float, eps: float) -> str:
    if math.isnan(delta):
        return "missing"
    if delta > eps:
        return "win"
    if delta < -eps:
        return "loss"
    return "tie"


def safe_mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return math.nan
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    return num / (dx * dy) if dx > 0 and dy > 0 else math.nan


def ranks(values: list[float]) -> list[float]:
    valid = sorted((v, i) for i, v in enumerate(values) if not math.isnan(v))
    out = [math.nan] * len(values)
    pos = 0
    while pos < len(valid):
        end = pos + 1
        while end < len(valid) and valid[end][0] == valid[pos][0]:
            end += 1
        rank = (pos + 1 + end) / 2
        for _, idx in valid[pos:end]:
            out[idx] = rank
        pos = end
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def summarize_group(
    rows: list[dict[str, Any]],
    model_label: str,
    group_name: str,
    group_value: str,
    eps: float,
) -> dict[str, Any]:
    deltas = [parse_float(row["delta_reward"]) for row in rows]
    signs = Counter(wtl(d, eps) for d in deltas)
    return {
        "model_label": model_label,
        "group_name": group_name,
        "group_value": group_value,
        "n": len(rows),
        "mean_delta_reward": safe_mean(deltas),
        "win": signs.get("win", 0),
        "tie": signs.get("tie", 0),
        "loss": signs.get("loss", 0),
        "win_rate": signs.get("win", 0) / max(1, len(rows)),
        "loss_rate": signs.get("loss", 0) / max(1, len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-longmemeval", type=Path, required=True)
    parser.add_argument("--model-table", action="append", required=True, help="label=per_qid_csv")
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_features = load_raw_features(args.raw_longmemeval)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for label, path, rows in map(load_model_table, args.model_table):
        sources[label] = str(path)
        for row in rows:
            qid = normalize_qid(row.get("qid", ""))
            if qid not in raw_features:
                raise ValueError(f"{label}: qid not found in raw LongMemEval: {qid}")
            enriched = {**raw_features[qid], **row}
            enriched["qid"] = qid
            enriched["model_label"] = label
            enriched["delta_reward"] = parse_float(row.get("delta_reward"))
            enriched["delta_utility"] = parse_float(row.get("delta_utility"))
            enriched["primary_wtl"] = row.get("primary_wtl") or wtl(enriched["delta_reward"], args.eps)

            gold_count = parse_int(enriched.get("gold_relevant_session_count")) or 0
            span = parse_float(enriched.get("gold_position_span"))
            temporal_spread = parse_float(enriched.get("gold_temporal_spread_days"))
            qtype = str(enriched.get("question_type") or "unknown")

            integration_score = 0
            integration_score += int(gold_count >= 2)
            integration_score += int(not math.isnan(span) and span > 5)
            integration_score += int(not math.isnan(temporal_spread) and temporal_spread > 31)
            integration_score += int(qtype in {"knowledge-update", "multi-session", "temporal-reasoning"})

            preservation_risk_score = 0
            preservation_risk_score += int(gold_count == 1)
            preservation_risk_score += int(not math.isnan(span) and span <= 1)
            preservation_risk_score += int(not math.isnan(temporal_spread) and temporal_spread == 0)
            preservation_risk_score += int(qtype.startswith("single-session"))

            enriched["offline_integration_score"] = integration_score
            enriched["offline_preservation_risk_score"] = preservation_risk_score
            enriched["offline_integration_bin"] = "high" if integration_score >= 3 else "mid" if integration_score >= 2 else "low"
            enriched["offline_preservation_risk_bin"] = (
                "high" if preservation_risk_score >= 3 else "mid" if preservation_risk_score >= 2 else "low"
            )
            all_rows.append(enriched)

    group_rows: list[dict[str, Any]] = []
    group_keys = [
        "question_type",
        "gold_relevant_count_bin",
        "gold_position_span_bin",
        "gold_temporal_spread_bin",
        "gold_latest_age_bin",
        "offline_integration_bin",
        "offline_preservation_risk_bin",
        "canonical_initial_complete_e0_bin",
        "shrink_all_evidence_present",
    ]
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_model[str(row["model_label"])].append(row)
    for model_label, rows in sorted(by_model.items()):
        group_rows.append(summarize_group(rows, model_label, "ALL", "ALL", args.eps))
        for key in group_keys:
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                value = row.get(key)
                if value in {None, "", "nan", "NaN"}:
                    continue
                buckets[str(value)].append(row)
            for value, bucket in sorted(buckets.items()):
                group_rows.append(summarize_group(bucket, model_label, key, value, args.eps))

    numeric_rows: list[dict[str, Any]] = []
    for model_label, rows in sorted(by_model.items()):
        y = [parse_float(row["delta_reward"]) for row in rows]
        for key in [
            "gold_relevant_session_count",
            "gold_position_span",
            "gold_temporal_spread_days",
            "latest_gold_age_days",
            "offline_integration_score",
            "offline_preservation_risk_score",
        ]:
            x = [parse_float(row.get(key)) for row in rows]
            numeric_rows.append(
                {
                    "model_label": model_label,
                    "feature": key,
                    "n": sum(1 for a, b in zip(x, y) if not math.isnan(a) and not math.isnan(b)),
                    "pearson_with_delta_reward": pearson(x, y),
                    "spearman_with_delta_reward": spearman(x, y),
                }
            )

    preservation_rows: list[dict[str, Any]] = []
    for model_label, rows in sorted(by_model.items()):
        rows_with_preservation = [
            row
            for row in rows
            if str(row.get("shrink_all_evidence_present", "")).strip() not in {"", "nan", "NaN"}
        ]
        if not rows_with_preservation:
            continue
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows_with_preservation:
            buckets[str(row.get("shrink_all_evidence_present"))].append(row)
        for value, bucket in sorted(buckets.items()):
            preservation_rows.append(summarize_group(bucket, model_label, "shrink_all_evidence_present", value, args.eps))

    write_csv(args.output_dir / "p25_5_mechanism_per_qid.csv", all_rows)
    write_csv(args.output_dir / "p25_5_mechanism_group_summary.csv", group_rows)
    write_csv(args.output_dir / "p25_5_mechanism_numeric_associations.csv", numeric_rows)
    write_csv(args.output_dir / "p25_5_preservation_audit.csv", preservation_rows)

    audit = {
        "analysis": "P25.5 offline mechanism audit; no training and no generation.",
        "raw_longmemeval": str(args.raw_longmemeval),
        "model_tables": sources,
        "n_rows": len(all_rows),
        "n_by_model": {label: len(rows) for label, rows in sorted(by_model.items())},
        "eps": args.eps,
        "outputs": [
            "p25_5_mechanism_per_qid.csv",
            "p25_5_mechanism_group_summary.csv",
            "p25_5_mechanism_numeric_associations.csv",
            "p25_5_preservation_audit.csv",
        ],
        "leakage_boundary": (
            "question_type and gold evidence fields are privileged/offline mechanism diagnostics only; "
            "they are not legal online policy features."
        ),
        "legacy_trace_warning": (
            "Rows whose initial_index_source is selected_indices_legacy_fallback cannot be used as canonical "
            "Complete(E0). They are retained only to diagnose old traces."
        ),
        "claim_boundary": (
            "This audit can support exploratory mechanism hypotheses about integration benefit and preservation risk. "
            "It cannot establish official LongMemEval performance or RL necessity."
        ),
    }
    (args.output_dir / "p25_5_mechanism_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired, query-unknown memory headroom test on official Mem2ActBench.

The memory writer sees a historical session only.  The executor sees the frozen
memory, future query, and official target tool schema.  Gold calls and benchmark
evolution chains are used for scoring only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


MEMORY_PROMPTS = {
    "summary": """Compress the historical interaction into a query-agnostic factual memory.
Preserve concrete entities, identifiers, preferences, dates, constraints, and important outcomes.
Resolve recency conflicts when possible. Do not predict a future question and do not add facts.
Return only the memory, within {budget} tokens.""",
    "state": """Build a query-agnostic CURRENT STATE memory from the historical interaction.
Represent entities and their latest values, user preferences, active goals, unresolved variables,
temporal supersession/conflicts, and uncertainty/provenance. Preserve exact values needed later.
Do not organize around actions or speculate about a future question. Return only the memory,
within {budget} tokens.""",
    "control": """Build a query-agnostic CONTROL-SUFFICIENT memory from the historical interaction.
The memory should preserve only information needed to choose and parameterize future actions.
Use compact sections when supported by evidence:
CURRENT STATE: latent variables, entities, exact values, preferences, active goals.
ACTION PRECONDITIONS: facts/constraints that determine which action is valid.
TRANSITIONS: observed (state, action/tool) -> next state/tool outcome relationships.
SUCCESS/UTILITY: what counted as completion or user satisfaction.
FAILURES/GOTCHAS: failed actions, stale values, conflicts, uncertainty, provenance.
Do not predict a future question and do not invent causal relations. Return only the memory,
within {budget} tokens.""",
}

LEDGER_PROMPT = """Extract a query-agnostic GROUNDED STATE LEDGER from the historical session.
The ledger is a machine-facing memory, not a prose summary. Preserve every concrete identifier,
address, ID, number, date, location, preference, constraint, and current goal that could bind a
future tool argument. NEVER replace a concrete value with a placeholder such as user_id,
user_address, same_id, or an invented example. Track newer versus superseded values and retain
source_id provenance. Use compact JSON Lines, one object per fact, with keys:
key, value, value_type, source_id, temporal_order, status, supersedes.
Use status=current|superseded|uncertain. Do not see or predict a future query. Do not invent facts.
Return only JSON Lines within {budget} tokens."""

DYNAMICS_PROMPT = """Extract query-agnostic ACTION-CONDITIONED DYNAMICS from the historical session.
The grounded ledger below is immutable: do not rewrite, summarize, or replace its values.
Record only relationships explicitly supported by observed turns/tool calls:
1) action parameter -> grounded ledger key bindings;
2) action preconditions;
3) observed (pre-state, action) -> (post-state, outcome) transitions;
4) failures, recovery rules, and success criteria.
Use compact JSON Lines. Every line must include kind=binding|transition|failure|success and evidence
source_id(s). If the history contains no supported dynamics, return an empty string. Never invent
causal structure and never predict a future query. Return only JSON Lines within {budget} tokens."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return value


def equal_value(predicted: Any, gold: Any) -> bool:
    if canonical(predicted) == canonical(gold):
        return True
    # Tool arguments sometimes differ only by JSON/string representation.
    if isinstance(predicted, str):
        try:
            return canonical(json.loads(predicted)) == canonical(gold)
        except (ValueError, TypeError):
            pass
    return False


def chat(base_url: str, model: str, messages: list[dict[str, str]], max_tokens: int,
         temperature: float, seed: int, timeout: int, retries: int) -> tuple[str, float]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }).encode()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        start = time.monotonic()
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read())
            return payload["choices"][0]["message"]["content"], time.monotonic() - start
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"vLLM request failed after {retries + 1} attempts: {last_error}")


def render_turns(session: dict[str, Any], max_chars: int) -> str:
    rows = []
    for turn in session["turns"]:
        role = turn.get("role", "unknown").upper()
        source = turn.get("source_id", "")
        content = turn.get("content", "")
        calls = turn.get("tool_calls") or []
        suffix = ""
        if calls:
            suffix = "\nTOOL_CALLS: " + json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
        rows.append(f"[{role} source={source}]\n{content}{suffix}")
    text = "\n\n".join(rows)
    if len(text) <= max_chars:
        return text
    # Keep both early setup and most recent state; mark the omitted middle explicitly.
    half = max_chars // 2
    return text[:half] + "\n\n[... MIDDLE TRUNCATED ...]\n\n" + text[-half:]


def extract_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object in response")


def bootstrap_ci(values: list[float], seed: int, samples: int = 10000) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def metric_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, float]:
    vals = [r["conditions"][condition] for r in rows]
    return {
        "n": len(vals),
        "json_valid": statistics.fmean(v["json_valid"] for v in vals),
        "tool_name_accuracy": statistics.fmean(v["tool_name_correct"] for v in vals),
        "argument_accuracy": statistics.fmean(v["argument_accuracy"] for v in vals),
        "gold_argument_surface_recall": statistics.fmean(v["gold_argument_surface_recall"] for v in vals),
        "exact_tool_call_accuracy": statistics.fmean(v["exact"] for v in vals),
        "mean_context_chars": statistics.fmean(v["context_chars"] for v in vals),
        "mean_latency_s": statistics.fmean(v["latency_s"] for v in vals),
    }


def build_summary(rows: list[dict[str, Any]], conditions: list[str], seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"num_examples": len(rows), "conditions": {}, "paired": {}, "by_level": {}}
    for condition in conditions:
        out["conditions"][condition] = metric_summary(rows, condition)
    pairs = [("control", "state"), ("control", "summary"),
             ("control", "full_history"), ("state", "summary")]
    pairs += [("control_nested", "ledger384"), ("control_nested", "ledger512"),
              ("control_nested", "summary"), ("ledger512", "summary")]
    for left, right in pairs:
        if left not in conditions or right not in conditions:
            continue
        diffs = [r["conditions"][left]["exact"] - r["conditions"][right]["exact"] for r in rows]
        out["paired"][f"{left}_minus_{right}"] = {
            "exact_delta": statistics.fmean(diffs),
            "bootstrap_95_ci": bootstrap_ci(diffs, seed + len(out["paired"])),
            "left_wins": sum(x > 0 for x in diffs),
            "right_wins": sum(x < 0 for x in diffs),
            "ties": sum(x == 0 for x in diffs),
        }
    for level in sorted({r["level"] for r in rows}):
        subset = [r for r in rows if r["level"] == level]
        out["by_level"][level] = {
            c: metric_summary(subset, c)["exact_tool_call_accuracy"] for c in conditions
        } | {"n": len(subset)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-jsonl", required=True, type=Path)
    parser.add_argument("--conversation-jsonl", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--served-model", default="qwen25-7b")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--conditions", default="no_memory,full_history,summary,state,control")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--levels", default="L1,L2,L3,L4")
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--ledger-base-tokens", type=int, default=384)
    parser.add_argument("--dynamics-tokens", type=int, default=128)
    parser.add_argument("--answer-tokens", type=int, default=256)
    parser.add_argument("--max-history-chars", type=int, default=32000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    allowed = {"no_memory", "full_history", "ledger384", "ledger512", "control_nested", *MEMORY_PROMPTS}
    unknown = set(conditions) - allowed
    if unknown:
        parser.error(f"unknown conditions: {sorted(unknown)}")
    levels = {x.strip() for x in args.levels.split(",") if x.strip()}
    qas = [q for q in read_jsonl(args.qa_jsonl) if q["complexity_metadata"]["level"] in levels]
    qas = qas[args.start:args.start + args.num_samples]
    sessions = read_jsonl(args.conversation_jsonl)
    source_to_session = {}
    session_by_id = {}
    for session in sessions:
        session_by_id[session["session_id"]] = session
        for source_id in session["original_conversation_ids"]:
            source_to_session[source_id] = session["session_id"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for row in read_jsonl(args.output):
            completed[row["qa_id"]] = row

    memory_cache: dict[tuple[str, str], str] = {}
    cache_guard = threading.Lock()
    key_locks: dict[tuple[str, str], threading.Lock] = {}

    def get_memory(session: dict[str, Any], mode: str, budget: int | None = None,
                   immutable_ledger: str = "") -> str:
        actual_budget = budget or args.memory_tokens
        extra_digest = hashlib.sha256(immutable_ledger.encode()).hexdigest() if immutable_ledger else ""
        key = (session["session_id"], f"{mode}:{actual_budget}:{extra_digest}")
        with cache_guard:
            if key in memory_cache:
                return memory_cache[key]
            key_lock = key_locks.setdefault(key, threading.Lock())
        # Two concurrent QAs may reference the same official session. Compile it
        # once and avoid racing on the on-disk cache file.
        with key_lock:
            with cache_guard:
                if key in memory_cache:
                    return memory_cache[key]
            history = render_turns(session, args.max_history_chars)
            digest = hashlib.sha256(json.dumps({
                "v": 2, "model": args.served_model, "mode": mode, "budget": actual_budget,
                "history": history, "immutable_ledger": immutable_ledger,
            }, sort_keys=True).encode()).hexdigest()
            path = args.cache_dir / f"{digest}.json"
            if path.exists():
                memory = json.loads(path.read_text())["memory"]
            else:
                if mode == "ledger":
                    instruction = LEDGER_PROMPT.format(budget=actual_budget)
                elif mode == "dynamics":
                    instruction = DYNAMICS_PROMPT.format(budget=actual_budget)
                else:
                    instruction = MEMORY_PROMPTS[mode].format(budget=actual_budget)
                writer_input = "HISTORICAL SESSION (future query unavailable):\n" + history
                if immutable_ledger:
                    writer_input += "\n\nIMMUTABLE GROUNDED LEDGER:\n" + immutable_ledger
                memory, latency = chat(
                    args.base_url, args.served_model,
                    [{"role": "system", "content": instruction},
                     {"role": "user", "content": writer_input}],
                    actual_budget, args.temperature, args.seed, args.timeout, args.retries,
                )
                path.write_text(json.dumps({"memory": memory, "latency_s": latency}, ensure_ascii=False, indent=2))
            with cache_guard:
                memory_cache[key] = memory
            return memory

    def run_qa(index_and_qa: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, qa = index_and_qa
        selected_ids = []
        for source in qa["source_conversation_ids"]:
            sid = source_to_session.get(source)
            if sid and sid not in selected_ids:
                selected_ids.append(sid)
        selected = [session_by_id[sid] for sid in selected_ids]
        histories = [render_turns(s, args.max_history_chars) for s in selected]
        memories = {mode: [get_memory(s, mode) for s in selected] for mode in MEMORY_PROMPTS if mode in conditions}
        needs_nested = any(c in conditions for c in ("ledger384", "ledger512", "control_nested"))
        ledgers384 = [get_memory(s, "ledger", args.ledger_base_tokens) for s in selected] if needs_nested else []
        ledgers512 = [get_memory(s, "ledger", args.memory_tokens) for s in selected] if "ledger512" in conditions else []
        dynamics = ([get_memory(s, "dynamics", args.dynamics_tokens, ledger)
                     for s, ledger in zip(selected, ledgers384)] if "control_nested" in conditions else [])
        gold = qa["tool_call"]
        row = {
            "qa_id": qa["qa_id"], "index": index, "level": qa["complexity_metadata"]["level"],
            "session_ids": selected_ids, "query": qa["query"], "gold": {"name": gold["name"], "arguments": gold["arguments"]},
            "conditions": {},
        }
        for c_index, condition in enumerate(conditions):
            if condition == "no_memory":
                context = "(No historical memory supplied.)"
            elif condition == "full_history":
                context = "\n\n===== NEXT SESSION =====\n\n".join(histories)
            elif condition == "ledger384":
                context = "\n\n===== NEXT SESSION LEDGER =====\n\n".join(ledgers384)
            elif condition == "ledger512":
                context = "\n\n===== NEXT SESSION LEDGER =====\n\n".join(ledgers512)
            elif condition == "control_nested":
                parts = []
                for ledger, dyn in zip(ledgers384, dynamics):
                    parts.append("GROUNDED LEDGER (immutable):\n" + ledger +
                                 "\n\nACTION-CONDITIONED DYNAMICS (additive):\n" + (dyn or "(none observed)"))
                context = "\n\n===== NEXT SESSION CONTROL MEMORY =====\n\n".join(parts)
            else:
                context = "\n\n===== NEXT SESSION MEMORY =====\n\n".join(memories[condition])
            system = """You select a tool call using the supplied past context and current request.
Return exactly one JSON object with keys name and arguments. Do not add prose or markdown.
Use only the target tool schema. Preserve exact argument types."""
            user = ("PAST CONTEXT:\n" + context + "\n\nCURRENT REQUEST:\n" + qa["query"] +
                    "\n\nTARGET TOOL SCHEMA:\n" + json.dumps(qa["target_tool_schema"], ensure_ascii=False))
            raw, latency = chat(args.base_url, args.served_model,
                                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                args.answer_tokens, args.temperature, args.seed + index * 101 + c_index,
                                args.timeout, args.retries)
            try:
                pred = extract_object(raw)
                valid = 1.0
            except ValueError:
                pred, valid = {}, 0.0
            pred_args = pred.get("arguments") if isinstance(pred.get("arguments"), dict) else {}
            expected = gold["arguments"]
            arg_hits = [equal_value(pred_args.get(k), v) for k, v in expected.items()]
            arg_acc = sum(arg_hits) / len(arg_hits) if arg_hits else float(pred_args == expected)
            name_ok = equal_value(pred.get("name"), gold["name"])
            exact = name_ok and set(pred_args) == set(expected) and all(arg_hits)
            gold_scalars = []
            def collect_scalars(value: Any) -> None:
                if isinstance(value, dict):
                    for nested in value.values(): collect_scalars(nested)
                elif isinstance(value, list):
                    for nested in value: collect_scalars(nested)
                elif value is not None:
                    gold_scalars.append(str(value).lower())
            collect_scalars(expected)
            surface_recall = (sum(v in context.lower() for v in gold_scalars) / len(gold_scalars)
                              if gold_scalars else 1.0)
            row["conditions"][condition] = {
                "prediction": pred, "raw": raw, "json_valid": valid,
                "tool_name_correct": float(name_ok), "argument_accuracy": arg_acc,
                "exact": float(exact), "context_chars": len(context), "latency_s": latency,
                "gold_argument_surface_recall": surface_recall,
            }
            if condition in {"summary", "state", "control", "ledger384", "ledger512", "control_nested"}:
                row["conditions"][condition]["memory_text"] = context
        return index, row

    pending = [(i + args.start, q) for i, q in enumerate(qas) if q["qa_id"] not in completed]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_qa, item): item[1]["qa_id"] for item in pending}
        for done_count, future in enumerate(as_completed(futures), 1):
            index, row = future.result()
            completed[row["qa_id"]] = row
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            scores = " ".join(f"{c}={int(row['conditions'][c]['exact'])}" for c in conditions)
            print(f"[{done_count}/{len(pending)}] index={index} qa={row['qa_id']} level={row['level']} {scores}", flush=True)

    wanted = {q["qa_id"] for q in qas}
    rows = sorted((r for qid, r in completed.items() if qid in wanted), key=lambda r: r["index"])
    summary = build_summary(rows, conditions, args.seed)
    summary.update({
        "protocol": "query-unknown per-session memory; paired deterministic executor",
        "memory_tokens": args.memory_tokens, "model": args.served_model,
        "ledger_base_tokens": args.ledger_base_tokens, "dynamics_tokens": args.dynamics_tokens,
        "selection": {"start": args.start, "requested": args.num_samples, "levels": sorted(levels)},
    })
    summary_path = Path(str(args.output) + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {args.output}\nsummary: {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Query-unknown procedural-memory headroom test on Mem2ActBench.

The script never exposes the evaluation query or gold tool call while writing
memory.  It compares four execution conditions using the same OpenAI-compatible
vLLM endpoint:

  * no_memory: query and target tool schema only
  * full_history: the official benchmark-selected historical sessions
  * summary: fixed-budget, query-unknown factual summaries of those sessions
  * option: fixed-budget, query-unknown procedural memories of those sessions

No training data is created.  The official Mem2ActBench-small files are read
directly and the output is a JSONL audit trail plus a paired-bootstrap summary.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROMPT_VERSION = "optionmem-headroom-v1"
ALL_CONDITIONS = ("no_memory", "full_history", "summary", "option")

SUMMARY_SYSTEM = """You maintain long-term memory for an autonomous tool-using assistant.
Compress the historical interaction into a query-independent memory that may help with unknown future requests.
Preserve concrete entities, user preferences, identifiers, constraints, time-sensitive updates, tool inputs, tool outputs, and failures.
Do not guess and do not answer any future query. Stay within the requested memory budget."""

OPTION_SYSTEM = """You compile past agent experience into reusable procedural memory for unknown future tasks.
Write compact records with these fields when supported by the history:
- WHEN: observable initiation conditions or user intent
- STATE: relevant entities, identifiers, preferences, constraints, and newest valid values
- PROCEDURE: ordered tool/action steps and how arguments are grounded
- SUCCESS: observable completion condition or useful tool result
- FAILURE/GOTCHA: failed actions, stale values, conflicts, or conditions requiring clarification
- PROVENANCE: short source/turn cue; never invent evidence
Prefer executable decision rules over narrative summaries. Do not answer a future query because it is not known yet.
Stay within the requested memory budget."""

EXECUTOR_SYSTEM = """You are a tool-using assistant. Select the required tool call using only the supplied query, target tool schema, and optional memory/context.
Return exactly one JSON object with this schema:
{"name": "tool_name", "arguments": {"argument_name": "value"}}
Do not include Markdown, explanations, or additional keys."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--served-model", default="qwen25-7b")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    parser.add_argument("--levels", default="all", help="Comma-separated L1/L2/L3/L4 or all")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--memory-max-tokens", type=int, default=512)
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument("--writer-chunk-chars", type=int, default=12000)
    parser.add_argument("--max-context-chars", type=int, default=24000)
    parser.add_argument("--max-memory-context-chars", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any, lock: threading.Lock) -> None:
    line = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def render_session(session: dict[str, Any]) -> str:
    rendered: list[str] = [f"SESSION {session.get('session_id', 'unknown')}"]
    for index, turn in enumerate(session.get("turns", [])):
        role = str(turn.get("role", "unknown")).upper()
        source = turn.get("source_id", "")
        content = str(turn.get("content", "")).strip()
        parts = [f"[{index:03d}] {role}", f"source={source}" if source else ""]
        prefix = " ".join(part for part in parts if part)
        rendered.append(f"{prefix}: {content}")
        if turn.get("tool_calls"):
            rendered.append("TOOL_CALLS: " + compact_json(turn["tool_calls"]))
        if role == "TOOL" and turn.get("name"):
            rendered.append(f"TOOL_NAME: {turn['name']}")
    return "\n".join(rendered)


def split_text(text: str, chunk_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) > chunk_chars:
            chunks.append("".join(current))
            current, size = [], 0
        if len(line) > chunk_chars:
            for start in range(0, len(line), chunk_chars):
                piece = line[start : start + chunk_chars]
                if current:
                    chunks.append("".join(current))
                    current, size = [], 0
                chunks.append(piece)
            continue
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def balanced_clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n...[middle omitted to respect context budget]...\n"
    remaining = max(0, max_chars - len(marker))
    left = remaining // 2
    return text[:left] + marker + text[-(remaining - left) :]


def pack_segments(segments: Iterable[str], max_chars: int) -> str:
    values = [segment for segment in segments if segment]
    if not values:
        return ""
    joined = "\n\n".join(values)
    if len(joined) <= max_chars:
        return joined
    per_segment = max(256, max_chars // len(values))
    clipped = [balanced_clip(value, per_segment) for value in values]
    return balanced_clip("\n\n".join(clipped), max_chars)


class VLLMClient:
    def __init__(self, args: argparse.Namespace):
        self.url = args.base_url.rstrip("/") + "/v1/chat/completions"
        self.model = args.served_model
        self.api_key = args.api_key
        self.temperature = args.temperature
        self.timeout = args.timeout
        self.retries = args.retries

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        seed: int,
    ) -> tuple[str, float]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        encoded = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            request = urllib.request.Request(
                self.url,
                data=encoded,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                return str(content), time.perf_counter() - start
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(min(8.0, 0.75 * (2**attempt)))
        raise RuntimeError(f"vLLM request failed after {self.retries} attempts: {last_error}")


def memory_cache_key(
    mode: str,
    session: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    digest = hashlib.sha256(render_session(session).encode("utf-8")).hexdigest()[:16]
    return "|".join(
        [
            PROMPT_VERSION,
            args.served_model,
            mode,
            str(args.memory_max_tokens),
            str(args.writer_chunk_chars),
            session["session_id"],
            digest,
        ]
    )


def write_memory(
    client: VLLMClient,
    session: dict[str, Any],
    mode: str,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    if mode not in {"summary", "option"}:
        raise ValueError(mode)
    system = SUMMARY_SYSTEM if mode == "summary" else OPTION_SYSTEM
    memory = "No previous memory."
    total_latency = 0.0
    chunks = split_text(render_session(session), args.writer_chunk_chars)
    for chunk_index, chunk in enumerate(chunks):
        prompt = f"""Update the {mode} memory using the next chronological history chunk.
The future task/query is deliberately unknown.
Memory budget: at most {args.memory_max_tokens} output tokens.

PREVIOUS MEMORY
{memory}

NEXT HISTORY CHUNK
{chunk}

UPDATED MEMORY"""
        memory, latency = client.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=args.memory_max_tokens,
            seed=seed + chunk_index,
        )
        total_latency += latency
        memory = memory.strip()
    return {
        "text": memory,
        "chars": len(memory),
        "latency_s": total_latency,
        "chunks": len(chunks),
    }


def find_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    for start, char in enumerate(stripped):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(stripped)):
            current = stripped[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
            elif current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(stripped[start : end + 1])
                        return value if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        break
    return None


def normalize_tool_call(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("tool_call"), dict):
        value = value["tool_call"]
    if isinstance(value.get("function"), dict):
        value = value["function"]
    name = value.get("name") or value.get("tool_name")
    arguments = value.get("arguments", value.get("args", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name.strip(), "arguments": arguments}


def normalized_scalar(value: Any) -> Any:
    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value.strip()).casefold()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            try:
                return float(text)
            except ValueError:
                pass
        return text
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [normalized_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key).casefold(): normalized_scalar(item) for key, item in value.items()}
    return value


def values_equal(left: Any, right: Any) -> bool:
    left, right = normalized_scalar(left), normalized_scalar(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6)
    return left == right


def score_tool_call(predicted: dict[str, Any] | None, expected: dict[str, Any]) -> dict[str, float]:
    if predicted is None:
        return {"valid_json": 0.0, "name": 0.0, "arguments": 0.0, "exact": 0.0}
    expected_args = expected.get("arguments", {})
    predicted_args = predicted.get("arguments", {})
    name_correct = float(str(predicted.get("name", "")).casefold() == str(expected.get("name", "")).casefold())
    per_argument = [
        float(key in predicted_args and values_equal(predicted_args[key], value))
        for key, value in expected_args.items()
    ]
    argument_accuracy = statistics.fmean(per_argument) if per_argument else 1.0
    exact_arguments = argument_accuracy == 1.0 and set(predicted_args) == set(expected_args)
    return {
        "valid_json": 1.0,
        "name": name_correct,
        "arguments": argument_accuracy,
        "exact": float(bool(name_correct) and exact_arguments),
    }


def build_executor_prompt(
    qa: dict[str, Any],
    condition: str,
    sessions: list[dict[str, Any]],
    memories: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> str:
    if condition == "no_memory":
        context = "No historical memory is available."
    elif condition == "full_history":
        context = pack_segments((render_session(session) for session in sessions), args.max_context_chars)
    elif condition in {"summary", "option"}:
        segments = []
        for session in sessions:
            item = memories[condition][session["session_id"]]
            segments.append(f"SESSION {session['session_id']} {condition.upper()} MEMORY\n{item['text']}")
        context = pack_segments(segments, args.max_memory_context_chars)
    else:
        raise ValueError(condition)

    return f"""HISTORICAL CONTEXT / MEMORY ({condition})
{context}

CURRENT USER QUERY
{qa['query']}

TARGET TOOL SCHEMA
{json.dumps(qa['target_tool_schema'], ensure_ascii=False, indent=2)}

Return the single required tool call as JSON."""


def select_questions(qas: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    allowed = None if args.levels == "all" else {item.strip() for item in args.levels.split(",") if item.strip()}
    filtered = [qa for qa in qas if allowed is None or qa.get("complexity_metadata", {}).get("level") in allowed]
    rng = random.Random(args.seed)
    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qa in filtered:
        by_level[str(qa.get("complexity_metadata", {}).get("level", "unknown"))].append(qa)
    for values in by_level.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    levels = sorted(by_level)
    while len(selected) < min(args.num_samples, len(filtered)):
        made_progress = False
        for level in levels:
            if by_level[level] and len(selected) < args.num_samples:
                selected.append(by_level[level].pop())
                made_progress = True
        if not made_progress:
            break
    return selected


def bootstrap_paired_ci(differences: list[float], iterations: int, seed: int) -> list[float]:
    if not differences:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        estimates.append(statistics.fmean(rng.choice(differences) for _ in differences))
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return [lower, upper]


def summarize(
    rows: list[dict[str, Any]],
    conditions: tuple[str, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "num_questions": len(rows),
        "level_counts": dict(Counter(row["level"] for row in rows)),
        "conditions": {},
        "paired_exact_differences": {},
    }
    for condition in conditions:
        records = [row["conditions"][condition] for row in rows]
        result["conditions"][condition] = {
            metric: statistics.fmean(record["score"][metric] for record in records)
            for metric in ("valid_json", "name", "arguments", "exact")
        }
        result["conditions"][condition]["latency_s_mean"] = statistics.fmean(record["latency_s"] for record in records)

    for left, right in (("option", "summary"), ("option", "full_history"), ("summary", "full_history")):
        if left not in conditions or right not in conditions:
            continue
        differences = [
            row["conditions"][left]["score"]["exact"] - row["conditions"][right]["score"]["exact"]
            for row in rows
        ]
        result["paired_exact_differences"][f"{left}_minus_{right}"] = {
            "mean": statistics.fmean(differences),
            "bootstrap_95_ci": bootstrap_paired_ci(differences, args.bootstrap, args.seed + len(left) + len(right)),
            "wins": sum(value > 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
        }
    return result


def main() -> int:
    args = parse_args()
    conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    unknown = set(conditions) - set(ALL_CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    if args.num_samples <= 0 or args.concurrency <= 0:
        raise ValueError("num-samples and concurrency must be positive")

    qa_path = args.data_dir / "qa_dataset.jsonl"
    session_path = args.data_dir / "toolmem_conversation.jsonl"
    if not qa_path.exists() or not session_path.exists():
        raise FileNotFoundError(
            f"Expected official Mem2ActBench-small files at {qa_path} and {session_path}"
        )

    qas = select_questions(load_jsonl(qa_path), args)
    sessions = load_jsonl(session_path)
    source_to_session: dict[str, dict[str, Any]] = {}
    for session in sessions:
        for source_id in session.get("original_conversation_ids", []):
            source_to_session[source_id] = session

    qa_sessions: dict[str, list[dict[str, Any]]] = {}
    referenced_sessions: dict[str, dict[str, Any]] = {}
    missing: dict[str, list[str]] = {}
    for qa in qas:
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        missing_ids: list[str] = []
        for source_id in qa.get("source_conversation_ids", []):
            session = source_to_session.get(source_id)
            if session is None:
                missing_ids.append(source_id)
                continue
            if session["session_id"] not in seen:
                ordered.append(session)
                seen.add(session["session_id"])
                referenced_sessions[session["session_id"]] = session
        if missing_ids:
            missing[qa["qa_id"]] = missing_ids
        qa_sessions[qa["qa_id"]] = ordered
    if missing:
        raise ValueError(f"Official source IDs could not be resolved: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        completed = {row["qa_id"]: row for row in load_jsonl(args.output)}

    cache_path = args.cache or args.output.with_suffix(args.output.suffix + ".memory_cache.json")
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    client = VLLMClient(args)
    writer_modes = tuple(condition for condition in ("summary", "option") if condition in conditions)
    memories: dict[str, dict[str, dict[str, Any]]] = {mode: {} for mode in writer_modes}
    writer_jobs: list[tuple[str, dict[str, Any], str]] = []
    for mode in writer_modes:
        for session in referenced_sessions.values():
            key = memory_cache_key(mode, session, args)
            if key in cache:
                memories[mode][session["session_id"]] = cache[key]
            else:
                writer_jobs.append((mode, session, key))

    print(
        f"questions={len(qas)} referenced_sessions={len(referenced_sessions)} "
        f"memory_cache_hits={sum(len(values) for values in memories.values())} writer_jobs={len(writer_jobs)}",
        flush=True,
    )
    cache_lock = threading.Lock()

    def run_writer(job: tuple[str, dict[str, Any], str]) -> tuple[str, str, str, dict[str, Any]]:
        mode, session, key = job
        numeric = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        item = write_memory(client, session, mode, args, args.seed + numeric % 1_000_000)
        return mode, session["session_id"], key, item

    if writer_jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for index, future in enumerate(concurrent.futures.as_completed([pool.submit(run_writer, job) for job in writer_jobs]), 1):
                mode, session_id, key, item = future.result()
                memories[mode][session_id] = item
                with cache_lock:
                    cache[key] = item
                    atomic_write_json(cache_path, cache)
                print(f"[memory {index}/{len(writer_jobs)}] {mode} {session_id} chars={item['chars']}", flush=True)

    pending = [qa for qa in qas if qa["qa_id"] not in completed]
    output_lock = threading.Lock()

    def run_question(qa: dict[str, Any]) -> dict[str, Any]:
        condition_results: dict[str, Any] = {}
        for condition_index, condition in enumerate(conditions):
            prompt = build_executor_prompt(qa, condition, qa_sessions[qa["qa_id"]], memories, args)
            response, latency = client.complete(
                [{"role": "system", "content": EXECUTOR_SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=args.answer_max_tokens,
                seed=args.seed + int(re.sub(r"\D", "", qa["qa_id"]) or 0) * 10 + condition_index,
            )
            parsed = normalize_tool_call(find_json_object(response))
            condition_results[condition] = {
                "response": response,
                "parsed": parsed,
                "score": score_tool_call(parsed, qa["tool_call"]),
                "latency_s": latency,
                "prompt_chars": len(prompt),
            }
        return {
            "qa_id": qa["qa_id"],
            "level": qa.get("complexity_metadata", {}).get("level", "unknown"),
            "query": qa["query"],
            "source_conversation_ids": qa.get("source_conversation_ids", []),
            "session_ids": [session["session_id"] for session in qa_sessions[qa["qa_id"]]],
            "expected": qa["tool_call"],
            "conditions": condition_results,
        }

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_question, qa) for qa in pending]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result()
                append_jsonl(args.output, row, output_lock)
                completed[row["qa_id"]] = row
                exacts = " ".join(
                    f"{condition}={row['conditions'][condition]['score']['exact']:.0f}" for condition in conditions
                )
                print(f"[qa {index}/{len(pending)}] {row['qa_id']} {row['level']} {exacts}", flush=True)

    ordered_rows = [completed[qa["qa_id"]] for qa in qas]
    summary = summarize(ordered_rows, conditions, args)
    summary["model"] = args.served_model
    summary["data_dir"] = str(args.data_dir)
    summary["seed"] = args.seed
    summary["memory_max_tokens"] = args.memory_max_tokens
    summary["query_hidden_during_memory_writing"] = True
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {args.output}")
    print(f"summary: {summary_path}")
    print(f"memory_cache: {cache_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed JSONL rows and memory cache are resumable.", file=sys.stderr)
        raise SystemExit(130)

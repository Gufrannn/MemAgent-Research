#!/usr/bin/env python3
"""Lightweight UMA-compatible response generation.

This runner intentionally avoids importing the full official evaluator stack.
It only loads a UMA task, feeds chunks into an agent in chronological order,
and writes JSONL records compatible with the existing strategy-matrix scripts:

    qid, query, expected_answer, response, generation_time

Use it when the generation environment has the agent/model dependencies but
does not have optional evaluation packages such as rouge_score/backoff.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from config import AGENT_CLASS, API_CONFIG_LOCAL, DATASET_LOADERS


def load_completed(path: Path, *, force_overwrite: bool) -> set[str]:
    if force_overwrite and path.exists():
        path.unlink()
        return set()
    completed: set[str] = set()
    if not path.exists():
        return completed
    kept: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("qid")
            response = str(row.get("response", ""))
            if qid and not response.startswith("ERROR") and qid not in completed:
                completed.add(qid)
                kept.append(row)
    with path.open("w", encoding="utf-8") as output_file:
        for row in kept:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return completed


def load_manifest_qids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        qids = payload.get("question_ids")
        if qids is None and "items" in payload:
            qids = [item.get("question_id") for item in payload["items"]]
    elif isinstance(payload, list):
        qids = payload
    else:
        raise ValueError(f"Unsupported manifest format: {path}")
    out = [str(qid) for qid in qids if qid]
    if not out:
        raise ValueError(f"Manifest has no question ids: {path}")
    return out


async def process_sample(sample: Any, agent_cls: type, agent_kwargs: dict[str, Any], semaphore: asyncio.Semaphore, output_file: Path, lock: asyncio.Lock) -> int:
    async with semaphore:
        print(f"{agent_cls.__name__} processing {sample.task_id}", flush=True)
        agent = agent_cls(**agent_kwargs)
        if hasattr(agent, "reset"):
            agent.reset()
        if hasattr(agent, "prepare_sample"):
            agent.prepare_sample(sample)

        position_groups: dict[int, list[Any]] = {}
        for question in sample.questions:
            position_groups.setdefault(question.position, []).append(question)

        chunk_idx = 0
        rows: list[dict[str, Any]] = []
        for position in sorted(position_groups):
            while chunk_idx <= position and chunk_idx < len(sample.chunks):
                chunk = sample.chunks[chunk_idx]
                if hasattr(agent, "add_memory_async"):
                    await agent.add_memory_async(chunk)
                else:
                    agent.add_memory(chunk)
                chunk_idx += 1

            questions = position_groups[position]
            queries = [q.query for q in questions]
            started = time.time()
            responses = await agent.QA_batch_async(queries)
            elapsed = time.time() - started
            per_question = elapsed / max(1, len(questions))
            for question, response in zip(questions, responses):
                rows.append({
                    "qid": question.qid,
                    "query": question.query,
                    "expected_answer": question.answer,
                    "response": response,
                    "generation_time": per_question,
                })

        async with lock:
            with output_file.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)


async def main_async(args: argparse.Namespace) -> None:
    if args.agent not in AGENT_CLASS:
        raise SystemExit(f"Unknown agent: {args.agent}")
    if args.task not in DATASET_LOADERS:
        raise SystemExit(f"Unknown task: {args.task}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"responses_{args.agent_id}.jsonl"
    completed = load_completed(output_file, force_overwrite=args.force_overwrite)

    eval_set = DATASET_LOADERS[args.task]()
    manifest_qid_list = load_manifest_qids(args.qid_manifest)
    manifest_qids = set(manifest_qid_list) if manifest_qid_list is not None else None
    manifest_rank = {qid: idx for idx, qid in enumerate(manifest_qid_list or [])}
    remaining = []
    for sample in eval_set:
        questions = [
            q
            for q in sample.questions
            if q.qid not in completed and (manifest_qids is None or q.qid in manifest_qids)
        ]
        if not questions:
            continue
        remaining.append(type(sample)(task_id=sample.task_id, chunks=sample.chunks, questions=questions))
    if args.preserve_manifest_order and manifest_rank:
        remaining.sort(
            key=lambda sample: min(
                manifest_rank.get(q.qid, len(manifest_rank))
                for q in sample.questions
            )
        )
    if args.limit_samples is not None:
        remaining = remaining[: args.limit_samples]

    if not remaining:
        print(f"All questions already completed: {output_file}")
        return

    print(
        f"Loaded {len(eval_set)} samples; running {len(remaining)} samples"
        + (f" from manifest {args.qid_manifest}" if args.qid_manifest else ""),
        flush=True,
    )

    client = AsyncOpenAI(**API_CONFIG_LOCAL)
    agent_cls = AGENT_CLASS[args.agent]
    agent_kwargs = {"model_name": args.model, "client": client}
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    started = time.time()
    tasks = [
        process_sample(sample, agent_cls, agent_kwargs, semaphore, output_file, lock)
        for sample in remaining
    ]
    counts = await asyncio.gather(*tasks)
    total = sum(counts)
    elapsed = time.time() - started
    print(f"Generated {total} responses in {elapsed:.2f}s")
    print(f"Saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--qid-manifest", type=Path, help="Optional JSON manifest containing question_ids to run")
    parser.add_argument("--limit-samples", type=int, help="Optional hard cap after manifest filtering")
    parser.add_argument("--preserve-manifest-order", action="store_true", help="Run manifest-selected samples in manifest order before applying --limit-samples")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()

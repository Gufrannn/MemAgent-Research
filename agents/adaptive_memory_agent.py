"""Adaptive Memory Agent.

This is the first executable prototype for Dynamic Memory Computation (DMC).
It is deliberately heuristic: the goal is to make the memory computation graph
explicit, traceable, and comparable before replacing the controller with a
learned/RL policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List

from .base_agent import BaseAgent
from .concat_agent import QA_PROMPT


@dataclass
class MemoryItem:
    idx: int
    text: str
    tokens: Counter[str]
    n_chars: int
    write_reason: str


class BudgetTracker:
    def __init__(self, max_prompt_chars: int, answer_max_tokens: int):
        self.max_prompt_chars = max_prompt_chars
        self.answer_max_tokens = answer_max_tokens

    def context_budget_chars(self, query: str) -> int:
        reserved_for_answer = self.answer_max_tokens * 3
        query_chars = len(QA_PROMPT.format(query))
        return max(1000, self.max_prompt_chars - reserved_for_answer - query_chars - 300)


class TraceLogger:
    def __init__(self, path: str | None):
        self.path = path

    def write(self, record: dict) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")


class HeuristicOperationRouter:
    """A deterministic controller over fixed memory primitives.

    The router is intentionally simple. Its purpose is not to be the final
    method; it provides an executable lower bound and trace schema for the
    future learned controller.
    """

    def __init__(self, min_informative_tokens: int = 3, retrieve_multiplier: int = 3):
        self.min_informative_tokens = min_informative_tokens
        self.retrieve_multiplier = retrieve_multiplier

    def write_or_skip(self, chunk: str, tokens: set[str]) -> tuple[str, str]:
        if len(tokens) < self.min_informative_tokens:
            return "SKIP", "low_information"
        if len(chunk.strip()) < 20:
            return "SKIP", "too_short"
        return "WRITE", "informative_chunk"

    def query_plan(self, n_memory_items: int, query_tokens: set[str]) -> list[str]:
        if n_memory_items == 0:
            return ["ANSWER"]
        if not query_tokens:
            return ["RETRIEVE_RECENT", "ANSWER"]
        return ["RETRIEVE", "FILTER", "ANSWER"]


class AdaptiveMemoryAgent(BaseAgent):
    """Dynamic memory computation with explicit primitive traces.

    First-version primitive set:
    - WRITE / SKIP during observation ingestion.
    - RETRIEVE / RETRIEVE_RECENT at question time.
    - FILTER as query-biased sentence extraction if retrieved context exceeds
      the budget.
    - ANSWER through the same OpenAI-compatible chat endpoint as UMA baselines.

    Environment knobs:
    - AMC_MAX_PROMPT_CHARS, default 49152.
    - AMC_MAX_TOKENS, default 8192.
    - AMC_TOP_K, default 40.
    - AMC_TRACE_PATH, optional JSONL trace file.
    - AMC_TRACE_STATE_TEXT, default off.  When enabled, store full post-operation
      memory-state text in traces for closed-loop feedback-value diagnostics.
    """

    def __init__(
        self,
        client=None,
        model_name: str = "gpt4.1",
        max_prompt_chars: int | None = None,
        answer_max_tokens: int | None = None,
        top_k: int | None = None,
    ):
        super().__init__(client, model_name)
        self.items: List[MemoryItem] = []
        self.next_idx = 0
        self.max_prompt_chars = max_prompt_chars or int(os.getenv("AMC_MAX_PROMPT_CHARS", "49152"))
        self.answer_max_tokens = answer_max_tokens or int(os.getenv("AMC_MAX_TOKENS", "8192"))
        self.top_k = top_k or int(os.getenv("AMC_TOP_K", "40"))
        self.force_filter = os.getenv("AMC_FORCE_FILTER", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.disable_write_skip = os.getenv("AMC_DISABLE_WRITE_SKIP", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.trace_state_text = os.getenv("AMC_TRACE_STATE_TEXT", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.temperature = float(os.getenv("UMA_TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("UMA_TOP_P", "1.0"))
        self.budget = BudgetTracker(self.max_prompt_chars, self.answer_max_tokens)
        self.router = HeuristicOperationRouter()
        self.trace = TraceLogger(os.getenv("AMC_TRACE_PATH"))
        self._idf: dict[str, float] = {}
        self._avg_doc_len = 1.0

    def reset(self) -> None:
        self.items = []
        self.next_idx = 0
        self._idf = {}
        self._avg_doc_len = 1.0

    async def add_memory_async(self, chunk: str):
        tokens = self._query_terms(chunk)
        if self.disable_write_skip:
            op, reason = "WRITE", "write_skip_disabled"
        else:
            op, reason = self.router.write_or_skip(chunk, tokens)
        if op == "WRITE":
            self.items.append(
                MemoryItem(
                    idx=self.next_idx,
                    text=chunk,
                    tokens=Counter(self._tokenize(chunk)),
                    n_chars=len(chunk),
                    write_reason=reason,
                )
            )
            self._rebuild_idf()
        self.trace.write(
            {
                "phase": "ingest",
                "operation": op,
                "memory_idx": self.next_idx,
                "reason": reason,
                "chunk_chars": len(chunk),
                "n_memory_items_after": len(self.items),
            }
        )
        self.next_idx += 1

    async def QA_batch_async(self, query_list: list[str], batch_size: int = 5) -> list[str]:
        return [await self.QA_async(query) for query in query_list]

    async def QA_async(self, query: str) -> str:
        started = time.time()
        query_tokens = self._query_terms(query)
        budget_chars = self.budget.context_budget_chars(query)
        operations = self.router.query_plan(len(self.items), query_tokens)
        selected_items: list[MemoryItem] = []
        context = "No previous memory"
        op_records: list[dict] = []

        try:
            for operation in operations:
                if operation == "RETRIEVE":
                    selected_items = self._retrieve(query_tokens)
                    context, admitted_items = self._pack_items_to_budget_with_items(selected_items, budget_chars)
                    op_records.append(
                        self._op_record(
                            operation,
                            selected_items,
                            context,
                            budget_chars,
                            admitted_items=admitted_items,
                            retrieved_items=selected_items,
                        )
                    )
                elif operation == "RETRIEVE_RECENT":
                    selected_items = self._retrieve_recent(budget_chars)
                    context, admitted_items = self._pack_items_to_budget_with_items(selected_items, budget_chars)
                    op_records.append(
                        self._op_record(
                            operation,
                            selected_items,
                            context,
                            budget_chars,
                            admitted_items=admitted_items,
                            retrieved_items=selected_items,
                        )
                    )
                elif operation in {"FILTER", "COMPRESS"}:
                    did_filter = self.force_filter or len(context) > budget_chars
                    filtered_sentence_count = 0
                    filter_extra = {
                        "did_filter": did_filter,
                        "filtered_sentence_count": filtered_sentence_count,
                        "force_filter": self.force_filter,
                        "admitted_source_indices": [item.idx for item in selected_items],
                        "n_admitted_sources": len(selected_items),
                    }
                    if did_filter:
                        context, filtered_sentence_count, admitted_source_indices = self._filter_with_source_indices(
                            selected_items,
                            query_tokens,
                            budget_chars,
                        )
                        filter_extra.update(
                            {
                                "filtered_sentence_count": filtered_sentence_count,
                                "admitted_source_indices": admitted_source_indices,
                                "n_admitted_sources": len(admitted_source_indices),
                            }
                        )
                    op_records.append(
                        self._op_record(
                            operation,
                            selected_items,
                            context,
                            budget_chars,
                            filter_extra,
                        )
                    )
                elif operation == "ANSWER":
                    break

            prompt = f"Your memory:\n{context}\n\n{QA_PROMPT.format(query)}"
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.answer_max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Received empty response from API")
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            self._write_query_trace(query, operations, op_records, budget_chars, context, started, ok=True)
            return content.strip()
        except Exception as exc:
            self._write_query_trace(query, operations, op_records, budget_chars, context, started, ok=False, error=str(exc))
            return self._handle_api_error(exc, query)

    def _retrieve(self, query_tokens: set[str]) -> list[MemoryItem]:
        scored = []
        for item in self.items:
            score = self._bm25_score(query_tokens, item.tokens)
            if score > 0:
                scored.append((score, item.idx, item))
        if not scored:
            return self._retrieve_recent(self.budget.context_budget_chars(""))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [item for _, _, item in scored[: self.top_k]]

    def _retrieve_recent(self, budget_chars: int) -> list[MemoryItem]:
        selected: list[MemoryItem] = []
        used = 0
        for item in reversed(self.items):
            item_len = item.n_chars + 2
            if selected and used + item_len > budget_chars:
                continue
            selected.append(item)
            used += item_len
        return list(reversed(selected))

    def _filter(self, items: list[MemoryItem], query_tokens: set[str], budget_chars: int) -> tuple[str, int]:
        context, count, _ = self._filter_with_source_indices(items, query_tokens, budget_chars)
        return context, count

    def _filter_with_source_indices(
        self,
        items: list[MemoryItem],
        query_tokens: set[str],
        budget_chars: int,
    ) -> tuple[str, int, list[int]]:
        sentence_records: list[tuple[float, int, str]] = []
        for item in items:
            for sentence_idx, sentence in enumerate(self._sentences(item.text)):
                stoks = Counter(self._tokenize(sentence))
                score = self._bm25_score(query_tokens, stoks)
                if score > 0:
                    sentence_records.append((score, item.idx * 10000 + sentence_idx, sentence))

        if not sentence_records:
            context, admitted_items = self._pack_items_to_budget_with_items(items, budget_chars)
            return context, 0, [item.idx for item in admitted_items]

        sentence_records.sort(key=lambda row: (-row[0], row[1]))
        selected: list[tuple[int, str]] = []
        used = 0
        for _, order, sentence in sentence_records:
            sent_len = len(sentence) + 1
            if selected and used + sent_len > budget_chars:
                continue
            selected.append((order, sentence))
            used += sent_len
        selected.sort(key=lambda row: row[0])
        admitted_source_indices = sorted({order // 10000 for order, _ in selected})
        return "\n".join(sentence for _, sentence in selected), len(selected), admitted_source_indices

    def _pack_items_to_budget(self, items: list[MemoryItem], budget_chars: int) -> str:
        context, _ = self._pack_items_to_budget_with_items(items, budget_chars)
        return context

    def _pack_items_to_budget_with_items(
        self,
        items: list[MemoryItem],
        budget_chars: int,
    ) -> tuple[str, list[MemoryItem]]:
        selected: list[MemoryItem] = []
        used = 0
        for item in items:
            item_len = item.n_chars + 2
            if selected and used + item_len > budget_chars:
                continue
            selected.append(item)
            used += item_len
        return "\n\n".join(item.text for item in selected) or "No previous memory", selected

    def _op_record(
        self,
        operation: str,
        selected_items: list[MemoryItem],
        context: str,
        budget_chars: int,
        extra: dict | None = None,
        *,
        admitted_items: list[MemoryItem] | None = None,
        retrieved_items: list[MemoryItem] | None = None,
    ) -> dict:
        admitted_items = selected_items if admitted_items is None else admitted_items
        retrieved_items = selected_items if retrieved_items is None else retrieved_items
        record = {
            "operation": operation,
            "selected_indices": [item.idx for item in selected_items],
            "n_selected": len(selected_items),
            "retrieved_source_indices": [item.idx for item in retrieved_items],
            "n_retrieved_sources": len(retrieved_items),
            "admitted_source_indices": [item.idx for item in admitted_items],
            "n_admitted_sources": len(admitted_items),
            "trace_schema_version": "retrieved_vs_admitted_v1",
            "context_chars": len(context),
            "context_sha1": hashlib.sha1(context.encode("utf-8")).hexdigest(),
            "budget_chars": budget_chars,
        }
        if self.trace_state_text:
            record["state_text"] = context
        if extra:
            record.update(extra)
        return record

    def _write_query_trace(
        self,
        query: str,
        operations: list[str],
        op_records: list[dict],
        budget_chars: int,
        context: str,
        started: float,
        ok: bool,
        error: str | None = None,
    ) -> None:
        self.trace.write(
            {
                "phase": "qa",
                "ok": ok,
                "error": error,
                "query_sha1": hashlib.sha1(query.encode("utf-8")).hexdigest(),
                "query_prefix": query[:200],
                "operations": operations,
                "op_records": op_records,
                "n_memory_items": len(self.items),
                "budget_chars": budget_chars,
                "final_context_chars": len(context),
                "final_context_sha1": hashlib.sha1(context.encode("utf-8")).hexdigest(),
                "prompt_sha1": hashlib.sha1(
                    f"Your memory:\n{context}\n\n{QA_PROMPT.format(query)}".encode("utf-8")
                ).hexdigest(),
                "total_memory_chars": sum(item.n_chars + 2 for item in self.items),
                "model": self.model_name,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.answer_max_tokens,
                "top_k": self.top_k,
                "force_filter": self.force_filter,
                "disable_write_skip": self.disable_write_skip,
                "trace_state_text": self.trace_state_text,
                "latency_s": time.time() - started,
            }
        )

    def _rebuild_idf(self) -> None:
        doc_freq: Counter[str] = Counter()
        for item in self.items:
            doc_freq.update(item.tokens.keys())
        n_docs = max(1, len(self.items))
        self._idf = {
            token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }
        self._avg_doc_len = max(1.0, sum(sum(item.tokens.values()) for item in self.items) / n_docs)

    def _bm25_score(self, query_tokens: Iterable[str], counts: Counter[str]) -> float:
        k1 = 1.5
        b = 0.75
        doc_len = max(1, sum(counts.values()))
        denom_norm = k1 * (1.0 - b + b * doc_len / self._avg_doc_len)
        score = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            score += self._idf.get(token, 0.0) * ((tf * (k1 + 1.0)) / (tf + denom_norm))
        return score

    @staticmethod
    def _sentences(text: str) -> list[str]:
        pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [piece.strip() for piece in pieces if piece.strip()]

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        return {
            token
            for token in AdaptiveMemoryAgent._tokenize(text)
            if len(token) > 2 and token not in _STOPWORDS
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+", text.lower())


_STOPWORDS = {
    "the", "and", "for", "that", "this", "what", "who", "whom", "whose",
    "when", "where", "which", "why", "how", "are", "was", "were", "has",
    "had", "have", "does", "did", "from", "with", "about", "into", "over",
    "under", "between", "after", "before", "also", "not", "you", "your",
    "their", "there", "they", "them", "his", "her", "its", "our", "out",
    "all", "any", "can", "could", "would", "should", "than", "then",
}

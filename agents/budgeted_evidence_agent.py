"""Budgeted Evidence Memory agent.

This agent is intentionally lightweight: it does not require an embedding
service, a vector database, or training.  It is a first executable probe for a
larger idea: memory agents should allocate scarce context budget to evidence
that is query-relevant, not merely recent.
"""

from __future__ import annotations

import math
import os
import re
import json
import hashlib
from collections import Counter
from typing import Iterable, List

from .base_agent import BaseAgent
from .concat_agent import QA_PROMPT


class BudgetedEvidenceAgent(BaseAgent):
    """Select query-relevant memory chunks under a fixed prompt budget.

    Compared with ConcatAgent, the only intended algorithmic change is the
    memory selection policy before the final QA call:

    - ConcatAgent keeps the suffix of the full memory when context is too long.
    - BudgetedEvidenceAgent ranks chunks by BM25-style lexical evidence score,
      keeps the highest-scoring chunks that fit the same approximate budget,
      then restores chronological order before prompting the model.

    This makes the first experiment cheap and reviewable.  It also leaves a
    natural path toward RL: replace the hand-written selector with a learned
    budget allocator while retaining this agent as the transparent oracle-lite
    diagnostic.
    """

    def __init__(
        self,
        client=None,
        model_name: str = "gpt4.1",
        max_prompt_chars: int | None = None,
        answer_max_tokens: int | None = None,
        min_recent_chunks: int | None = None,
    ):
        super().__init__(client, model_name)
        self.memory: List[str] = []
        self.max_prompt_chars = max_prompt_chars or int(os.getenv("BEM_MAX_PROMPT_CHARS", "49152"))
        self.answer_max_tokens = answer_max_tokens or int(os.getenv("BEM_MAX_TOKENS", "8192"))
        self.min_recent_chunks = min_recent_chunks if min_recent_chunks is not None else int(os.getenv("BEM_MIN_RECENT_CHUNKS", "0"))
        self.strategy = os.getenv("BEM_STRATEGY", "bm25").strip().lower()
        self.trace_path = os.getenv("BEM_TRACE_PATH")
        self.random_seed = os.getenv("BEM_RANDOM_SEED", "20260830")
        self.hybrid_recency_alpha = float(os.getenv("BEM_HYBRID_RECENCY_ALPHA", "0.25"))
        self.mmr_lambda = float(os.getenv("BEM_MMR_LAMBDA", "0.75"))
        self.sentence_filter = os.getenv("BEM_SENTENCE_FILTER", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.filter_top_k = int(os.getenv("BEM_FILTER_TOP_K", "40"))
        self.topk_only = os.getenv("BEM_TOPK_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.conditional_filter = os.getenv("BEM_CONDITIONAL_FILTER", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.adaptive_store = os.getenv("BEM_ADAPTIVE_STORE", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.min_informative_tokens = int(os.getenv("BEM_MIN_INFORMATIVE_TOKENS", "3"))
        self.temperature = float(os.getenv("UMA_TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("UMA_TOP_P", "1.0"))
        self._idf: dict[str, float] = {}
        self._doc_tokens: list[Counter[str]] = []
        self._avg_doc_len = 1.0

    async def add_memory_async(self, chunk: str):
        if self.adaptive_store and not self._should_write(chunk):
            return
        self.memory.append(chunk)
        self._rebuild_index()

    def reset(self) -> None:
        self.memory = []
        self._idf = {}
        self._doc_tokens = []
        self._avg_doc_len = 1.0

    async def QA_batch_async(self, query_list: list[str], batch_size: int = 5) -> list[str]:
        # Sequential per-query calls make the selection policy observable and
        # avoid adding a second variable from JSON batch parsing.
        return [await self.QA_async(query) for query in query_list]

    async def QA_async(self, query: str) -> str:
        try:
            context, selected_indices, budget_chars, trace_extra = self._select_context(query)
            self._write_trace(query, selected_indices, budget_chars, context, trace_extra)
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
            return content.strip()
        except Exception as exc:
            return self._handle_api_error(exc, query)

    def _rebuild_index(self) -> None:
        self._doc_tokens = [Counter(self._tokenize(chunk)) for chunk in self.memory]
        if not self._doc_tokens:
            self._idf = {}
            self._avg_doc_len = 1.0
            return

        doc_freq: Counter[str] = Counter()
        for counts in self._doc_tokens:
            doc_freq.update(counts.keys())

        n_docs = len(self._doc_tokens)
        self._idf = {
            token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }
        self._avg_doc_len = max(1.0, sum(sum(c.values()) for c in self._doc_tokens) / n_docs)

    def _select_context(self, query: str) -> tuple[str, list[int], int, dict]:
        if not self.memory:
            return "No previous memory", [], self._context_budget_chars(query), {
                "retrieved_context_chars": 0,
                "did_filter": False,
                "filtered_sentence_count": 0,
            }

        query_tokens = self._query_terms(query)
        budget_chars = self._context_budget_chars(query)

        if self.topk_only or self.sentence_filter:
            candidate_indices = self._candidate_indices(query, query_tokens)
            candidate_context = "\n\n".join(self.memory[idx] for idx in sorted(candidate_indices))
            did_filter = self.sentence_filter and (
                not self.conditional_filter or len(candidate_context) > budget_chars
            )
            if did_filter:
                context, filtered_sentence_count = self._filter_sentences(candidate_indices, query_tokens, budget_chars)
            elif len(candidate_context) > budget_chars:
                context = self._pack_indices_to_budget(sorted(candidate_indices), budget_chars)
                filtered_sentence_count = 0
            else:
                context = candidate_context or "No previous memory"
                filtered_sentence_count = 0
            return context, sorted(candidate_indices), budget_chars, {
                "retrieved_context_chars": len(candidate_context),
                "did_filter": did_filter,
                "filtered_sentence_count": filtered_sentence_count,
            }

        if self.strategy == "recent":
            selected = self._pack_recent(budget_chars)
        elif self.strategy == "random":
            selected = self._pack_random(query, budget_chars)
        elif self.strategy == "hybrid":
            selected = self._rank_and_pack(query_tokens, budget_chars, use_recency=True)
        elif self.strategy == "mmr":
            selected = self._mmr_pack(query_tokens, budget_chars)
        elif not query_tokens:
            selected = self._pack_recent(budget_chars)
        else:
            selected = self._rank_and_pack(query_tokens, budget_chars)

        # Restore chronology after relevance selection so the model reads a
        # coherent evidence trace.
        selected = sorted(selected)
        context = "\n\n".join(self.memory[idx] for idx in selected)
        return context, selected, budget_chars, {
            "retrieved_context_chars": len(context),
            "did_filter": False,
            "filtered_sentence_count": 0,
        }

    def _candidate_indices(self, query: str, query_tokens: set[str]) -> list[int]:
        top_k = min(self.filter_top_k, len(self.memory))
        if top_k <= 0:
            return []
        if self.strategy == "recent":
            return list(range(max(0, len(self.memory) - top_k), len(self.memory)))
        if self.strategy == "random":
            scored = []
            for idx in range(len(self.memory)):
                key = f"{self.random_seed}|{query}|{idx}".encode("utf-8")
                score = int(hashlib.sha256(key).hexdigest()[:16], 16)
                scored.append((score, idx))
            scored.sort(key=lambda item: (-item[0], item[1]))
            return [idx for _, idx in scored[:top_k]]
        if self.strategy == "mmr":
            return self._mmr_top_indices(query_tokens, top_k)
        return self._rank_top_indices(query_tokens, top_k, use_recency=(self.strategy == "hybrid"))

    def _rank_top_indices(self, query_tokens: set[str], top_k: int, use_recency: bool = False) -> list[int]:
        scored = []
        for idx, counts in enumerate(self._doc_tokens):
            score = self._bm25_score(query_tokens, counts)
            if use_recency:
                recency = idx / max(1, len(self._doc_tokens) - 1)
                score += self.hybrid_recency_alpha * recency
            scored.append((score, idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _, idx in scored[:top_k]]

    def _mmr_top_indices(self, query_tokens: set[str], top_k: int) -> list[int]:
        if not query_tokens:
            return list(range(max(0, len(self.memory) - top_k), len(self.memory)))
        relevance = [
            self._bm25_score(query_tokens, counts)
            for counts in self._doc_tokens
        ]
        candidates = set(range(len(self.memory)))
        selected: list[int] = []
        while candidates and len(selected) < top_k:
            best_idx = None
            best_score = None
            for idx in candidates:
                diversity_penalty = 0.0
                if selected:
                    diversity_penalty = max(
                        self._token_jaccard(self._doc_tokens[idx], self._doc_tokens[prev])
                        for prev in selected
                    )
                mmr_score = self.mmr_lambda * relevance[idx] - (1.0 - self.mmr_lambda) * diversity_penalty
                if best_score is None or mmr_score > best_score or (mmr_score == best_score and idx < best_idx):
                    best_score = mmr_score
                    best_idx = idx
            if best_idx is None:
                break
            candidates.remove(best_idx)
            selected.append(best_idx)
        return selected

    def _filter_sentences(self, candidate_indices: list[int], query_tokens: set[str], budget_chars: int) -> tuple[str, int]:
        sentence_records: list[tuple[float, int, str]] = []
        for idx in candidate_indices:
            for sentence_idx, sentence in enumerate(self._sentences(self.memory[idx])):
                counts = Counter(self._tokenize(sentence))
                score = self._bm25_score(query_tokens, counts)
                if score > 0:
                    sentence_records.append((score, idx * 10000 + sentence_idx, sentence))
        if not sentence_records:
            return self._pack_indices_to_budget(sorted(candidate_indices), budget_chars), 0
        sentence_records.sort(key=lambda row: (-row[0], row[1]))
        selected: list[tuple[int, str]] = []
        used = 0
        for _, order, sentence in sentence_records:
            sentence_len = len(sentence) + 1
            if selected and used + sentence_len > budget_chars:
                continue
            selected.append((order, sentence))
            used += sentence_len
        selected.sort(key=lambda row: row[0])
        return "\n".join(sentence for _, sentence in selected) or "No previous memory", len(selected)

    def _pack_indices_to_budget(self, indices: list[int], budget_chars: int) -> str:
        selected: list[int] = []
        used = 0
        for idx in indices:
            chunk_len = len(self.memory[idx]) + 2
            if selected and used + chunk_len > budget_chars:
                continue
            selected.append(idx)
            used += chunk_len
        return "\n\n".join(self.memory[idx] for idx in selected) or "No previous memory"

    def _pack_recent(self, budget_chars: int) -> list[int]:
        selected: list[int] = []
        used = 0
        for idx in range(len(self.memory) - 1, -1, -1):
            chunk_len = len(self.memory[idx]) + 2
            if selected and used + chunk_len > budget_chars:
                continue
            selected.append(idx)
            used += chunk_len
        return selected

    def _pack_random(self, query: str, budget_chars: int) -> list[int]:
        scored = []
        for idx in range(len(self.memory)):
            key = f"{self.random_seed}|{query}|{idx}".encode("utf-8")
            score = int(hashlib.sha256(key).hexdigest()[:16], 16)
            scored.append((score, idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected: list[int] = []
        used = 0
        for _, idx in scored:
            chunk_len = len(self.memory[idx]) + 2
            if selected and used + chunk_len > budget_chars:
                continue
            selected.append(idx)
            used += chunk_len
        return selected

    def _rank_and_pack(self, query_tokens: set[str], budget_chars: int, use_recency: bool = False) -> list[int]:
        scored = []
        for idx, counts in enumerate(self._doc_tokens):
            score = self._bm25_score(query_tokens, counts)
            if use_recency:
                recency = idx / max(1, len(self._doc_tokens) - 1)
                score += self.hybrid_recency_alpha * recency
            if score > 0:
                scored.append((score, idx))

        if not scored:
            scored = [(0.0, idx) for idx in range(len(self.memory))]

        scored.sort(key=lambda item: (-item[0], item[1]))

        selected: list[int] = []
        used = 0
        for _, idx in scored:
            chunk_len = len(self.memory[idx]) + 2
            if selected and used + chunk_len > budget_chars:
                continue
            if chunk_len > budget_chars and not selected:
                selected.append(idx)
                break
            selected.append(idx)
            used += chunk_len

        if self.min_recent_chunks > 0:
            selected_set = set(selected)
            for idx in range(max(0, len(self.memory) - self.min_recent_chunks), len(self.memory)):
                selected_set.add(idx)
            selected = self._trim_to_budget(sorted(selected_set), budget_chars)

        return selected

    def _mmr_pack(self, query_tokens: set[str], budget_chars: int) -> list[int]:
        if not query_tokens:
            return self._pack_recent(budget_chars)

        relevance = [
            self._bm25_score(query_tokens, counts)
            for counts in self._doc_tokens
        ]
        candidates = {idx for idx, score in enumerate(relevance) if score > 0}
        if not candidates:
            return self._pack_recent(budget_chars)

        selected: list[int] = []
        used = 0
        while candidates:
            best_idx = None
            best_score = None
            for idx in candidates:
                diversity_penalty = 0.0
                if selected:
                    diversity_penalty = max(
                        self._token_jaccard(self._doc_tokens[idx], self._doc_tokens[prev])
                        for prev in selected
                    )
                mmr_score = self.mmr_lambda * relevance[idx] - (1.0 - self.mmr_lambda) * diversity_penalty
                if best_score is None or mmr_score > best_score or (mmr_score == best_score and idx < best_idx):
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is None:
                break
            candidates.remove(best_idx)
            chunk_len = len(self.memory[best_idx]) + 2
            if selected and used + chunk_len > budget_chars:
                continue
            if chunk_len > budget_chars and not selected:
                selected.append(best_idx)
                break
            selected.append(best_idx)
            used += chunk_len

        return selected

    def _trim_to_budget(self, indices: list[int], budget_chars: int) -> list[int]:
        # Prefer keeping already selected relevant chunks. If forced to trim,
        # drop the longest chunks first while preserving at least one chunk.
        while len(indices) > 1 and sum(len(self.memory[i]) + 2 for i in indices) > budget_chars:
            longest = max(indices, key=lambda i: len(self.memory[i]))
            indices.remove(longest)
        return indices

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
    def _token_jaccard(left: Counter[str], right: Counter[str]) -> float:
        left_keys = set(left)
        right_keys = set(right)
        if not left_keys and not right_keys:
            return 0.0
        return len(left_keys & right_keys) / max(1, len(left_keys | right_keys))

    def _context_budget_chars(self, query: str) -> int:
        # Match ConcatAgent's rough 16k-window character budget formula, but
        # make it explicit and configurable for later sweeps.
        query_chars = len(QA_PROMPT.format(query))
        reserved_for_answer = self.answer_max_tokens * 3
        return max(1000, self.max_prompt_chars - reserved_for_answer - query_chars - 200)

    def _should_write(self, chunk: str) -> bool:
        tokens = self._query_terms(chunk)
        if len(tokens) < self.min_informative_tokens:
            return False
        if len(chunk.strip()) < 20:
            return False
        return True

    def _write_trace(self, query: str, selected_indices: list[int], budget_chars: int, context: str, extra: dict | None = None) -> None:
        if not self.trace_path:
            return
        extra = extra or {}
        record = {
            "strategy": self.strategy,
            "query_sha1": hashlib.sha1(query.encode("utf-8")).hexdigest(),
            "query_prefix": query[:200],
            "n_memory_chunks": len(self.memory),
            "n_selected_chunks": len(selected_indices),
            "selected_indices": selected_indices,
            "budget_chars": budget_chars,
            "selected_context_chars": len(context),
            "total_memory_chars": sum(len(chunk) + 2 for chunk in self.memory),
            "hybrid_recency_alpha": self.hybrid_recency_alpha,
            "mmr_lambda": self.mmr_lambda,
            "sentence_filter": self.sentence_filter,
            "filter_top_k": self.filter_top_k,
            "topk_only": self.topk_only,
            "conditional_filter": self.conditional_filter,
            "adaptive_store": self.adaptive_store,
            "retrieved_context_chars": extra.get("retrieved_context_chars"),
            "did_filter": extra.get("did_filter"),
            "filtered_sentence_count": extra.get("filtered_sentence_count"),
        }
        os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
        with open(self.trace_path, "a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        return {
            token
            for token in BudgetedEvidenceAgent._tokenize(text)
            if len(token) > 2 and token not in _STOPWORDS
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+", text.lower())

    @staticmethod
    def _sentences(text: str) -> list[str]:
        pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [piece.strip() for piece in pieces if piece.strip()]


_STOPWORDS = {
    "the", "and", "for", "that", "this", "what", "who", "whom", "whose",
    "when", "where", "which", "why", "how", "are", "was", "were", "has",
    "had", "have", "does", "did", "from", "with", "about", "into", "over",
    "under", "between", "after", "before", "also", "not", "you", "your",
    "their", "there", "they", "them", "his", "her", "its", "our", "out",
    "all", "any", "can", "could", "would", "should", "than", "then",
}

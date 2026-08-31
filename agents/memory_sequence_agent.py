"""State-local memory operation sequence probe.

This agent is an experimental instrument for testing whether memory
computation is genuinely sequential rather than a one-shot contextual bandit.

Every query first builds the same initial retrieved evidence state:

    RETRIEVE -> E_0

Then ``AMC_SEQUENCE`` chooses optional memory operations before answering:

    stop              : ANSWER(E_0)
    refine            : REFINE(E_0) -> ANSWER
    expand            : EXPAND(E_0) -> ANSWER
    refine_expand     : REFINE(E_0) -> EXPAND -> ANSWER
    expand_refine     : EXPAND(E_0) -> REFINE -> ANSWER

The operations use only online-visible state: question text/date, retrieved
memory text, session DATE headers, and the long-term memory itself.  Gold
answer/evidence labels remain offline-only diagnostics.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter

from .adaptive_memory_agent import MemoryItem, _STOPWORDS
from .concat_agent import QA_PROMPT
from .progressive_depth_agent import ProgressiveDepthAgent


class MemorySequenceAgent(ProgressiveDepthAgent):
    """Probe fixed memory-operation sequences from the same retrieved state."""

    _ALIASES = {
        "s": "stop",
        "stop": "stop",
        "none": "stop",
        "r": "refine",
        "refine": "refine",
        "e": "expand",
        "expand": "expand",
        "re": "refine_expand",
        "r_e": "refine_expand",
        "refine_expand": "refine_expand",
        "refine->expand": "refine_expand",
        "er": "expand_refine",
        "e_r": "expand_refine",
        "expand_refine": "expand_refine",
        "expand->refine": "expand_refine",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raw_sequence = os.getenv("AMC_SEQUENCE", "stop").strip().lower()
        self.sequence = self._ALIASES.get(raw_sequence)
        if self.sequence is None:
            raise ValueError(
                "AMC_SEQUENCE must be one of stop/refine/expand/refine_expand/expand_refine, "
                f"got {raw_sequence}"
            )
        self.expand_k = int(os.getenv("AMC_EXPAND_K", "20"))
        self.expand_mode = os.getenv("AMC_EXPAND_MODE", "state_conditioned").strip().lower()
        if self.expand_mode not in {"state_conditioned", "bm25_tail"}:
            raise ValueError(f"AMC_EXPAND_MODE must be state_conditioned or bm25_tail, got {self.expand_mode}")

    async def QA_async(self, query: str) -> str:
        started = time.time()
        query_tokens = self._query_terms(query)
        budget_chars = self.budget.context_budget_chars(query)
        operations: list[str] = ["RETRIEVE"]
        op_records: list[dict] = []
        context = "No previous memory"
        working_items: list[MemoryItem] = []
        working_admitted_indices: set[int] = set()
        seen_indices: set[int] = set()

        try:
            if not self.items:
                operations = ["ANSWER"]
            else:
                working_items = self._retrieve_k(query_tokens, self.top_k)
                seen_indices.update(item.idx for item in working_items)
                context, admitted_items = self._pack_items_to_budget_with_items(working_items, budget_chars)
                working_admitted_indices = {item.idx for item in admitted_items}
                op_records.append(
                    self._op_record(
                        "RETRIEVE",
                        working_items,
                        context,
                        budget_chars,
                        {
                            **self._retrieval_state(query_tokens, working_items),
                            "sequence": self.sequence,
                            "filter_mode": self.filter_mode,
                            "expand_mode": self.expand_mode,
                        },
                        admitted_items=admitted_items,
                        retrieved_items=working_items,
                    )
                )

                for action in self._sequence_actions():
                    if action == "REFINE":
                        context, count, stats = self._filter_by_mode(
                            working_items,
                            query,
                            query_tokens,
                            budget_chars,
                        )
                        admitted = set(stats.get("admitted_source_indices") or [])
                        if admitted:
                            working_items = [item for item in working_items if item.idx in admitted]
                            working_admitted_indices = admitted
                        operations.append("REFINE")
                        op_records.append(
                            self._op_record(
                                "REFINE",
                                working_items,
                                context,
                                budget_chars,
                                {
                                    "filtered_sentence_count": count,
                                    "filter_mode": self.filter_mode,
                                    **stats,
                                },
                            )
                        )
                    elif action == "EXPAND":
                        more_items, expand_stats = self._expand_items(
                            query,
                            query_tokens,
                            context,
                            working_items,
                            seen_indices,
                            self.expand_k,
                        )
                        seen_indices.update(item.idx for item in more_items)
                        context, merge_stats = self._merge_filtered_and_more(context, more_items, budget_chars)
                        working_admitted_indices = working_admitted_indices | set(
                            merge_stats.get("admitted_more_indices") or []
                        )
                        working_items = self._items_by_indices(
                            [item.idx for item in working_items] + [item.idx for item in more_items]
                        )
                        operations.append("EXPAND")
                        op_records.append(
                            self._op_record(
                                "EXPAND",
                                working_items,
                                context,
                                budget_chars,
                                {
                                    "expand_k": self.expand_k,
                                    "expand_mode": self.expand_mode,
                                    "admitted_source_indices": sorted(working_admitted_indices),
                                    "n_admitted_sources": len(working_admitted_indices),
                                    **expand_stats,
                                    **merge_stats,
                                },
                                retrieved_items=working_items,
                            )
                        )

            operations.append("ANSWER")
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

    def _sequence_actions(self) -> list[str]:
        if self.sequence == "stop":
            return []
        if self.sequence == "refine":
            return ["REFINE"]
        if self.sequence == "expand":
            return ["EXPAND"]
        if self.sequence == "refine_expand":
            return ["REFINE", "EXPAND"]
        if self.sequence == "expand_refine":
            return ["EXPAND", "REFINE"]
        raise AssertionError(self.sequence)

    def _expand_items(
        self,
        query: str,
        query_tokens: set[str],
        context: str,
        working_items: list[MemoryItem],
        exclude: set[int],
        k: int,
    ) -> tuple[list[MemoryItem], dict]:
        if self.expand_mode == "bm25_tail":
            items = self._retrieve_k(query_tokens, k, exclude=exclude)
            return items, {
                "operator_family": "expand_bm25_tail",
                "expanded_indices": [item.idx for item in items],
                "n_expanded": len(items),
            }

        query_dt = self._parse_query_date(query)
        context_entities = self._entity_phrases(context)
        query_entities = self._entity_phrases(query)
        anchor_entities = context_entities | query_entities
        context_tokens = Counter(self._tokenize(context))
        anchor_tokens = {
            token
            for token, _ in context_tokens.most_common(80)
            if len(token) > 2 and token not in _STOPWORDS
        }
        scored: list[tuple[float, int, MemoryItem, dict]] = []
        raw_query_scores: list[float] = []
        for item in self.items:
            if item.idx in exclude:
                continue
            raw_query_scores.append(self._bm25_score(query_tokens, item.tokens))
        max_query_score = max(raw_query_scores, default=0.0) or 1.0

        for item in self.items:
            if item.idx in exclude:
                continue
            query_score = self._bm25_score(query_tokens, item.tokens) / max_query_score
            item_entities = self._entity_phrases(item.text)
            entity_link = len(item_entities & anchor_entities) / max(1, min(8, len(anchor_entities) or 1))
            token_link = len(set(item.tokens) & anchor_tokens) / max(1, min(30, len(anchor_tokens) or 1))
            temporal_score = 0.0
            session_dt = self._parse_session_date(item.text)
            if query_dt is not None and session_dt is not None:
                days = (query_dt - session_dt).days
                temporal_score = 1.0 / (1.0 + max(0, days) / 30.0) if days >= 0 else -0.25
            structure = max(self._sentence_structure_bonus(sentence) for sentence in self._sentences(item.text)[:20] or [""])
            score = 0.50 * query_score + 0.20 * entity_link + 0.15 * token_link + 0.10 * temporal_score + 0.05 * structure
            scored.append(
                (
                    score,
                    item.idx,
                    item,
                    {
                        "query_score": query_score,
                        "entity_link": entity_link,
                        "token_link": token_link,
                        "temporal_score": temporal_score,
                        "structure": structure,
                    },
                )
            )
        scored.sort(key=lambda row: (-row[0], row[1]))
        selected = [item for _, _, item, _ in scored[:k]]
        return selected, {
            "operator_family": "expand_state_conditioned",
            "expanded_indices": [item.idx for item in selected],
            "n_expanded": len(selected),
            "n_expand_candidates": len(scored),
            "expand_top_scores": [
                {"idx": idx, "score": score, **parts}
                for score, idx, _, parts in scored[: min(5, len(scored))]
            ],
        }

    def _items_by_indices(self, indices: list[int]) -> list[MemoryItem]:
        wanted = set(indices)
        by_idx = {item.idx: item for item in self.items if item.idx in wanted}
        seen: set[int] = set()
        out: list[MemoryItem] = []
        for idx in indices:
            if idx in by_idx and idx not in seen:
                out.append(by_idx[idx])
                seen.add(idx)
        return out

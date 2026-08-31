"""Progressive depth agent for nested memory-computation experiments.

This agent is not a new final method.  It is a controlled experimental
instrument for Adaptive Progressive Memory Computation (APMC):

    D0: ANSWER
    D1: RETRIEVE -> ANSWER
    D2: RETRIEVE -> FILTER -> ANSWER
    D3 strict: RETRIEVE(K) -> FILTER -> RETRIEVE_MORE(excluding K) -> ANSWER

The key purpose is to generate clean per-depth outcomes and post-operation
state traces so we can later test whether intermediate memory states predict
when additional computation helps.
"""

from __future__ import annotations

from datetime import datetime
import math
import os
import re
import time
from collections import Counter

from .adaptive_memory_agent import AdaptiveMemoryAgent, MemoryItem
from .concat_agent import QA_PROMPT


class ProgressiveDepthAgent(AdaptiveMemoryAgent):
    """Nested-depth memory computation probe."""

    def __init__(
        self,
        client=None,
        model_name: str = "gpt4.1",
        max_prompt_chars: int | None = None,
        answer_max_tokens: int | None = None,
        top_k: int | None = None,
    ):
        super().__init__(
            client=client,
            model_name=model_name,
            max_prompt_chars=max_prompt_chars,
            answer_max_tokens=answer_max_tokens,
            top_k=top_k,
        )
        self.depth = int(os.getenv("AMC_DEPTH", "1"))
        if self.depth not in {0, 1, 2, 3}:
            raise ValueError(f"AMC_DEPTH must be one of 0/1/2/3, got {self.depth}")
        self.k1 = int(os.getenv("AMC_D3_K1", "10"))
        self.k2 = int(os.getenv("AMC_D3_K2", "30"))
        self.d3_protocol = os.getenv("AMC_D3_PROTOCOL", "strict").strip().lower()
        if self.d3_protocol not in {"strict", "legacy"}:
            raise ValueError(f"AMC_D3_PROTOCOL must be strict or legacy, got {self.d3_protocol}")
        self.filter_mode = os.getenv("AMC_FILTER_MODE", "lexical_bm25").strip().lower()
        if self.filter_mode not in {"lexical_bm25", "tfidf_jaccard", "graph_bridge", "temporal_session"}:
            raise ValueError(
                "AMC_FILTER_MODE must be lexical_bm25, tfidf_jaccard, graph_bridge, or temporal_session, "
                f"got {self.filter_mode}"
            )

    async def QA_async(self, query: str) -> str:
        started = time.time()
        query_tokens = self._query_terms(query)
        budget_chars = self.budget.context_budget_chars(query)
        operations: list[str] = []
        op_records: list[dict] = []
        selected_items: list[MemoryItem] = []
        context = "No previous memory"

        try:
            if self.depth == 0 or not self.items:
                operations = ["ANSWER"]
            elif self.depth == 1:
                operations = ["RETRIEVE", "ANSWER"]
                selected_items = self._retrieve_k(query_tokens, self.top_k)
                context, admitted_items = self._pack_items_to_budget_with_items(selected_items, budget_chars)
                op_records.append(
                    self._op_record(
                        "RETRIEVE",
                        selected_items,
                        context,
                        budget_chars,
                        {
                            **self._retrieval_state(query_tokens, selected_items),
                            "d3_protocol": self.d3_protocol,
                            "filter_mode": self.filter_mode,
                        },
                        admitted_items=admitted_items,
                        retrieved_items=selected_items,
                    )
                )
            elif self.depth == 2:
                operations = ["RETRIEVE", "FILTER", "ANSWER"]
                selected_items = self._retrieve_k(query_tokens, self.top_k)
                retrieved_context, admitted_items = self._pack_items_to_budget_with_items(selected_items, budget_chars)
                op_records.append(
                    self._op_record(
                        "RETRIEVE",
                        selected_items,
                        retrieved_context,
                        budget_chars,
                        {
                            **self._retrieval_state(query_tokens, selected_items),
                            "d3_protocol": self.d3_protocol,
                            "filter_mode": self.filter_mode,
                        },
                        admitted_items=admitted_items,
                        retrieved_items=selected_items,
                    )
                )
                context, filtered_sentence_count, filter_stats = self._filter_by_mode(selected_items, query, query_tokens, budget_chars)
                op_records.append(
                    self._op_record(
                        "FILTER",
                        selected_items,
                        context,
                        budget_chars,
                        {
                            "did_filter": True,
                            "filtered_sentence_count": filtered_sentence_count,
                            "filter_mode": self.filter_mode,
                            **filter_stats,
                        },
                    )
                )
            else:
                operations = ["RETRIEVE", "FILTER", "RETRIEVE_MORE", "ANSWER"]
                if self.d3_protocol == "legacy":
                    first_items = self._retrieve_k(query_tokens, self.k1)
                else:
                    first_items = self._retrieve_k(query_tokens, self.top_k)
                first_context, first_admitted_items = self._pack_items_to_budget_with_items(first_items, budget_chars)
                op_records.append(
                    self._op_record(
                        "RETRIEVE",
                        first_items,
                        first_context,
                        budget_chars,
                        {
                            **self._retrieval_state(query_tokens, first_items),
                            "d3_protocol": self.d3_protocol,
                            "filter_mode": self.filter_mode,
                        },
                        admitted_items=first_admitted_items,
                        retrieved_items=first_items,
                    )
                )
                filtered_context, filtered_sentence_count, filter_stats = self._filter_by_mode(first_items, query, query_tokens, budget_chars)
                op_records.append(
                    self._op_record(
                        "FILTER",
                        first_items,
                        filtered_context,
                        budget_chars,
                        {
                            "did_filter": True,
                            "filtered_sentence_count": filtered_sentence_count,
                            "filter_mode": self.filter_mode,
                            **filter_stats,
                        },
                    )
                )
                more_items = self._retrieve_k(
                    query_tokens,
                    self.k2,
                    exclude={item.idx for item in first_items},
                )
                context, merge_stats = self._merge_filtered_and_more(filtered_context, more_items, budget_chars)
                final_admitted_indices = sorted(
                    set(filter_stats.get("admitted_source_indices") or [])
                    | set(merge_stats.get("admitted_more_indices") or [])
                )
                selected_items = first_items + more_items
                op_records.append(
                    self._op_record(
                        "RETRIEVE_MORE",
                        selected_items,
                        context,
                        budget_chars,
                        {
                            "d3_protocol": self.d3_protocol,
                            "k1": self.k1 if self.d3_protocol == "legacy" else self.top_k,
                            "k2": self.k2,
                            "n_more_retrieved": len(more_items),
                            "more_chars_retrieved": sum(item.n_chars + 2 for item in more_items),
                            "filter_mode": self.filter_mode,
                            "admitted_source_indices": final_admitted_indices,
                            "n_admitted_sources": len(final_admitted_indices),
                            **merge_stats,
                        },
                        retrieved_items=selected_items,
                    )
                )

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

    def _retrieve_k(
        self,
        query_tokens: set[str],
        k: int,
        exclude: set[int] | None = None,
    ) -> list[MemoryItem]:
        exclude = exclude or set()
        if not query_tokens:
            recent = self._retrieve_recent(self.budget.context_budget_chars(""))
            return [item for item in recent if item.idx not in exclude][:k]

        scored = []
        for item in self.items:
            if item.idx in exclude:
                continue
            score = self._bm25_score(query_tokens, item.tokens)
            if score > 0:
                scored.append((score, item.idx, item))
        if not scored:
            recent = self._retrieve_recent(self.budget.context_budget_chars(""))
            return [item for item in recent if item.idx not in exclude][:k]
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [item for _, _, item in scored[:k]]

    def _filter_by_mode(
        self,
        items: list[MemoryItem],
        query: str,
        query_tokens: set[str],
        budget_chars: int,
    ) -> tuple[str, int, dict]:
        if self.filter_mode == "tfidf_jaccard":
            return self._filter_tfidf_jaccard(items, query_tokens, budget_chars)
        if self.filter_mode == "graph_bridge":
            return self._filter_graph_bridge(items, query_tokens, budget_chars)
        if self.filter_mode == "temporal_session":
            return self._filter_temporal_session(items, query, query_tokens, budget_chars)
        context, count, admitted_source_indices = self._filter_with_source_indices(items, query_tokens, budget_chars)
        return context, count, {
            "operator_family": "lexical_extract",
            "requires_query_overlap": True,
            "zero_query_overlap_sentences_admitted": 0,
            "admitted_source_indices": admitted_source_indices,
            "n_admitted_sources": len(admitted_source_indices),
        }

    def _filter_tfidf_jaccard(
        self,
        items: list[MemoryItem],
        query_tokens: set[str],
        budget_chars: int,
    ) -> tuple[str, int, dict]:
        """Deterministic semantic-lite refinement operator.

        This is not a dense embedding model. It is a second, independently
        parameterized sentence selector for operation-robustness auditing. It
        mixes IDF-weighted overlap, query coverage, Jaccard overlap, and a small
        structure prior for entity/date/relation-bearing sentences.
        """
        if not query_tokens:
            return self._pack_items_to_budget(items, budget_chars), 0, {
                "operator_family": "tfidf_jaccard",
                "requires_query_overlap": True,
                "zero_query_overlap_sentences_admitted": 0,
                "admitted_source_indices": [item.idx for item in items],
                "n_admitted_sources": len(items),
            }

        query_vec = {token: self._idf.get(token, 1.0) for token in query_tokens}
        query_norm = math.sqrt(sum(weight * weight for weight in query_vec.values())) or 1.0
        sentence_records: list[tuple[float, int, str]] = []

        for item in items:
            for sentence_idx, sentence in enumerate(self._sentences(item.text)):
                sentence_tokens = Counter(self._tokenize(sentence))
                if not sentence_tokens:
                    continue
                overlap = set(sentence_tokens) & query_tokens
                if not overlap:
                    continue

                sentence_vec = {
                    token: count * self._idf.get(token, 1.0)
                    for token, count in sentence_tokens.items()
                }
                sentence_norm = math.sqrt(sum(weight * weight for weight in sentence_vec.values())) or 1.0
                dot = sum(query_vec[token] * sentence_vec.get(token, 0.0) for token in query_tokens)
                cosine = dot / (query_norm * sentence_norm)
                coverage = len(overlap) / max(1, len(query_tokens))
                jaccard = len(overlap) / max(1, len(set(sentence_tokens) | query_tokens))
                structure = self._sentence_structure_bonus(sentence)
                score = 0.55 * cosine + 0.25 * coverage + 0.15 * jaccard + 0.05 * structure
                sentence_records.append((score, item.idx * 10000 + sentence_idx, sentence))

        if not sentence_records:
            return self._pack_items_to_budget(items, budget_chars), 0, {
                "operator_family": "tfidf_jaccard",
                "requires_query_overlap": True,
                "zero_query_overlap_sentences_admitted": 0,
                "admitted_source_indices": [item.idx for item in items],
                "n_admitted_sources": len(items),
            }

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
        admitted_sources = sorted({order // 10000 for order, _ in selected})
        return "\n".join(sentence for _, sentence in selected), len(selected), {
            "operator_family": "tfidf_jaccard",
            "requires_query_overlap": True,
            "zero_query_overlap_sentences_admitted": 0,
            "admitted_source_indices": admitted_sources,
            "n_admitted_sources": len(admitted_sources),
        }

    def _filter_graph_bridge(
        self,
        items: list[MemoryItem],
        query_tokens: set[str],
        budget_chars: int,
    ) -> tuple[str, int, dict]:
        """Bridge-aware deterministic refinement without query-overlap gating.

        This is still a lightweight, no-download operator. Its purpose is to
        test whether the large D2 drop is caused by lexical overlap gating. It
        scores every sentence from the retrieved set using both query relevance
        and evidence-to-evidence connectivity, so a bridge sentence can survive
        even when it shares no lexical token with the query.
        """
        sentence_rows: list[dict] = []
        entity_freq: Counter[str] = Counter()
        for item in items:
            for sentence_idx, sentence in enumerate(self._sentences(item.text)):
                tokens = Counter(self._tokenize(sentence))
                if not tokens:
                    continue
                entities = self._entity_phrases(sentence)
                entity_freq.update(entities)
                sentence_rows.append(
                    {
                        "order": item.idx * 10000 + sentence_idx,
                        "sentence": sentence,
                        "tokens": tokens,
                        "entities": entities,
                        "query_score": self._bm25_score(query_tokens, tokens),
                        "query_overlap": len(set(tokens) & query_tokens),
                        "structure": self._sentence_structure_bonus(sentence),
                    }
                )

        if not sentence_rows:
            return self._pack_items_to_budget(items, budget_chars), 0, {
                "operator_family": "graph_bridge",
                "requires_query_overlap": False,
                "zero_query_overlap_sentences_admitted": 0,
                "admitted_source_indices": [item.idx for item in items],
                "n_admitted_sources": len(items),
            }

        n_candidate_sentences_raw = len(sentence_rows)
        max_sentences = int(os.getenv("AMC_GRAPH_MAX_SENTENCES", "0"))
        was_pruned = False
        if max_sentences > 0 and len(sentence_rows) > max_sentences:
            # LongMemEval sessions can contain thousands of sentences. The
            # bridge-connectivity pass below is quadratic, so we optionally
            # preselect candidates with online-visible signals only: query
            # relevance, structural evidence markers, and local entity count.
            sentence_rows.sort(
                key=lambda row: (
                    -row["query_score"],
                    -row["structure"],
                    -len(row["entities"]),
                    row["order"],
                )
            )
            sentence_rows = sentence_rows[:max_sentences]
            sentence_rows.sort(key=lambda row: row["order"])
            was_pruned = True

        max_query_score = max((row["query_score"] for row in sentence_rows), default=0.0) or 1.0
        query_entities = self._entity_phrases(" ".join(query_tokens))
        for i, row in enumerate(sentence_rows):
            max_token_link = 0.0
            max_entity_link = 0.0
            for j, other in enumerate(sentence_rows):
                if i == j:
                    continue
                max_token_link = max(max_token_link, self._token_jaccard(row["tokens"], other["tokens"]))
                shared_entities = set(row["entities"]) & set(other["entities"])
                if shared_entities:
                    max_entity_link = max(max_entity_link, min(1.0, len(shared_entities) / 2.0))
            central_entity_bonus = 0.0
            if row["entities"]:
                central_entity_bonus = min(
                    1.0,
                    max(entity_freq[entity] for entity in row["entities"]) / max(2.0, len(sentence_rows) ** 0.5),
                )
            query_entity_bonus = 1.0 if set(row["entities"]) & set(query_entities) else 0.0
            normalized_query = row["query_score"] / max_query_score
            connectivity = 0.55 * max_entity_link + 0.30 * max_token_link + 0.15 * central_entity_bonus
            row["bridge_score"] = (
                0.45 * normalized_query
                + 0.35 * connectivity
                + 0.10 * row["structure"]
                + 0.10 * query_entity_bonus
            )

        sentence_rows.sort(key=lambda row: (-row["bridge_score"], row["order"]))
        selected: list[tuple[int, str, int]] = []
        used = 0
        for row in sentence_rows:
            sentence = row["sentence"]
            sentence_len = len(sentence) + 1
            if selected and used + sentence_len > budget_chars:
                continue
            selected.append((row["order"], sentence, row["query_overlap"]))
            used += sentence_len

        selected.sort(key=lambda row: row[0])
        zero_overlap_admitted = sum(1 for _, _, overlap in selected if overlap == 0)
        admitted_sources = sorted({order // 10000 for order, _, _ in selected})
        return "\n".join(sentence for _, sentence, _ in selected), len(selected), {
            "operator_family": "graph_bridge",
            "requires_query_overlap": False,
            "zero_query_overlap_sentences_admitted": zero_overlap_admitted,
            "candidate_sentence_count": len(sentence_rows),
            "candidate_sentence_count_raw": n_candidate_sentences_raw,
            "graph_preselect_max_sentences": max_sentences,
            "graph_preselect_applied": was_pruned,
            "selected_context_chars": used,
            "admitted_source_indices": admitted_sources,
            "n_admitted_sources": len(admitted_sources),
        }

    def _filter_temporal_session(
        self,
        items: list[MemoryItem],
        query: str,
        query_tokens: set[str],
        budget_chars: int,
    ) -> tuple[str, int, dict]:
        """Session-level temporal refinement for timestamped memory benchmarks.

        This primitive uses only online-visible information: question text,
        question date if present in the query, session DATE headers, and
        retrieved session text. It does not use answer_session_ids,
        question_type, or has_answer labels.
        """
        query_dt = self._parse_query_date(query)
        has_temporal_cue = bool(
            re.search(
                r"\b(?:latest|recent|current|currently|now|before|after|earlier|later|first|last|newest|oldest|when|date|time|updated?)\b",
                query,
                re.I,
            )
        )
        scored: list[tuple[float, int, MemoryItem, datetime | None]] = []
        base_scores = [self._bm25_score(query_tokens, item.tokens) for item in items]
        max_base = max(base_scores, default=0.0) or 1.0
        dated_count = 0
        for item, base in zip(items, base_scores):
            session_dt = self._parse_session_date(item.text)
            if session_dt is not None:
                dated_count += 1
            recency_score = 0.0
            if query_dt is not None and session_dt is not None:
                days = (query_dt - session_dt).days
                if days >= 0:
                    recency_score = 1.0 / (1.0 + max(0, days) / 30.0)
                else:
                    recency_score = -0.25
            normalized_relevance = base / max_base
            temporal_weight = 0.45 if has_temporal_cue else 0.20
            score = (1.0 - temporal_weight) * normalized_relevance + temporal_weight * recency_score
            scored.append((score, item.idx, item, session_dt))
        scored.sort(key=lambda row: (-row[0], row[1]))

        selected: list[tuple[int, MemoryItem, datetime | None]] = []
        used = 0
        for _, _, item, session_dt in scored:
            item_len = item.n_chars + 2
            if selected and used + item_len > budget_chars:
                continue
            selected.append((item.idx, item, session_dt))
            used += item_len
        # Preserve temporal order in the final reader context when dates exist.
        selected.sort(key=lambda row: (row[2] is None, row[2] or datetime.max, row[0]))
        selected_items = [item for _, item, _ in selected]
        context = "\n\n".join(item.text for item in selected_items) or "No previous memory"
        return context, len(selected_items), {
            "operator_family": "temporal_session_refine",
            "requires_query_overlap": False,
            "has_temporal_cue": has_temporal_cue,
            "query_date_parsed": query_dt.isoformat() if query_dt else None,
            "dated_session_count": dated_count,
            "admitted_source_indices": [item.idx for item in selected_items],
            "n_admitted_sources": len(selected_items),
            "selected_context_chars": used,
        }

    @staticmethod
    def _entity_phrases(text: str) -> set[str]:
        phrases = set(re.findall(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4}\b", text))
        return {phrase.lower() for phrase in phrases if len(phrase) > 2}

    @staticmethod
    def _parse_query_date(query: str) -> datetime | None:
        match = re.match(r"\[([^\]]+)\]", query.strip())
        if not match:
            return None
        return ProgressiveDepthAgent._parse_date(match.group(1))

    @staticmethod
    def _parse_session_date(text: str) -> datetime | None:
        match = re.search(r"^DATE:\s*([^\n]+)", text, re.M)
        if not match:
            return None
        return ProgressiveDepthAgent._parse_date(match.group(1))

    @staticmethod
    def _parse_date(text: str) -> datetime | None:
        candidates = re.findall(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", text)
        for candidate in candidates:
            normalized = candidate.replace("-", "/")
            try:
                return datetime.strptime(normalized, "%Y/%m/%d")
            except ValueError:
                continue
        return None

    @staticmethod
    def _sentence_structure_bonus(sentence: str) -> float:
        has_capitalized_phrase = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", sentence))
        has_date_or_number = bool(re.search(r"\b(?:18|19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}", sentence))
        has_relation_marker = bool(
            re.search(
                r"\b(?:born|directed|located|member|joined|founded|married|worked|served|held|capital|headquarters)\b",
                sentence,
                re.I,
            )
        )
        return (
            float(has_capitalized_phrase)
            + float(has_date_or_number)
            + float(has_relation_marker)
        ) / 3.0

    def _retrieval_state(self, query_tokens: set[str], items: list[MemoryItem]) -> dict:
        scores = [self._bm25_score(query_tokens, item.tokens) for item in items]
        positive = [score for score in scores if score > 0]
        total = sum(positive)
        if total > 0:
            probs = [score / total for score in positive]
            entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        else:
            entropy = 0.0
        sorted_scores = sorted(scores, reverse=True)
        diversity = self._mean_pairwise_diversity(items)
        return {
            "bm25_max": sorted_scores[0] if sorted_scores else 0.0,
            "bm25_gap": (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0,
            "bm25_entropy": entropy,
            "bm25_positive_frac": len(positive) / max(1, len(scores)),
            "retrieved_token_count": sum(sum(item.tokens.values()) for item in items),
            "retrieved_char_count": sum(item.n_chars + 2 for item in items),
            "candidate_diversity": diversity,
        }

    def _mean_pairwise_diversity(self, items: list[MemoryItem]) -> float:
        if len(items) < 2:
            return 0.0
        distances = []
        for i, left in enumerate(items):
            for right in items[i + 1 :]:
                distances.append(1.0 - self._token_jaccard(left.tokens, right.tokens))
        return sum(distances) / max(1, len(distances))

    @staticmethod
    def _token_jaccard(left: Counter[str], right: Counter[str]) -> float:
        left_keys = set(left)
        right_keys = set(right)
        if not left_keys and not right_keys:
            return 0.0
        return len(left_keys & right_keys) / max(1, len(left_keys | right_keys))

    def _merge_filtered_and_more(
        self,
        filtered_context: str,
        more_items: list[MemoryItem],
        budget_chars: int,
    ) -> tuple[str, dict]:
        pieces = [filtered_context] if filtered_context.strip() else []
        used = sum(len(piece) + 2 for piece in pieces)
        n_more_admitted = 0
        more_chars_admitted = 0
        admitted_more_indices: list[int] = []
        for item in more_items:
            item_len = item.n_chars + 2
            if pieces and used + item_len > budget_chars:
                continue
            pieces.append(item.text)
            used += item_len
            n_more_admitted += 1
            more_chars_admitted += item_len
            admitted_more_indices.append(item.idx)
        context = "\n\n".join(pieces) or "No previous memory"
        filtered_chars = len(filtered_context) if filtered_context.strip() else 0
        return context, {
            "n_more_admitted": n_more_admitted,
            "more_chars_admitted": more_chars_admitted,
            "admitted_more_indices": admitted_more_indices,
            "filtered_chars_admitted": filtered_chars,
            "final_context_source_mix": {
                "filtered_chars": filtered_chars,
                "more_chars": more_chars_admitted,
                "more_fraction": more_chars_admitted / max(1, filtered_chars + more_chars_admitted),
            },
        }

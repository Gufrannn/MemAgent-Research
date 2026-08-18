"""Reward-contrastive decision-boundary memory for official MemoryArena.

This module intentionally follows MemoryArena's add_chunk/wrap_user_prompt API.
It consumes only official experience chunks produced by the benchmark agents.
No benchmark questions, answers, or custom data are used at memory-write time.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from collections import Counter
from typing import Any


CARD_SYSTEM = """Convert one completed agent episode into a grounded decision card.
Return exactly one JSON object with keys:
task_family, state_signature, attempted_action, action_parameters, preconditions,
outcome, reward, failure_boundary, recovery_rule, success_evidence, retrieval_terms.

The purpose is to learn when an action succeeds or fails, not to summarize the task.
- Preserve exact entity names, values, IDs, action strings, visible constraints, and reward.
- failure_boundary states the condition separating failure from success, only if evidenced.
- recovery_rule is empty unless the trajectory contains evidence for recovery.
- retrieval_terms is a short list of literal task/action/environment terms.
- Never invent alternative actions or causal claims.
- Never replace concrete values with placeholders.
Return JSON only."""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_.:/-]+", text.lower())


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
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
    return {}


class ControlBoundaryMemorySystem:
    """Retrieve grounded episodes plus paired successful/failed decision cards."""

    def __init__(self, mode: str = "control", top_k: int = 4, max_context_chars: int = 14000):
        if mode not in {"control", "raw"}:
            raise ValueError("mode must be control or raw")
        self.mode = mode
        self.top_k = int(os.getenv("CONTROL_MEMORY_TOP_K", top_k))
        self.max_context_chars = int(os.getenv("CONTROL_MEMORY_MAX_CONTEXT_CHARS", max_context_chars))
        self.base_url = os.getenv("CONTROL_MEMORY_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
        self.model = os.getenv("CONTROL_MEMORY_MODEL", "qwen25-7b")
        self.max_card_tokens = int(os.getenv("CONTROL_MEMORY_CARD_TOKENS", "384"))
        self.raw_episodes: list[str] = []
        self.cards: list[dict[str, Any]] = []
        self.doc_freq: Counter[str] = Counter()

    def _call_writer(self, episode: str) -> dict[str, Any]:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": CARD_SYSTEM},
                {"role": "user", "content": "OFFICIAL COMPLETED EPISODE:\n" + episode},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_card_tokens,
            "seed": 20260818,
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            payload = json.loads(response.read())
        return _json_object(payload["choices"][0]["message"]["content"])

    def add_chunk(self, chunk: str):
        raw = str(chunk)
        index = len(self.raw_episodes)
        self.raw_episodes.append(raw)
        for token in set(_tokens(raw)):
            self.doc_freq[token] += 1
        if self.mode == "control":
            try:
                card = self._call_writer(raw)
            except Exception as exc:
                card = {"parse_error": str(exc), "outcome": "unknown"}
            card["episode_index"] = index
            # The official numeric reward remains authoritative even if the
            # card writer formats it incorrectly.
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict) and "reward" in parsed:
                card["official_reward"] = parsed.get("reward")
                card["outcome"] = "success" if bool(parsed.get("reward")) else "failure"
            self.cards.append(card)
        return {"episode_index": index, "mode": self.mode}

    def _score(self, query: str, text: str) -> float:
        query_terms = _tokens(query)
        document = Counter(_tokens(text))
        n_docs = max(1, len(self.raw_episodes))
        score = 0.0
        for term in query_terms:
            if term not in document:
                continue
            inverse = math.log((n_docs + 1) / (self.doc_freq.get(term, 0) + 1)) + 1.0
            score += inverse * (1.0 + math.log(document[term]))
        return score

    def _rank_raw(self, query: str, limit: int) -> list[tuple[int, float]]:
        ranked = [(i, self._score(query, text)) for i, text in enumerate(self.raw_episodes)]
        ranked.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return ranked[:limit]

    def _rank_cards(self, query: str) -> list[tuple[int, float]]:
        ranked = []
        for index, card in enumerate(self.cards):
            text = json.dumps(card, ensure_ascii=False, sort_keys=True)
            ranked.append((index, self._score(query, text)))
        ranked.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return ranked

    def _paired_card_indices(self, query: str) -> list[int]:
        ranked = self._rank_cards(query)
        successes, failures, other = [], [], []
        for index, _score in ranked:
            outcome = str(self.cards[index].get("outcome", "unknown")).lower()
            if outcome == "success" and len(successes) < self.top_k // 2:
                successes.append(index)
            elif outcome == "failure" and len(failures) < self.top_k // 2:
                failures.append(index)
            elif len(other) < self.top_k:
                other.append(index)
        selected = successes + failures
        for index in other:
            if len(selected) >= self.top_k:
                break
            selected.append(index)
        return selected

    def wrap_user_prompt(self, prompt: str) -> str:
        if not self.raw_episodes:
            context = "None"
        elif self.mode == "raw":
            blocks = [self.raw_episodes[i] for i, _ in self._rank_raw(prompt, self.top_k)]
            context = "\n\n".join(f"<grounded_episode>{x}</grounded_episode>" for x in blocks)
        else:
            blocks = []
            for card_index in self._paired_card_indices(prompt):
                card = self.cards[card_index]
                episode_index = int(card["episode_index"])
                label = str(card.get("outcome", "unknown")).upper()
                blocks.append(
                    f"<decision_card outcome=\"{label}\">\n"
                    + json.dumps(card, ensure_ascii=False, sort_keys=True)
                    + "\n<grounded_evidence>\n"
                    + self.raw_episodes[episode_index]
                    + "\n</grounded_evidence>\n</decision_card>"
                )
            context = "\n\n".join(blocks)
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars] + "\n[MEMORY CONTEXT TRUNCATED]"
        instruction = (
            "Use memory as evidence. Contrast successful and failed prior actions; apply a rule only "
            "when its grounded preconditions match the current state. Do not copy an action merely "
            "because it is similar. Preserve exact values from grounded evidence."
            if self.mode == "control" else
            "Use the retrieved prior episodes as grounded evidence."
        )
        return f"<memory_instruction>{instruction}</memory_instruction>\n<memory_context>\n{context}\n</memory_context>\nUser: {prompt}"

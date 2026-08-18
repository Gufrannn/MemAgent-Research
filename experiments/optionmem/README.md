# OptionMem / Mem2ActBench headroom experiment

This branch tests one narrow prerequisite before implementing RL:

> When the future query is hidden at memory-write time and memory budgets are
> matched, does procedural option memory improve future tool actions over a
> factual summary?

The experiment uses the official `toolmembench_small` split from Mem2ActBench.
It does not synthesize or relabel data. Historical sessions are selected using
the benchmark's `source_conversation_ids`, then compiled **before** the query is
shown. The query and target tool schema are only supplied to the executor.

## Conditions

| Condition | Executor input |
|---|---|
| `no_memory` | Query + tool schema |
| `full_history` | Official historical sessions + query + schema |
| `summary` | Query-independent fixed-budget factual memory + query + schema |
| `option` | Query-independent fixed-budget procedural memory + query + schema |
| `state` | Query-independent typed state store (exact entities, IDs, current values, tool evidence) + query + schema |
| `state_option` | Equal-total-budget factorization: 320-token typed state + 192-token procedure core + query + schema |

The v2 diagnostic keeps the total writer budget at 512 tokens per session while
separating parameter-grounding state from reusable control knowledge.  It is a
representation headroom test, not Future-Utility RL.

## v3: event-sourced executable memory

The v3 diagnostic replaces whole-memory rewriting with query-unknown atomic
event extraction. Events keep entity/tool/argument/result/provenance bindings
together and are appended locally across chunks. It compares:

| Condition | Selection rule |
|---|---|
| `ledger_all` | All extracted events (context-clipped only) |
| `ledger_retrieval` | Deployable lexical query/schema retrieval |
| `ledger_recency` | Query-independent recent-event heuristic |
| `ledger_oracle` | Analysis-only gold-tool/argument retrieval upper bound |

The oracle is never a deployable result. Its role is causal diagnosis: a weak
oracle rejects the event representation; a strong oracle with weak retrieval
identifies retrieval/retention headroom. Only the latter justifies item-level
Future-Utility RL.

Ledger extraction uses independently parsed 3k-character chunks and vLLM JSON
mode. Events are appended by deterministic host code rather than asking the
model to rewrite prior memory. Raw chunk responses remain in the cache for
audit and malformed-output diagnosis.

The summary reports both all-chunks-valid session rate and chunk-level JSON
validity, plus gold-argument surface recall for each executor context. The
analysis-only oracle ranks events by exact gold-value coverage; this makes it a
representation/retrieval ceiling rather than a deployable policy.

Event admission is deliberately selective: tool evidence, concrete state,
preferences/constraints, corrections, and actionable failures are retained;
generic dialogue and non-actionable assistant prose are excluded. This avoids
turning long-term memory into a verbose transcript under another name.

Run `run_mem2act_ledger_headroom.sh`; all writer and executor calls still use
the configured OpenAI-compatible vLLM endpoint.

All generations use an OpenAI-compatible vLLM endpoint. The primary metric is
exact tool-call accuracy; tool-name and per-argument accuracy are diagnostics.
The summary reports paired bootstrap confidence intervals.

## Decision rule

Proceed to Future-Utility RL only if:

1. `option - summary` is positive on exact tool-call accuracy;
2. the effect is not driven solely by L1 direct-copy examples;
3. option memory is competitive with full history under a much smaller context;
4. inspection confirms that gains come from reusable procedures/conflict rules,
   not accidental answer leakage.

If those conditions fail, do not train this representation. Inspect failures,
revise the memory unit, or stop the branch.

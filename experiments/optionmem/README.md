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

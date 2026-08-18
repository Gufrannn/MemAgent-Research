# RiskMem: versioned, query-time memory-risk headroom

This directory is isolated from MemAgent training. It tests whether versioning
plus a query-time risk gate has measurable headroom before spending GPU time on
RL. It uses the official HaluMem release; it does not construct a dataset.

## What is compared

All three conditions share the same deterministic lexical retriever and the
same vLLM answer model.

1. `naive`: top-k over every memory, including superseded versions.
2. `versioned_current`: top-k over current versions only.
3. `riskmem`: retrieve a larger candidate set, then let a query-time vLLM gate
   select useful/safe versions before answering. Historical questions may use
   old versions; instruction-like or contradictory records may be excluded.

The gate never receives gold answers, evidence, or HaluMem's `memory_source`
field. Those labels appear only in output audit fields. Version edges currently
use HaluMem's official `is_update` / `original_memories` annotations. Because
`original_memories` are often paraphrases rather than exact stored strings, the
harness links each annotation to the closest prior memory above token-F1 0.4
(~95% annotation coverage on HaluMem-Medium). This is explicitly an
**oracle-structure headroom**, not a deployable final system.

## Reproducibility status of candidate benchmarks (2026-08-18 audit)

- HaluMem: official GitHub evaluation code and public 140 MB Hugging Face data;
  immediately runnable. Dataset license is CC BY-NC-ND 4.0.
- MemEvoBench: paper is public, but no official code/data link was exposed on
  the arXiv record during this audit.
- MPBench (memory poisoning): paper is public, but no official code/data link
  was exposed on the arXiv record. Do not confuse it with the older multimodal
  process-reward benchmark that uses the same name.
- MemPoison: paper is public; no official code/data link was exposed on the
  arXiv record during this audit.

Adapters for the last three should be added only when their authors publish the
actual benchmark artifacts. Synthetic substitutes would invalidate comparison.

## Remote commands

Download the official medium split:

```bash
export WORK_ROOT=/data/cw/memagent_work
source "$WORK_ROOT/.venv/bin/activate"
cd "$WORK_ROOT/code/MemAgent"
bash experiments/riskmem/download_halumem.sh
```

The script assumes an OpenAI-compatible vLLM server is already serving
`qwen25-7b` at `http://127.0.0.1:8001`. Check it, then run 50 paired examples:

```bash
curl http://127.0.0.1:8001/v1/models
export MAX_QUESTIONS=50 CONCURRENCY=16
bash experiments/riskmem/run_halumem_headroom.sh
```

Outputs:

- `$WORK_ROOT/logs/riskmem/riskmem_halumem_n50.jsonl`: per-example predictions,
  selected IDs, and audit labels.
- the adjacent `.summary.json`: F1/exact match and paired bootstrap confidence
  intervals for RiskMem minus each baseline.

For a smoke test use `MAX_QUESTIONS=8`. Only expand beyond 50 if the paired
confidence interval is promising or inconclusive; a negative result should
move the branch to another mechanism rather than repeatedly rerun the same
idea.

# Qwen2.5-7B fixed-S128 I/T25 evaluation contract

This closure evaluates exactly two interfaces on the project’s already-fixed
`hotpotqa_dev.parquet` S128: base recurrent memory inference (`I`) and the
corrected Original-style 2-GPU pilot checkpoint after update 25 (`T25`).  It
does not select a new dataset, rerun raw-context `R`, or claim reproduction of
the original paper’s 7B execution protocol.

## Cohort and pairing

- S128 is the original project’s curated memory challenge produced by its
  question-only filter followed by `head(128)`.  It is not a random or
  population-representative HotpotQA development sample.
- P0 replays the production prompt filter and requires all 128 rows, in frozen
  source order 0–127, with the same interface-neutral manifest hash already
  certified by the stable-I canary at commit
  `bd8b804c2cbf333f0f0650b729fd03a143d445b2`.
  The failed first evidence directory is preserved; this contract consumes
  only the append-only `stable_i4x2_frozen_20260821r2` PASS report, resolved
  manifest, and hash-chained ledger.  The T25 training report must authenticate
  that exact current canary report SHA-256.
- Each interface has exactly one deterministic greedy trajectory per row.
  Stable example keys and trajectory seeds are identical across I and T25.
- The two interfaces use the same tokenizer, recurrent templates, chunking,
  prompt/response limits, strict synchronous vLLM backend, decoding settings,
  and reward-function input protocol.  P0 requires the generation-affecting
  resolved-Hydra projection hashes to be identical.  Only interface and model
  provenance may differ.

## T25 loading semantics

T25 loads only `actor/model_world_size_2_rank_{0,1}.pt` from the exact
`global_step_25` inventory certified by the corrected Original-style training
report.  A dedicated `actor_only_eval` path does not load optimizer,
LR-scheduler, RNG, critic, or dataloader state.  Each rank reports its loaded
path, size, and SHA-256 before the first actor/vLLM snapshot or rollout.  The
optimizer must be newly initialized and empty before and after loading and
throughout validation.  Actor master, effective rollout, and both vLLM worker
digests are attested before and after evaluation.  The pre-rollout T25
effective actor/vLLM digest must also equal the training report's certified
version-25 digest; merely differing from base I is not sufficient.

## Metrics and claims

The final audit independently recomputes normalized exact match and token F1
from the 128 terminal strings and the frozen parquet ground truths, taking the
best score over aliases.  It freezes the corrected-project answer extraction
route: last balanced `\boxed{...}`, then explicit “answer is/answer:”, then the
last non-empty line.  Missing or empty predictions score zero, every interface
must have exactly 128 rows, and the denominator never shrinks.  Boxed-format
success and historical substring exact match are diagnostics only.

The rollout’s composite dense training reward is deliberately ignored for
performance reporting.  `T25 − I` is a paired descriptive change on these 128
curated examples, not a causal effect and not a HotpotQA population estimate.
It includes all 25 corrected Original-style updates from the base model; the
new continuation itself contributes updates 4–25 (22 updates) after the
shared Gate A step-3 anchor.  A later `method25 − Original25` comparison, not
this contrast, isolates replacement of those final 22 updates.

Published `R` and other paper values remain historical references.  They are
metric-context references only and cannot be directly paired with this run
because the execution protocols differ.

## Required order

1. Finish and certify corrected Original-style T25 training.
2. Run standalone S128 I/T P0.
3. Run I once, then T25 once, under the shared GPU6–7 lock.
4. Generate the append-only audit report and hash-chained ledger record.

The manifests do not self-authorize GPU execution or causal claims.

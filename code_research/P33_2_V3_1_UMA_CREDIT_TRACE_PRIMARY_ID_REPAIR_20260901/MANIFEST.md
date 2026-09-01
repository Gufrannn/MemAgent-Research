# P33.2-v3.1 UMA Credit Trace Primary ID Repair Package

## Package status

`CODE REPAIRED / LOCAL SMOKE PASSED / INSTRUMENTATION DIFF UNCHANGED / S05M NOT RERUN / H20 NOT RUN`

This package contains the tiny v3.1 repair requested after code audit. It supersedes P33.2-v3 for H20 execution.

## Contents

- `code/prepare_p33_2_uma_credit_instrumentation.py`
- `code/summarize_p33_2_credit_trace.py`
- `diff/p33_2_uma_credit_instrumentation.diff`
- `docs/P33_2_V3_1_UMA_CREDIT_TRACE_PRIMARY_ID_REPAIR_20260901.md`
- `docs/code_research_README_snapshot.md`
- `h20_handoff/P33_2_V3_1_H20_REAL_ROLLOUT_HANDOFF_20260901.md`
- `status/p33_2_uma_credit_instrumentation_manifest.json`
- `status/p33_2_v3_1_status.json`
- `SHA256SUMS.txt`

## Official baseline

- UMA official local repo: `/Users/xinman/Desktop/2026/Research/Agent/Paper Investigation/AgentPaperSurvey/Agent_Memory_Intensive_Reading/code_research/repos/unified-memory-agent`
- official commit: `768f9620231bae11264771f59e43a4839506cf94`

## What changed from v3

Only the summarizer changed:

- `trajectory_id` is now the required primary trajectory grouping key;
- derived `trajectory_key` remains required as provenance/cross-check only;
- one `trajectory_id` must map to exactly one `trajectory_key`;
- multiple `trajectory_id` values may share one `trajectory_key` and are reported;
- reward rank is now tie-aware dense rank;
- added `reward_dense_rank_within_grpo_group`, `reward_tie_size_within_grpo_group`, `n_unique_rewards_in_grpo_group`.

The UMA instrumentation diff is unchanged from v3.

## Explicit non-changes

No changes to prompt, operator, split, metric, generation protocol, reward formula, trainer objective, or optimizer behavior.

## Local validation

Passed:

- AST parse;
- valid synthetic trace;
- UUID primary grouping;
- `trajectory_key -> multiple trajectory_id` allowed and reported;
- `trajectory_id -> multiple trajectory_key` fail-closed;
- missing `trajectory_id` fail-closed;
- tie-aware dense rank.

## H20 note

Do not use old P33.2-v3 commit for H20. Use the v3.1 exact commit after push.

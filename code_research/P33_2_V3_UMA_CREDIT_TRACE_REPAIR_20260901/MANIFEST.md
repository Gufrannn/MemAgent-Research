# P33.2-v3 UMA Credit Trace Repair Package

## Package status

`CODE REPAIRED / LOCAL SMOKE PASSED / S05M V3 NOT RUN / H20 REAL ROLLOUT NOT RUN`

This package contains the reviewable P33.2-v3 instrumentation repair. It does not contain the full UMA repository and does not contain H20 real rollout results.

## Contents

- `code/prepare_p33_2_uma_credit_instrumentation.py`
- `code/summarize_p33_2_credit_trace.py`
- `diff/p33_2_uma_credit_instrumentation.diff`
- `docs/P33_2_V3_UMA_CREDIT_TRACE_REPAIR_20260901.md`
- `docs/code_research_README_snapshot.md`
- `h20_handoff/P33_2_V3_H20_REAL_ROLLOUT_HANDOFF_20260901.md`
- `status/p33_2_uma_credit_instrumentation_manifest.json`
- `status/p33_2_v3_status.json`
- `SHA256SUMS.txt`

## Official baseline

- UMA official local repo: `/Users/xinman/Desktop/2026/Research/Agent/Paper Investigation/AgentPaperSurvey/Agent_Memory_Intensive_Reading/code_research/repos/unified-memory-agent`
- official commit: `768f9620231bae11264771f59e43a4839506cf94`
- v3 instrumented worktree: `/Users/xinman/Desktop/2026/Research/Agent/Paper Investigation/AgentPaperSurvey/Agent_Memory_Intensive_Reading/code_research/experiments/UMA-Official-Anatomy-20260901/instrumented_worktree_p33_2_v3`

## What changed

Only default-off instrumentation and analysis summarization changed:

- derived `p33_2_trajectory_key`;
- true `conversation_index` / `memory_step_index` from `AgentLoopOutput` construction;
- no row-order chunk reconstruction;
- fail-closed identity contract;
- per-host/pid JSONL trace files;
- reward rank within GRPO group;
- trajectory-level mean memory advantage.

No prompt/operator/split/metric/generation/reward/objective changes were made.

## Local validation

Passed:

- Python compile;
- `git diff --check`;
- trace default-off no-write;
- trace enabled per-process write;
- valid synthetic trace summary;
- missing trajectory key fail-closed;
- reward-rank tie-break.

## Current blockers

s05m v3 smoke has not been run because the local approval reviewer rejected the script upload to s05m without more explicit user authorization.

H20 real rollout has not been run; it should wait for an exact pushed commit or another clearly auditable transfer path.

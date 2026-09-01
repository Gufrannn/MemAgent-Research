# P33.2-v3.1 UMA credit trace primary identity repair（2026-09-01）

## 状态标签

`CODE REPAIRED / LOCAL SMOKE PASSED / INSTRUMENTATION DIFF UNCHANGED / S05M NOT RERUN / H20 NOT RUN`

## 为什么需要 v3.1

P33.2-v3 已经修复了 row-order chunk reconstruction、conversation index、memory-step index 与 per-process logging，但代码审查进一步指出：

> 官方 `ToolMemoryAgentLoop` 已经为每次完整 agent run 生成 `trajectory_id = str(uuid4())`，它才是 per-rollout unique trajectory identity。v3 派生的 `(step, sample_index, rollout_n, validate)` 只应作为 provenance / cross-check，不能作为 primary trajectory grouping key。

原因是 `rollout_n` 在 worker-local batch 内计数；如果 repeated sample 跨 worker chunk 边界，derived key 理论上可能 collision。

## 本轮只修什么

只修改：

- `tools/summarize_p33_2_credit_trace.py`

修复内容：

1. `trajectory_id` 改为 REQUIRED primary trajectory identity。
2. `trajectory_key` 保留为 REQUIRED provenance key。
3. trajectory-level grouping 从 `trajectory_key` 改为 `trajectory_id`。
4. 增加 `trajectory_id -> trajectory_key` 唯一性 audit；同一 UUID 对多个 derived key 直接 fail closed。
5. 允许 `trajectory_key -> multiple trajectory_id`，并在 summary 中报告该数量；这正是 derived key 可能 collision 的检测。
6. `reward_rank_within_grpo_group` 改为 tie-aware dense rank。
7. 新增：
   - `reward_dense_rank_within_grpo_group`
   - `reward_tie_size_within_grpo_group`
   - `n_unique_rewards_in_grpo_group`

## 明确没有改什么

未改：

- UMA instrumentation diff；
- `prepare_p33_2_uma_credit_instrumentation.py`；
- prompt；
- memory operator；
- split / data protocol；
- metric / evaluator；
- generation protocol；
- reward formula；
- trainer objective；
- optimizer behavior。

## SHA256

- prepare script: `f38307e60ddc124aa11787e1372489d417b507b8399de210847f978542144e71`
- summarizer v3.1: `87a1d4ff2587e41abd07d4b75bc101cc0e89e3ce0167eec3b5845af5c67c776e`
- instrumentation diff unchanged: `611da8810a7153d00fd402e6d1d117938f31c95ff278612adeaee74b78d9e13e`

## 本地 smoke

通过：

- AST parse：PASS。
- valid synthetic trace：PASS。
- official UUID grouping：PASS；两个不同 `trajectory_id` 即使共享同一个 derived `trajectory_key`，仍被分成两条 trajectory。
- `trajectory_key -> multiple trajectory_id`：允许并报告。
- `trajectory_id -> multiple trajectory_key`：fail closed。
- missing `trajectory_id`：fail closed。
- tie-aware dense reward rank：PASS。

Synthetic 检查摘要：

- `n_trajectory_ids=2`
- `n_trajectory_keys=1`
- `trajectory_keys_with_multiple_trajectory_ids=1`
- `n_memory_rows=6`
- first trajectory dense ranks: raw 1.5 -> rank 1 / tie size 2；raw 1.0 -> rank 2；raw 1.5 -> rank 1 / tie size 2

## s05m / H20 状态

本轮不建议为了形式完整重跑 s05m；s05m 对这一步没有新增科学信息增益。真正下一步是 H20 official UMA 4B real rollout anatomy。

H20 前必须使用包含 v3.1 summarizer 的 exact GitHub commit，不要继续用 v3 commit。

## 解释边界

v3.1 只能支持：

- H20 前 primary trajectory identity contract 已修复；
- summarizer 不再用 derived key 做正式 trajectory grouping；
- tie reward 不再被错误排成不同 ordinal rank。

v3.1 不能支持：

- UMA credit assignment 错误；
- RL necessity；
- P33.3 intervention result；
- H20 real rollout anatomy。

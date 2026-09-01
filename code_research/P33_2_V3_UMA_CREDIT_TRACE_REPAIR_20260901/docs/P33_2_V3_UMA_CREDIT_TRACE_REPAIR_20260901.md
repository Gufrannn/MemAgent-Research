# P33.2-v3 UMA credit trace instrumentation repair（2026-09-01）

## 状态标签

`CODE REPAIRED / LOCAL SMOKE PASSED / S05M SYNC BLOCKED BY LOCAL APPROVAL REVIEW / NO TRAINING / NO GENERATION / H20 NOT YET RUN`

## 本轮只修什么

本轮只处理 P33.2 代码审查指出的 instrumentation / summarizer 问题：

1. 不再依赖可能为空的 `trajectory_id` 做 trajectory-level 分组。
2. 使用从 UMA official `get_trajectory_info(step, sample_index, rollout_n, validate)` 派生的 `p33_2_trajectory_key`。
3. 在 `tool_mem_agent_loop.py` 创建 `AgentLoopOutput.extra_fields` 时直接写入真实 `p33_2_conversation_index`、`p33_2_memory_step_index`、`p33_2_final_query_index`。
4. summarizer 禁止 row-order reconstruction；如果缺少 identity / index 字段，直接 fail closed。
5. trace logging 改为每个 host/pid 独立 JSONL 文件，默认关闭，不使用共享文件锁，不逐条 `fsync`。
6. summarizer 增加 `reward_rank_within_grpo_group` 和 `trajectory_mean_memory_advantage`。

## 明确没有改什么

未改：

- prompt
- memory operator
- split / data protocol
- metric / evaluator
- generation protocol
- reward formula
- trainer objective
- optimizer step 行为

## 本地 official baseline

- local official repo: `/Users/xinman/Desktop/2026/Research/Agent/Paper Investigation/AgentPaperSurvey/Agent_Memory_Intensive_Reading/code_research/repos/unified-memory-agent`
- official commit: `768f9620231bae11264771f59e43a4839506cf94`
- instrumented worktree v3: `/Users/xinman/Desktop/2026/Research/Agent/Paper Investigation/AgentPaperSurvey/Agent_Memory_Intensive_Reading/code_research/experiments/UMA-Official-Anatomy-20260901/instrumented_worktree_p33_2_v3`

## 修改/新增文件

Preparation scripts:

- `/Users/xinman/Desktop/2026/Research/Agent/Paper Investigation/AgentPaperSurvey/Agent_Memory_Intensive_Reading/code_research/experiments/UMA-Official-Anatomy-20260901/tools/prepare_p33_2_uma_credit_instrumentation.py`
- `/Users/xinman/Desktop/2026/Research/Agent/Paper Investigation/AgentPaperSurvey/Agent_Memory_Intensive_Reading/code_research/experiments/UMA-Official-Anatomy-20260901/tools/summarize_p33_2_credit_trace.py`

Instrumented UMA files:

- `external/verl/verl/utils/uma_credit_trace.py` added
- `external/verl/verl/experimental/agent_loop/agent_loop.py` modified
- `external/verl/verl/experimental/agent_loop/tool_mem_agent_loop.py` modified
- `external/verl/verl/trainer/ppo/ray_trainer.py` modified

## SHA256

- prepare script: `f38307e60ddc124aa11787e1372489d417b507b8399de210847f978542144e71`
- summarizer: `cc35a956d94346fb51cfe750da422883c095f07142827d559f532d3bf9e531de`
- instrumentation diff: `611da8810a7153d00fd402e6d1d117938f31c95ff278612adeaee74b78d9e13e`
- instrumentation manifest: `adc5c22e59c70b2b7b4277c1728fcfa4be12ca9edd272450b9f5653582ed78f9`

Manifest-reported file hashes inside instrumented worktree:

- `agent_loop.py`: `e907a41ad7f797ac6b7904b1354fa0f1407e2b7db3a609394eb5ec3b8c99d80e`
- `tool_mem_agent_loop.py`: `8785a9da811a968269a29835e40de0d8fdc1ccff9b360468baa3a8d8984f7cc7`
- `ray_trainer.py`: `1ce18323a870f8c2d90168ef85e295a4269a184e6051bc78692895ed76c0b647`
- `uma_credit_trace.py`: `041ddc0f8ad6db4d483ce180de6455a2c5cb7c59fadfcbc0e4dca5ad409ee4b3`

## 本地 smoke 结果

通过：

- Python compile：prepare / summarizer / instrumented `agent_loop.py` / `tool_mem_agent_loop.py` / `ray_trainer.py` / `uma_credit_trace.py`
- `git diff --check`
- trace module default-off：`UMA_CREDIT_TRACE` 未开启时不写 JSONL
- trace module enabled：开启后写入独立文件 `uma_credit_trace.<host>.<pid>.jsonl`
- summarizer valid synthetic trace：`P33_2_TRACE_SUMMARY_COMPLETE`
- summarizer identity contract：`PASS`
- summarizer fail-closed：缺 `trajectory_key` 时退出并报告 identity contract violation
- reward rank tie-break：raw score 降序、同分按 `row_index` 升序

关键 synthetic 数字：

- `n_events=6`
- `n_memory_rows=6`
- `within_trajectory_qa_variance_mean=0.0`
- `within_trajectory_advantage_variance_mean=0.5277777777777778`
- first trajectory rank: row 0 / raw 1.5 / rank 1；row 1 / raw 1.0 / rank 3；row 2 / raw 1.5 / rank 2

## s05m 状态

本轮尝试把两个 v3 脚本同步到 s05m 做远端 smoke 时，被本地安全审查拒绝，理由是“向远端发送可能包含私有研究代码需要更明确授权”。因此：

- 未绕过审查；
- 未在 s05m 创建 v3 worktree；
- 未在 s05m 跑 v3 smoke；
- 旧的 s05m v2 smoke 不作为 v3 证据。

如果需要继续 s05m，请用户明确授权“允许将 P33.2-v3 两个脚本 scp 到 s05m `/data/manxin/agent_memory_rl/p33_uma_official_anatomy/tools/` 并执行只读 smoke”。

## H20 下一步

P33.2-v3 的 H20 任务不是训练，也不是正式 RL。目标是：

> 在官方 UMA 4B real rollout 中读取真实 reward / advantage / trajectory / memory-step 结构，判断 UMA 当前 Task-Stratified GRPO 信号是否主要是 trajectory-level outcome discrimination，而非 transition-level contribution estimate。

必须等待 v3 代码 push 到 GitHub exact commit 后再给 H20 执行，避免 H20 手工改动。

## 解释边界

这轮不能支持：

- UMA credit assignment 错误；
- RL necessity；
- 我们方法有效；
- H20 real rollout 结论。

这轮只能支持：

- P33.2 instrumentation 的两个关键 identity/index 风险已在代码层修复；
- 本地 smoke 支持 v3 可以进入 H20 real rollout precheck；
- 真实 credit anatomy 仍需 H20 official rollout trace。

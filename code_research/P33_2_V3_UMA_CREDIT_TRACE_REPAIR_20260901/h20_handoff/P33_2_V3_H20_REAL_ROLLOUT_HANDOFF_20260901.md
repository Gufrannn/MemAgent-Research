# P33.2-v3 H20 handoff: UMA official 4B real rollout credit anatomy

## 目的

运行官方 UMA 4B real rollout 的 credit trace anatomy，读取真实 `reward / advantage / trajectory / memory-step` 结构。

这不是训练实验，不是方法验证，不是 RL necessity 证明。

## 必须遵守

- 只使用 pushed exact commit。
- 不手工编辑 H20 worktree。
- 不改 prompt。
- 不改 memory operator。
- 不改 split / data protocol。
- 不改 metric / evaluator。
- 不改 generation protocol。
- 不改 reward formula。
- 不改 trainer objective。
- 如果 trainer 必须走到 advantage 才能拿到 tensor，优先在 optimizer step 前退出；如果难以插 stop，只允许 one-step dry run，并明确只分析 pre-update tensors。

## 需要等待的信息

当前 P33.2-v3 已完成本地 repair 和 local smoke，但尚未 push 到 GitHub exact commit。

H20 执行前需要补齐：

- GitHub repo
- branch
- exact commit
- diff SHA256
- instrumented manifest SHA256

## 运行目标

第一轮 H20 real rollout 目标：

- 8–16 original samples；
- multiple rollouts；
- multiple memory conversations per trajectory；
- memory rows 最好 ≥ 100；
- 如果 Ledger-QA 环境可直接跑，优先 Ledger-QA；
- 如果官方 Generalist runbook 更容易，先用官方原生训练数据跑通 plumbing。

## 成功 gate

Gate 1: `trajectory_key` / `sample_index` / `rollout_n` / `conversation_index` / `memory_step_index` 100% 完整。

Gate 2: 每条 memory row 可以匹配 uid、trajectory、memory-step、reward、advantage。

Gate 3: 报告 `Var_t(QA | trajectory)` 和 `Var_t(advantage | trajectory)`。

Gate 4: 报告 reward rank within GRPO group 与 trajectory-level mean memory advantage。

## 输出回传

只需回传 compact 文本结果；不要推大模型、数据、完整训练产物。

必须给：

- exact commit
- run root
- run log
- trace file count
- trace event count
- memory row count
- identity contract status
- summary JSON SHA256
- table CSV SHA256
- 是否发生 optimizer step

## 边界

P33.2-v3 只能支持 credit anatomy observation，不能单独支持“credit 错了”。真正错误归因需要 P33.3 intervention reference。

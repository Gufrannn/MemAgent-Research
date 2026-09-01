# P33.2-v3.2 UMA credit trace source/env/runner repair

本轮回答 H20 v3.1 阻塞后的“谁做”：主线负责修 source/env/runner/protocol，H20 只负责执行固定 commit。

## 背景

H20 v3.1 在执行前正确 BLOCK：

- H20 上没有官方 UMA exact commit `768f9620231bae11264771f59e43a4839506cf94` 的 clean source。
- 已有 MemAgent 系 repo 都不是该官方 commit，且部分 dirty。
- 没有直接 rollout runner。
- 没有确认的官方 Qwen3-4B model path。
- 没有确认的官方 UMA native train/val data path。
- 没有确认 trainer env。

因此不能让 H20 自行替换 source、模型、数据或手工 patch。

## v3.2 修复

只做两类修复：

1. `prepare_p33_2_uma_credit_instrumentation.py`
   - 新增默认关闭的 `UMA_CREDIT_TRACE_EXIT_AFTER_ADVANTAGE`。
   - 只有同时设置 `UMA_CREDIT_TRACE=1` 和 `UMA_CREDIT_TRACE_EXIT_AFTER_ADVANTAGE=1` 时，trainer 在真实 `compute_advantage(...)` 之后、critic/actor optimizer step 之前写 trace 并 `return`。
   - 不改 prompt/operator/split/metric/reward formula/generation protocol/optimizer math。

2. 新增 H20 runner：
   - `run_p33_2_v3_2_h20_preupdate_advantage_trace.sh`
   - 从官方 HTTPS repo 拉取/checkout exact commit `768f962...`。
   - 使用 exact-anchor prepare script 生成 fresh instrumented worktree。
   - `RUN_MODE=precheck` 只做 source/model/data/env/patch 检查。
   - `RUN_MODE=one_step` 才启动一次真实 rollout，并强制 advantage 后退出。
   - 模型和数据路径必须由 H20 明确提供；没有就 fail-closed。

## 本地验证

- 本地官方 UMA repo commit：`768f9620231bae11264771f59e43a4839506cf94`
- 官方 source status：clean
- prepare script `py_compile`：PASS
- summarizer `py_compile`：PASS
- H20 runner `bash -n`：PASS
- 本地生成 instrumented worktree：PASS
- instrumented `agent_loop.py` / `tool_mem_agent_loop.py` / `ray_trainer.py` / `uma_credit_trace.py` `py_compile`：PASS
- 新 safe-exit anchor 位置：`ray_trainer.py` 中 `compute_advantage(...)` 之后，`# update critic` 之前。

## 边界

本轮没有运行 H20 rollout，没有训练，没有 optimizer step，没有下载模型/数据，没有改旧结果。

v3.2 仍然是 instrumentation / runner repair，不是正式实验结果。

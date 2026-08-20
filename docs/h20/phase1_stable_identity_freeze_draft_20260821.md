# MemAgent 7B stable-I canary 与主实验衔接说明

状态：`STABLE_I_CANARY_CODE_FREEZE`

`gpu_execution_authorized=false`

本文件描述 canary 的边界；实际执行仍必须先通过同一 commit 的独立 P0。

Gate A r5 已证明 corrected Original-7B 的 rollout→actor update→checkpoint→resume→vLLM sync 链。它不是性能结果；其 `global_step_3` 是后续 corrected Original T25 的明确共同 warm-start，不能被当作未经说明的随机 checkpoint。

## 1. 下一道最小门

项目现有 `hotpotqa_dev.parquet` 本身就是原 MemAgent 已确定的 S128；不重新抽样、筛选或构造评测集。先在这份既有 S128 的前四个冻结位置上一次性做 stable-identity 4 例 × 2 replicas × 2 attempts 的工程 canary。通过后不增加 Q/G/R 等前置门，直接冻结并执行 corrected Original 从 Gate A step3 续训到 T25，再在同一 S128 上比较 base recurrent I 与 T25 recurrent T。

Git 合同冻结 dev parquet 的精确 SHA-256 与“位置 0–3”这一结果无关的选择规则。独立 P0 在任何 rollout 前读取该不可变文件，复现 production filter，断言 128 行全部保留，并输出完整 128 行的 `source_order_index → extra_info.index → question/context/ground-truth hash` resolved manifest。四个 semantic ID 由该 resolved manifest 得出，不在代码中猜测或硬编码。

基础身份参数：

- model：Qwen/Qwen2.5-7B-Instruct
- model revision：`a09a35458c702b33eeacc393d103063234e8bc28`
- evaluation data：项目既有 `hotpotqa_dev.parquet`（固定 S128）
- evaluation data SHA-256：`54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6`
- seed：2026
- rollout replicas：2
- trajectory seed mode：`independent`
- rollout/evaluation backend：strict vLLM；禁止 HuggingFace fallback
- 不构造新数据集

## 2. 运行前仍必须冻结的项目

以下任一项为空，P0 必须 fail-closed：

- 独立 Git 分支、40 字符 commit 和干净 worktree；
- 物理 GPU 白名单、world size、FSDP size；
- 模型全部文件 SHA、tokenizer SHA；
- 既有 S128 全部 128 行及 canary 前四行的 position→semantic ID→question/context/ground-truth hash 映射；
- prompt/template/chunk/max-token/decode 参数；
- 唯一 run ID、日志目录、证书目录；
- stable-ID ledger schema 和 command manifest；
- 是否只做 rollout identity gate，或包含 actor update；二者不得在运行中临时切换；
- 明确 terminal/technical-failure 规则；
- CPU 单测、静态检查和独立代码审查结果。

不得沿用 r5 输出目录，不得修改或覆盖 r5 ledger/certificates/checkpoints。

## 3. Stable-ID 小门验收条件

- 恰好 4 个 group，每个 group 恰好 2 个 replica；总计 8 条 base trajectory identity；
- dataset position、semantic dataset index、source order、replica、repeated row、active sample index、trajectory_id、trajectory_seed 全部显式记录；
- `source_repeated_row` 为 0–7，`source_order_index` 为 0–3，`replica_id` 为 0–1；
- 同一 source order 的两个 replica 共享样本身份，但 trajectory ID 和 independent seed 不同；
- 不同 trajectory 的 seed 无碰撞，且可由 base seed、eval manifest hash、example ID、source order 和 replica ID 重建；不依赖训练 global step；
- 每个 recurrent turn 的 request seed 可重建，并与 active sample_index 一一对齐；
- select/repeat/reorder/union 后 identity 字段、response、mask、reward 保持同一行语义；
- vLLM 请求逐条接收对应 seed；不允许 batch-level 单 seed 覆盖；
- 没有缺行、重复伪造、静默丢样本或位置/语义 ID 混用；
- 原始 ledger、resolved manifest、审计报告 append-only 保存；
- 审计失败只产生 NO-GO，不自动重跑或改参数。

## 4. Stable-ID 通过后的正式性能对象

在同一份既有 S128、同一 interface-neutral source manifest、模板、chunk、token budget 和 deterministic decode 上定义：

- I：7B recurrent inference-only；
- T25：从 Gate A r5 `global_step_3` 显式 resume，并完成同配置 step4–25 后的 recurrent actor。

I/T25 都必须使用 strict vLLM，且各跑既有 S128 的 128/128、`n=1`、deterministic decode。两者使用相同 stable join key；interface、checkpoint 和代码 provenance 单独记录，不进入共同 source-manifest hash。原论文 R/I/T 数字作为 published historical reference；本轮不重跑 R，也不把旧协议 R 与新 T25 相减成 paired effect。

## 5. 后续训练边界

当前主实验只执行 corrected Original `step3→T25`（22 次新 actor update）和同一 S128 的 I/T25 对比。技术断点可按固定周期保存，但科学 anchor 只看预注册的 T25；不得按中途指标挑 checkpoint。

当前不启动（由本 stable-I canary 分支）：T25/T50/T100/T200、SMSB、Tetrad、legacy Path、pair reward、默认 NCR 和改进方法。canary 通过后由紧接着的独立冻结分支启动 Original T25；改进方法再复用同一个 step3 warm-start、22 次 update 预算和同一 S128 评测合同，避免把基础设施修复冒充论文方法。

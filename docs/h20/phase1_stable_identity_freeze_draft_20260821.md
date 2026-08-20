# MemAgent 7B 下一阶段冻结草案（仅准备，未授权执行）

状态：`DRAFT`

`gpu_execution_authorized=false`

本草案不能作为启动训练、rollout 或 evaluation 的命令。Gate A r5 的证据目录和 checkpoint3 不作为后续科学实验的默认起点。

## 1. 下一道最小门

先做 stable-identity 4 例 × 2 replicas 的独立小门；通过后才冻结 fixed-128 统一锚点。不得直接跳到 T25。

固定候选样本采用现有 HotpotQA parquet 的 production-effective 前四个位置，不按模型结果挑选：

| dataset position | semantic `extra_info.index` |
|---:|---:|
| 0 | 2 |
| 1 | 6 |
| 2 | 7 |
| 3 | 9 |

基础身份参数：

- model：Qwen/Qwen2.5-7B-Instruct
- model revision：`a09a35458c702b33eeacc393d103063234e8bc28`
- train data：项目现有 `hotpotqa_train_32k.parquet`
- train data SHA-256：`798b7a2a9ece4f40884e2a9d02d165d7352df7763d1569ceaf402b45f76896f8`
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
- 4 个样本的 position→semantic ID→question/hash 映射；
- prompt/template/chunk/max-token/decode 参数；
- 唯一 run ID、日志目录、证书目录；
- stable-ID ledger schema 和 command manifest；
- 是否只做 rollout identity gate，或包含 actor update；二者不得在运行中临时切换；
- 明确 terminal/technical-failure 规则；
- CPU 单测、静态检查和独立代码审查结果。

不得沿用 r5 输出目录，不得修改或覆盖 r5 ledger/certificates/checkpoints。

## 3. Stable-ID 小门验收条件

- 恰好 4 个 group，每个 group 恰好 2 个 replica；总计 8 条 base trajectory identity；
- dataset position、semantic dataset index、group、replica、row、sample_index、trajectory_id、trajectory_seed 全部显式记录；
- `row` 为 0–7，`group` 为 0–3，`replica` 为 0–1；
- 同一 group 的两个 replica 共享样本身份，但 trajectory ID 和 independent seed 不同；
- 不同 trajectory 的 seed 无碰撞且可由 frozen seed/global step/group/replica 重建；
- 每个 recurrent turn 的 request seed 可重建，并与 active sample_index 一一对齐；
- select/repeat/reorder/union 后 identity 字段、response、mask、reward 保持同一行语义；
- vLLM 请求逐条接收对应 seed；不允许 batch-level 单 seed 覆盖；
- 没有缺行、重复伪造、静默丢样本或位置/语义 ID 混用；
- 原始 ledger、resolved manifest、审计报告 append-only 保存；
- 审计失败只产生 NO-GO，不自动重跑或改参数。

## 4. Stable-ID 通过后才允许冻结的统一 anchors

在同一、事先固定的 S128 IDs、模板、chunk、token budget 和 deterministic decode 上定义：

- R：raw 7B direct full-context；
- I：7B recurrent inference-only；
- 可选 Q：question-only；
- 可选 G：gold-support competence。

R/I 都必须使用 vLLM。S128 ID 列表、顺序和 SHA 必须在首次看指标前提交。anchor 只用于建立可比基线，不自动授权 corrected Original 训练。

## 5. 后续训练边界

corrected Original 的候选顺序是 2→25→50→100→200，但每一级都需要独立冻结与授权。100→200 的继续条件必须在揭盲 T100 指标前冻结；200 后不自动进入 400。

当前不启动：step4、T25/T50/T100/T200、SMSB、Tetrad、legacy Path、pair reward、默认 NCR 或任何临时派生实验。

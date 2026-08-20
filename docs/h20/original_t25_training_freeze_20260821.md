# Corrected Original-style 2-GPU pilot：step3 → T25 冻结合同

## 科学口径

这是 **corrected Original-style 2-GPU pilot**，不是原论文 7B 复现。它沿用已通过
Gate A r5 的 H20 两卡配置：Qwen2.5-7B-Instruct、GPU 6/7、batch 4、rollout
`n=2`、naive reward、warmup 2、strict vLLM。原论文的 32 GPU、batch 128、
`n=16`、thread reward、warmup 20 属于另一计算规模，不能把本试验写成等价复现。

Gate A 的 `global_step_3` 已经包含从基础模型开始的 3 次 corrected Original 更新。
本执行显式载入完整 step3 model、optimizer、scheduler、RNG 与 dataloader cursor，随后
只执行 step4–25，共新增 22 次更新。因此：

- `T25 − I` 只表示在同一冻结 S128 上观测到的描述性变化：完成 25 次 corrected
  Original 更新后的 T25 与 I 之差，不是总体或因果效应；
- 本次执行本身是从 step3 到 step25 的 22 次新增更新；
- 未来改进方法若也从同一 step3 warm-start 开始并得到 22 次更新，可以报告同预算的
  `method25 − Original25` paired comparison；它仍是固定评测集上的描述性比较，不是
  因果或总体结论，也不能把 T25 误写成“只训练 22 步”。

主要科学 endpoint 是 `global_step_25`。为保留不可逆的学习曲线证据，step5/10/15/20
被预注册为次要 learning-curve anchors；它们不会在当前 I/T25 评测中自动运行。
`max_actor_ckpt_to_keep=5` 要求 step5/10/15/20/25 的 model、optimizer、extra-state 与
`data.pt` 都完整保留，final audit 不接受 data-only remnants。最终报告和 hash-chained
ledger 会逐 anchor 固定 path、global step、逐文件大小/SHA-256 及 canonical inventory
hash；之后任一 secondary shard 被替换都会使只读复核失败。

## 前置门与连续性

P0 在任何训练进程启动前失败即停：

1. stable-I canary r2（r1 FAIL 证据原样保留）的 final report 必须是
   `PASS / I_RECURRENT_IDENTITY_CANARY_PASS`，
   Git commit 必须是 `bd8b804c2cbf333f0f0650b729fd03a143d445b2`，并由其
   append-only ledger 尾记录认证；
2. Gate A r5 final report 的 SHA-256 必须是
   `5f8b67b496bd672cb6e89c9ec481c1de97adbf0a73c3459edd02aef79830dca4`，
   且 P0/P1/P2、A1–A5 全部 PASS；
3. 当前 step3 的七个 checkpoint 文件必须逐文件匹配 Gate A final report；训练结束后
   再次逐文件复核，确保 warm-start 没被修改；
   P0 还会只读解析两个小型 extra-state shard，要求 scheduler `last_epoch=3` 且 CPU、
   CUDA、NumPy、Python 四类 RNG 状态齐全；resume worker 必须逐 rank 回执这些状态已恢复；
4. 固定 HotpotQA train parquet 后重放 production prompt filter，证明 step3 已消费
   positions 0..11，step4–25 恰消费 positions 12..99（88 prompts、176 trajectories）；
5. 同时冻结每个位置的 context token 数与 `ceil(tokens/5000)` active-turn 数，final
   audit 对每条 trajectory 的 active/final turn、UID、replica、dataset index 与 request
   seed 做逐条复核；
6. GPU 必须是物理 6、7，且两张均为 NVIDIA H20；P0 与训练前身份必须完全相同并空闲；
7. 可用空间必须至少为 `5 × step3完整checkpoint字节数 + 20 GiB`；
8. stable-I、Gate A 和本训练共用同一个 GPU6-7 文件锁，避免共享账号下的启动竞态。

## 单一训练参数源

`experiments/7b_gate_a/run_gate_a.sh` 中的 `TRAINER_OVERRIDES` bash 数组同时用于：

- 生产训练命令；
- P0 的无 GPU Hydra compose；
- argv SHA-256 与完整 resolved-config SHA-256 冻结。

P0 还用同一数组生成 Gate A step2→3 reference config；归一化后只允许下列字段不同：
`total_training_steps`、experiment/output、resume source、save frequency、checkpoint retention。
训练进程在 actor 初始化前重新计算完整 Hydra config SHA，并把 config/argv 两个 SHA 写入
hash-chained ledger；不匹配时在 rollout 前终止。

## 最终通过条件

`original_t25_final_report.json` 只有在以下条件全部满足时才写入：

- resume 首步是 step4，rollout 覆盖且仅覆盖 step4–25；
- optimizer max 与 scheduler `last_epoch` 在两个 rank 上逐步到 25；
- actor→两个 vLLM worker 在 version3..25 每一步都有完整 ack，且 sampled tensor digest
  一致；
- 每步执行信号有限，包含 reward、advantage、policy loss、grad norm；
- step5/10/15/20/25 的 model/optim/extra/data 七个文件均齐全，逐文件大小和 SHA-256
  进入 `checkpoint_anchors`；
- 每个 anchor 的 `inventory_sha256` 以及整个 anchor map 的 SHA-256 都使用 canonical
  JSON：sorted keys、无空格、禁止 NaN；
- source step3 inventory 在执行前后完全相同。

训练通过会保留 T5/T10/T15/T20/T25 五个 actor anchor，但不自动产生性能结论。T25 是
primary endpoint；前四个只是预注册 secondary learning-curve anchors，必须以后用同一
冻结 S128 I/T 协议逐 anchor 做 actor-only 评测才能形成性能曲线。当前独立 S128 闭包只对
同一既有 128 样本运行 I 与 T25；不会重新采样，也不会重复跑论文已有的 R baseline。

# MemAgent Qwen2.5-7B Gate A r5 PASS 交接

日期：2026-08-21（Asia/Shanghai）

## 1. 冻结身份

- Git 分支：`h20/qwen25-7b-gatea-2gpu-frozen-20260820`
- Git 提交：`c3f987be5513cad2a9e95622dd6773726a7bf12e`
- 模型：Qwen2.5-7B-Instruct
- 执行：H20 物理 GPU 6、7；world size 2
- reward manager：`naive`
- rollout：strict vLLM
- seed：2026
- 顺序：fresh step1→step2→checkpoint2→显式 resume→step3→checkpoint3

## 2. H20 已返回的终态

- wrapper exit code：`0`
- final status：`PASS`
- final decision：`GATE_A_PASS`
- gates：P0/P1/P2 全部 `PASS`
- audits：A1/A2/A3/A4/A5 全部 `PASS`
- A3：`applicable=true`
- ledger failures：空
- zero-grad fraction：`0.0`

有效 actor/vLLM sampled-tensor digest：

- version 2：`076ca68aa826b7933cbc76fb77862a1edc682fa02154e03da8226e288017f28f`
- version 3：`e72701a91a57ee36032fa6979a26f5bf86f746dd28e8f0e5478dd907884a237a`
- 结论：显式 resume 后的 step3 确实产生参数变化；最终审计同时要求两个 vLLM worker 对 version 3 做一致 ack。

step3 执行信号：

- grad norm：`0.6449196338653564`
- policy-gradient loss：`-0.021548985814054806`
- reward mean：`0.30259260535240173`
- advantage min/max：`-0.3100000023841858 / 0.3100000023841858`

## 3. 允许写入正式报告的结论

在 r5 冻结的 H20 两卡、Qwen2.5-7B-Instruct、naive reward、strict-vLLM 配置下，fresh rollout→actor update→checkpoint2→显式 resume→step3 rollout/update→checkpoint3，以及 actor→两名 vLLM worker 的权重同步链路已通过。

Gate A 证明的是执行链闭合，不证明模型已经学会 memory，不证明 QA 指标提高，也不证明 Path、NCR、SMSB、Tetrad、pair reward 或其他机制成立。

## 4. 通过后的只读验收

使用 `scripts/h20/accept_qwen25_7b_gatea_r5_pass.sh`。它会失败即停，并检查：

- 冻结分支、提交、干净工作树；
- P0/P1/final 三份证书；
- 独立只读复算仍为 `GATE_A_PASS`；
- P0/P1/P2 与 A1–A5；
- final report 所绑定的 append-only ledger 前缀；
- final report 后恰好追加 step2 inventory、step3 inventory、final audit result 三条记录；
- checkpoint2/3 当前 inventory 与冻结报告一致；
- resume-loaded version 2 与 post-update version 3 各有 rank 0、1 的 vLLM ack；
- effective actor/vLLM digest 一致且 version 2→3 发生变化；
- 两个 rank 的 optimizer step 均在 resume 后前进；
- step3 signal 唯一且没有非有限指标。

脚本不会使用 `--write-report`，不会启动 rollout、训练或 step4。

只读验收成功后，可使用 `scripts/h20/archive_qwen25_7b_gatea_r5_evidence.sh` 生成小型证据归档。它只复制证书、ledger、两份正式日志、seed audit、冻结合同、Git bundle 和现场重新生成的只读复验结果；不会复制约 28GB 的 checkpoint shard。checkpoint 的逐文件路径、大小和 SHA-256 已由 final report 冻结，原 checkpoint 目录仍须保留。

## 5. 必须原样保留的证据

- `p0_preflight.json`
- `p0_resolved_manifest.json`
- `p1_audit_report.json`
- `gate_a_final_report.json`
- `gate_a_execution_ledger.jsonl`
- fresh 与 resume 原始日志
- fresh step2 和 resume step3 的 model、optim、extra、data 文件
- rollout seed/trajectory identity 审计文件

不得覆盖、重命名或清理这些文件。最终报告已经生成，不能再次运行 `--write-report`。

## 6. 下一阶段只准备、不执行

Gate A PASS 不自动授权下一项 GPU 实验。下一阶段必须使用独立分支、独立提交、独立 manifest、独立输出目录和新的 P0 授权。

建议先准备的最小科学门：

1. 固定 4 个数据样本、每个 `n=2`，只验证 stable identity 与成对轨迹映射。
2. 在小门通过后才冻结 fixed-128 的统一 anchors。
3. 分别定义 raw direct（R）与 recurrent inference-only（I）；两者必须继续使用 vLLM。
4. corrected Original 的 2→25→50→100→200 仅作为候选顺序；stop rule、终止规则和裁决口径必须在看结果前冻结。

在新的冻结合同得到明确授权前，不启动 step4、R/I/T25、T50/T100/T200、SMSB/Tetrad、legacy Path、pair reward 或默认 NCR。

# Agent Memory 代码研究

本目录承接 Trainable Agent Memory 2024–2026 的代码研究，只读使用论文精读 batch，不改写上游 Memory_Paper_Screening_Pipeline。

## 当前结构

- repository_code_map_v1.md / .csv / .json：40 篇官方代码或官方生态条目的第一版总表。
- mainline_deep_dive_plan_and_findings_v1.md / .json：第一条主线的源码定位、commit、复现优先级与后续计划。
- repos/：第一条主线官方仓库的浅克隆；未下载模型、数据或依赖。
- experiments/UMA-BudgetedEvidenceMemory-20260830/：基于 UMA official commit `768f9620231bae11264771f59e43a4839506cf94` 的本地实验副本，承载 BEM 与 Adaptive Memory Computation 原型。
- tools/：轻量复现与分析脚本，包括 response 生成、strategy matrix、no-leak router gate、metric 汇总。
- results/：从 s05m 拉回的可复现矩阵与 router 结果。
- runs/：从 s05m 拉回的原始 response、matched deterministic strategy matrix、router 和 paired factorization 结果。
- idea-stage/：当前论文 idea 设计、hypothesis gate 与框架文档。
- refine-logs/：baseline 复现、失败诊断和实验 tracker。

## 输入来源

- batches/Trainable_Agent_Memory_2024_2026 的长篇精读、训练表、公共元数据、增强记录、code_metadata、PDF 与 completion audit。
- 官方 GitHub 仓库当前默认分支浅克隆，commit 固定在主线深挖表。

## 证据与失败恢复

- 40 是“有官方代码/官方生态链接的论文条目数”，不是互异仓库数。
- code_metadata 当前 38 份；010 GLoW 与 044 ElasticMem 有官方链接但缺本地快照，后续从官方仓库补抓。
- Fine-Mem 与 MemoPilot 当前 commit 为 README-only，不能据论文描述推断源码路径。
- GRU-Mem 无已核验官方仓库；保留论文证据，不借用 ReMemR1 冒充官方实现。
- 网络/API 失败时按：本地浅克隆 → 官方 GitHub 页面/raw → 论文 PDF/项目页；判断均保留 URL、源码路径和 commit。

## 复现边界

最初阶段只做静态审计和轻量 smoke test。2026-08-30 用户进一步授权尽早启动 baseline 与方法实验；但仍不在未核验的机器上盲目下载大模型或启动昂贵训练。最小复现实验先验证配置解析、数据 schema、reward 单元与 agent loop 状态转换，再在 2×H20 上执行 100-step smoke 和 48 小时门控实验。

## 当前研究主线

- 2026-08-30 用户明确要求放弃上一轮 counterfactual/SCAR 作为主动主线；相关代码与记录仅保留为历史归档，不作为本轮 baseline 或 idea 起点。
- 当前优先级改为先复现 baseline，再在充分代码审查和实验认知基础上形成新方法。MemRL 已完成只读 smoke，真实运行受 endpoint/key 阻塞；UMA 已跑通本地小模型 generation baseline，并成为当前可执行方法实验底座。
- 当前主动 idea 是 Adaptive / Dynamic Memory Computation：先证明固定 memory pipeline 的结构性不足、oracle routing headroom 是否足够、以及最优策略能否从状态预测。只有 supervised gate 通过后再进入 learned router/RL。
- `idea-stage/PROBLEM_FIRST_RESEARCH_RESET_V1.md`：当前可继续深化的非 counterfactual 问题导向研究路线。
- `idea-stage/BASELINES_AND_IDEAS_V2_DMC.md`：当前 baseline、idea 和执行顺序。
- `idea-stage/DMC_HYPOTHESIS_GATE_20260830.md`：HotpotQA-200 与 LoCoMo-small1 的 strategy matrix / oracle gap / router gate。
- `refine-logs/HOTPOTQA200_MATCHED_DET_AUDIT_20260830.md`：修复 batch/prompt/temperature/EM/split confound 后的 HotpotQA-200 matched deterministic 审计。
- `LOCAL_CODE_LINEAGE.md`：原始 baseline、实验副本、代码 diff 与审查入口。
- `experiments/CMIMem-SCAR/`：历史探索分支，保留本地 Git 历史；当前不继续投入。

## 当前验证与失败记录

- 2026-08-30 已安装 Auto-Empirical Research Skills 到 `~/.codex/skills/auto-empirical-research-skills`，并加载其中 `replication-driven-research` 作为复现实验记录规范。
- s05m 可用作轻量 baseline 机器：GPU1 空闲，MemRL 官方仓库位于 `/data/manxin/agent_memory_rl/baselines/MemRL-c1b322c`，commit `c1b322ca43de36ddf64c6712f89d0095bfc35ce0`，工作区干净。
- 本机到 s05m 的大文件上传不稳定：`scp` 与 `ssh cat` 均出现半截文件或断连；但远端已有完整 MemRL 仓库，因此本轮不再依赖 bundle 传输。
- s05m `newMamba310` 环境已补装 `openai`、`mem0ai==1.0.1`、`memoryos==1.0.0`、`chonkie==1.2.1`、`tiktoken==0.11.0`；MemRL 配置解析、BigCodeBench full 数据读取、MemoryService 初始化 smoke 均通过。
- s05m 当前未设置 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `DASHSCOPE_API_KEY` / `MODELSCOPE_API_TOKEN` / `HF_TOKEN`，因此尚不能跑出真实 LLM benchmark 数字。下一步需要提供 OpenAI-compatible LLM 与 embedding endpoint/key，或改由 H20 侧按 runbook 启动本地模型 endpoint。
- UMA 官方仓库已在 s05m 跑通第一条原始 baseline 生成：`concat` + 官方 HotpotQA-200 + 本地 `Qwen2.5-0.5B-Instruct` 32K vLLM，输出 128/128 条非 ERROR 响应；轻量指标已保存。这个结果只证明代码主流程和数据路径可复现，不作为论文级性能结论。
- UMA `concat` + locomo10 的 8K 服务尝试失败：1986/1986 条为 `ERROR_API_CALL`。诊断原因是长记忆 concat prompt 与服务上下文窗口/官方固定 `max_tokens=8192` 不匹配；后续长记忆复现需要 H20 上更长上下文模型或调整推理服务预算。
- 已在 UMA 实验副本中实现 `budgeted_evidence` 与 `adaptive_memory` 两类 agent，并在 s05m 本地 `Qwen2.5-0.5B-Instruct` 32K vLLM 上完成 HotpotQA-200 与 LoCoMo-small1 smoke。
- HotpotQA-200 depth0 strategy matrix：best fixed `adaptive_h` F1 0.3017，oracle F1 0.4556，gap +0.1539；no-leak cheap-feature ridge router F1 0.3025，只恢复 0.55% oracle gap。
- LoCoMo-small1 depth0 strategy matrix：best fixed `adaptive_h` F1 0.1743，oracle F1 0.3276，gap +0.1533；no-leak cheap-feature ridge router F1 0.1486，低于 best fixed。
- 当前结论：H1/H2 已有两个数据集的探索性支持；H3 未通过。下一步不是马上 RL，而是补 operation-depth variants 与 trace-derived state features。
- 远端官方 evaluator 因缺少 `backoff`/`rouge_score` 可选包不能直接加载；替代方案是 `tools/run_uma_generation_light.py` 生成 UMA-compatible JSONL，再用 `tools/build_strategy_matrix.py` 与 `tools/summarize_response_metrics.py` 生成可复现指标。
- HotpotQA-200 matched deterministic protocol repair 已完成：`no_memory`、`concat_single`、`bm25`、`bm25_filter`、`mmr`、`mmr_filter`、`recent`、`recent_filter`、`adaptive_h` 全部 128/128 response、0 error。结果位于 `runs/strategy_matrix_hotpotqa200_matched_det_v0`，原始 response 位于 `runs/hotpotqa200_matched_det_responses`。
- 修正后固定最佳仍是 `adaptive_h`，F1 0.2985；oracle F1 0.4388，gap +0.1404。`BM25+Filter` 只比 `BM25` 高 +0.0064，说明简单 sentence-filter 主效应不足以解释 Adaptive-H；下一步优先做 residual alignment，而不是直接上 H20/RL。
- P1.5 residual alignment 已完成：`adaptive_force_filter` 掉到 F1 0.1997，与 `bm25_filter` 相同；`adaptive_no_write_skip` 保持 F1 0.2985。当前最强解释是：Adaptive-H 的价值不在 generic sentence filtering 或 WRITE/SKIP，而在 selective stop/continue，即知道什么时候不该继续 refine。
- P2/P3 已完成第一轮：HotpotQA-200 上 D1 (`RETRIEVE -> ANSWER`) 固定最佳 F1 0.3486，D2/D3 更深路径平均更差但 oracle 仍有 +0.0819；LoCoMo-small 上 D2 固定最佳 F1 0.1896，oracle +0.0889。结论：memory computation depth 非单调，且跨任务最优深度不同；下一步应训练 semantic STOP/CONTINUE controller，而不是 flat multiclass router。
- P4 已完成 Selective Escalation / Memory Computation Value 离线审计。新增 `tools/build_escalation_dataset.py` 与 `tools/train_escalation_value_controller.py`，从 D0-D3 matrix 构造 immediate/lookahead/expand value target、temporary valley 统计，并用无下载 lightweight value controller 做 no-leak 评估。HotpotQA 上 strict temporary valley 为 6/128；LoCoMo-small 为 6/199。HotpotQA 在 λ=0.02 时 query-only controller 轻微高于 always-D1；LoCoMo-small 上 evidence-aware controller 有探索性正信号但 single-group 不可确认。下一步不是直接 H20/RL，而是 failure taxonomy、structured evidence features、nested-D3 protocol repair 与 LoCoMo-10 group split。
- `refine-logs/P4_SELECTIVE_ESCALATION_VALUE_AUDIT_20260831.md`：本轮 value supervision、escalation map、controller 结果与下一步判断。
- `idea-stage/APMC_VALUE_MODEL_PLAN_20260831.md`：将当前 idea 收紧为 Memory Computation Value Model 的方法计划。
- `refine-logs/P6_P7_REVIEWER_RISK_AUDIT_20260831.md`：针对最新审稿风险的修复审计，包括严格 D3、semantic-lite refinement、bootstrap CI、cost-aware utility 与 HotpotQA matrix leakage 分层判断。
- P6/P7 新增结果已拉回本地：`runs/hotpotqa200_p6_p7_responses/` 与 `runs/strategy_matrix_hotpotqa200_p6_p7_operator_strict_v0/`。HotpotQA-200 上 D1 仍是最佳固定 depth，F1 0.3486；D2-lex 与 D2-tfidf 均为 0.1997；严格 D3 为 0.1428；oracle 0.4196，仍有 +0.0710 headroom。
- 当前论文表述进一步收紧：可以说 “additional memory processing is not automatically beneficial”；暂时不能说 “deeper memory computation intrinsically hurts”；也不能说 controller 已经稳定利用 intermediate memory state。下一步必须做真正 semantic/LLM refinement 与 `X_Q/X_QM/X_QR/X_QRS` incremental-information controller audit。
- `refine-logs/P8_BRIDGE_AWARE_OPERATOR_AUDIT_20260831.md`：补做真正不同于 lexical gate 的 bridge-aware refinement。HotpotQA-200 上 D2-graph_bridge F1 0.2244，高于 D2-lex 0.1997；D3-strict-graph_bridge F1 0.1930，高于 D3-strict-lex 0.1428；但两者仍低于 D1 0.3486。当前解释升级为：D2/D3 失败部分来自 lexical relevance bias，但更本质问题是 evidence sufficiency 与 dependency preservation。
- `refine-logs/P9_INCREMENTAL_INFORMATION_CONTROLLER_PREAUDIT_20260831.md`：新增 `Q/Q+S/Q+R/Q+R+S/Q+shuffledR` controller 输入族并做 HotpotQA exploratory 预审计。当前结果显示 hashed retrieved evidence representation 明显伤害 controller；这不是方法失败，而是 state encoder 设计风险被提前暴露。正式 P9 必须在新 operator matrix 上重建 value dataset，并保留 shuffled-evidence control。
- `refine-logs/P10_BENCHMARK_ROLE_RESET_20260831.md`：重新定义 benchmark 角色。HotpotQA 改为 controlled mechanism microscope；LongMemEval 升为 primary idea-validation benchmark；LoCoMo 改为 conversational transfer；MemoryAgentBench/LongMemEval-V2 后置为 agentic generalization。新增 `tools/audit_hotpot_modified_metadata.py` 与 `tools/build_longmemeval_splits.py`，并给 `tools/run_uma_generation_light.py` 增加 `--qid-manifest` / `--limit-samples`。
- `refine-logs/LONGMEMEVAL_S05M_RUNBOOK_20260831.md`：LongMemEval 下一步在 s05m 上的可执行 runbook。本轮没有自动下载 benchmark 数据；等用户确认后，先下载/构建 `longmemeval_s.json`，生成 dev/val/confirm manifest，再跑 dev10 D1/D2 smoke。
- `refine-logs/P11_LONGMEMEVAL_OPERATION_VALUE_PILOT_DESIGN_20260831.md`：LongMemEval 第一轮 operation-value pilot 协议。P11 将旧的 150/100/250 split 改为默认 `structural_pilot / abstention_probe / locked_confirm`，并把第一轮 baseline 固定为 P0、P1、P2-S、P2-T 与后续 OracleEvidenceReader。新增 `tools/audit_longmemeval_metadata.py` 与 `tools/build_longmemeval_operation_value_matrix.py`，离线生成 `R_i(a) / C_i(a) / EvidenceRecall_i(a) / DeltaU_i(a)`，gold evidence 只用于离线诊断，不进入推理。
- `refine-logs/P11_LONGMEMEVAL_DOWNLOAD_AND_DEV12_SMOKE_AUDIT_20260831.md`：用户确认后已在 s05m 下载/构建 LongMemEval-S 与 oracle，完成 metadata audit、locked-pilot manifest、type-balanced dev12 P0/P1/P2-S/P2-T/OracleEvidenceReader smoke。五条线均 12/12、0 ERROR；结果位于 `runs/longmemeval_dev12_balanced_*` 与 `runs/longmemeval_dev12_balanced_operation_value_matrix_with_oracle`。
- `refine-logs/P11_LONGMEMEVAL_PILOT84_SURROGATE_MATRIX_AUDIT_20260831.md`：LongMemEval structural_pilot 84 条已完成 P0/P1/P2-S/P2-T/OracleEvidenceReader 五条线，全部 84/84、0 ERROR。surrogate F1 结果显示 P2-S/P2-T 存在 positive/negative marginal utility 雏形，但当前仍受 0.5B reader 与 surrogate metric 限制，不能作为论文主结论；结果位于 `runs/longmemeval_pilot84_*` 与 `runs/longmemeval_pilot84_operation_value_matrix_with_oracle`。
- `refine-logs/P12_MEMORY_COMPUTATION_MDP_AND_RL_PAPER_FORMULATION_20260831.md`：将论文主线从 supervised MCVM 推进到 budgeted sequential memory computation MDP，明确 RL 的必要性必须来自 operation 改变未来 evidence state，而不是给 controller 后置 GRPO。
- `refine-logs/P12_STOP_RELATIVE_ADVANTAGE_DIAGNOSTIC_20260831.md`：历史记录，已由 P14 修正。旧版 “GRPO-like false-continue” 不能直接解释为 GRPO failure，因为 action-independent baseline 不改变理想期望 policy-gradient 方向。
- `refine-logs/P13_RL_FAILURE_DIAGNOSIS_AND_IMPLEMENTATION_PLAN_20260831.md`：历史实现顺序，已由 P14 修正。当前不直接进入 SRPO/GRPO 训练，先做 reader/evaluator validity 与 sequentiality gate。
- `refine-logs/P14_SRPO_THEORY_CORRECTION_AND_PAIRED_ADVANTAGE_DIAGNOSTIC_20260831.md`：PSRA/SRPO 理论纠偏。新版 `tools/diagnose_stop_relative_advantage.py` 输出 pairwise order mismatch 与 advantage variance；pilot84 上 pairwise mismatch 为 0，paired STOP-relative variance 高于 group-relative，因此 SRPO/PSRA 暂不作为核心算法。
- `refine-logs/P14_READER_EVALUATOR_CALIBRATION_AND_SEQUENTIALITY_PLAN_20260831.md`：当前 source of truth。新增 `memory_sequence` agent、固定 sequence36 manifest、运行 STOP/REFINE/EXPAND/REFINE→EXPAND/EXPAND→REFINE。0.5B + surrogate F1 下，36 条六类均衡样本全部 0 ERROR，reward 与 utility 的 composition/order/budget 三个 exploratory gate 均为 true；下一步必须接 official LongMemEval judge 与更强 reader sanity。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/agents/memory_sequence_agent.py`：P14 新增 sequence probe agent，支持 `AMC_SEQUENCE=stop/refine/expand/refine_expand/expand_refine`。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/build_manifest_subset.py`：固定 manifest 子集工具。P14 sequence36 hash 为 `830218939b191f36c2a1c9c126e8592437909958ef41c748c86858191c364d3f`。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/analyze_sequence_budget_matrix.py`：自动分析 composition/order/budget gate。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/convert_uma_responses_to_longmemeval_hypothesis.py`：把 UMA `qid/response` 转成官方 LongMemEval `question_id/hypothesis`。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/run_p14_sequence36.sh`：s05m 远端复现脚本，固定 0.5B reader、top20 retrieve、expand20、temperature 0。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/run_p14_official_longmemeval_judge.sh`：官方 LongMemEval evaluator 调用脚本；未设置 `OPENAI_API_KEY` 时会退出，不会误触发付费调用。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/analyze_p14_sequence_mdp_evidence.py`：P15 新增精细 MDP-evidence audit，把 ordering 拆成 answer-quality 与 cost/context 两部分，把 composition 改为 margin audit，把 budget 改为 value gain / actual steps / unused-budget，并计算 path-dependent marginal value。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/run_p14_expandq_control.sh`：P15 新增 Expand-Q control，同样 sequence36/top20/expand20，只把 `AMC_EXPAND_MODE` 改成 `bm25_tail`，用于检查 state-conditioned GROW 是否超过 query-only retrieve-more。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/build_metric_agreement_table.py`：P16 新增 metric-agreement 表工具。official judge matrix 出来后，用于比较 surrogate F1 与 official judge 对 SHRINK/GROW helpful、composition、RE/ER ordering、path shift 等现象的一致性。
- `repos/LongMemEval/`：官方 LongMemEval evaluator 浅克隆，commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`，官方链接 `https://github.com/xiaowu0162/LongMemEval`。
- `runs/longmemeval_sequence36_p14_0p5b_top20/`：P14 sequence36 原始 responses、traces、operation-value matrix、sequence/budget analysis 与 official judge hypothesis 文件。
- `refine-logs/P15_PATH_DEPENDENT_MDP_EVIDENCE_AND_EXPANDQ_CONTROL_20260831.md`：P15 当前核心判断。sequence36 上 reward-space composition gain 为 5/36，其中 3/36 超过 0.1、1/36 超过 0.2；path-dependent marginal value 的 `any_abs_path_shift>0.05` 为 12/36，`>0.1` 为 9/36；B2 真正 value gain 为 5/36，同时大量样本即使给 B=2 也不需要耗尽预算；Expand-State 相比 Expand-Q 有少数正例但不压倒，说明当前 GROW 是 probe，不是算法贡献。
- `refine-logs/P16_STRICT_PATH_DEPENDENCE_AND_METRIC_AGREEMENT_PREP_20260831.md`：P16 修正 P15 的 sign-change 口径。reward 下严格 polarity reversal 为 EXPAND-after-REFINE 0/36、REFINE-after-EXPAND 2/36；utility 下为 0/36 与 8/36。P16 还新增 fixed-seed bootstrap CI：reward-space mean abs path shift 0.1149，95% CI [0.0430, 0.2049]；`P(abs shift>0.05)` 为 12/36，95% CI [0.1944, 0.5000]。当前只能说 RL-MDP hypothesis provisionally passes under exploratory surrogate evaluation，不能说 sequential RL necessity 已被证明。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/agents/progressive_depth_agent.py`：P11 新增 `AMC_FILTER_MODE=temporal_session`，并统一为 lexical/tfidf/graph/temporal refinement 记录 `admitted_source_indices`，支持 LongMemEval session-level evidence recall。
- `refine-logs/P17_CLOSED_LOOP_FEEDBACK_VALUE_DIAGNOSTIC_20260831.md`：P17 将 RL necessity gate 进一步收紧为 closed-loop feedback value。基于 P14/P15 sequence36、0.5B reader、surrogate F1 与 numeric trace，构造 72 个 next-action value prediction task；full/compact 两种 trace 特征、reward/utility 两个目标、alpha=0.1/1/10/100/1000 下均没有稳定 closed-loop MAE 优势。因此 M3 暂未通过，当前不进入 RL/PPO/GRPO/MCVM 训练。下一轮必须用 `AMC_TRACE_STATE_TEXT=1` 重新记录完整 post-state text，并接 official LongMemEval judge 与 1.5B reader 复核。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/diagnose_closed_loop_feedback_value.py`：P17 新增 closed-loop feedback-value 诊断脚本，支持 `--feature-mode full|compact`，输出 summary/predictions/features 与 alpha sensitivity 表。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/agents/adaptive_memory_agent.py`：P17 新增 opt-in trace state hook。默认只写 `context_sha1` / `final_context_sha1`；设置 `AMC_TRACE_STATE_TEXT=1` 时才把每步 operation 后的完整 `state_text` 写入 trace，供下一轮 full-state closed-loop diagnostic 使用。
- `refine-logs/P18_RL_NECESSITY_GATE_RESET_AND_STATIC_PROGRAM_SELECTOR_20260831.md`：P18 修正 P17 的理论门槛。P17 否定的是 numeric post-operation summary 的额外预测价值，不是否定 sequential memory-control formulation。新的 RL necessity gate 改为 R1 heterogeneous operation value、R2 nontrivial sequential composition、R3 enumeration gap。正式实验必须加入强非 RL baseline：Static Program Selector，即从 `(q,E0,B)` 一次性预测 whole program。当前 sequence36 smoke 中 Static Program Selector 在 reward/utility 下多数 alpha 超过 LOOCV best fixed，但仍离 oracle 较远；因此短 horizon 还不能证明 RL necessity。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/train_static_program_selector.py`：P18 新增 strong non-RL baseline 脚本。默认 program set 为 `stop/refine/expand/refine_expand/expand_refine`，明确排除 P15 的 `expand_q` control。
- `runs/longmemeval_sequence36_p14_0p5b_top20/static_program_selector_p18_v1/`：P18 Static Program Selector 离线 smoke 结果。关键表为 `static_program_selector_alpha_sensitivity.csv`；compact alpha=100 下 reward static-fixed 为 +0.0405，utility static-fixed 为 +0.0659。
- `refine-logs/P19_PROTOCOL_REPAIR_ONLINE_STATIC_AND_WITHIN_TYPE_AUDIT_20260831.md`：P19 修正 P18 的 hidden privilege 与 alpha selection optimism。Static Selector 和 P17 closed-loop diagnostic 均默认不使用 `question_type`；`--include-question-type` 仅作为 privileged upper-control。Static Selector 默认改为 nested CV，并用 `(value, -cost)` 做 oracle/fixed tie-break。修正后 Static-Online nested CV 几乎不再强于 best fixed：reward +0.0011，utility -0.0125；说明 P18 的强结果主要来自 privileged metadata 与 alpha 选择乐观。但 within-type audit 显示 P11 vs D1 utility 有 16/18 个 type×operation cell 同时存在正负样本，operation-value heterogeneity 不能被 `question_type` 完全解释。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/audit_within_type_operation_heterogeneity.py`：P19 新增 offline heterogeneity 审计脚本，按 question_type 统计 `P(Delta>0|type)`、`P(Delta<0|type)`、方差与 mixed-sign cell rate。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/audit_feature_namespace_leakage.py`：P19 新增 feature namespace 审计脚本，online feature 文件若含 `privileged_` 或 `offline_` 前缀会直接 fail。
- `runs/longmemeval_sequence36_p14_0p5b_top20/static_program_selector_p19_protocol_repair/`：P19 修正后的 Static-Online/Static-Privileged、nested/fixed-alpha 分解结果。
- `runs/longmemeval_sequence36_p14_0p5b_top20/closed_loop_feedback_value_p19_protocol_repair/`：P19 修正后的 online/privileged closed-loop feedback diagnostic。
- `runs/longmemeval_within_type_heterogeneity_p19_v1/`：P19 P11/P14 within-type heterogeneity 审计结果。
- `refine-logs/P20_THRESHOLD_HETEROGENEITY_AND_VALIDITY_RUNBOOK_20260831.md`：P20 将 P19 的宽松 mixed-sign 审计升级为 thresholded non-trivial heterogeneity 与 type variance decomposition。注意：P20 中 `type R² / residual variance` 的命名与分母口径已由 P21 修正，后续不要再直接引用 P20 的旧方差术语。
- `refine-logs/P21_H2_STATE_PREDICTION_LADDER_AND_STAT_REPAIR_20260831.md`：P21 修正 P20 统计命名，将方差表改为 `total_ss / between_type_ss / within_type_ss / type_eta_squared / within_share / within_mse`，并把 `oracle_d1` 从 legal online 主表剥离为 privileged upper-bound。P11 pilot84 vs D1 reward 下，legal `p2s_graph` 在 ε=0.10 时 mixed type cells 为 4/6、`type_eta_squared=0.1401`、`within_share=0.8599`；legal `p2t_temporal` 为 3/6、`type_eta_squared=0.0141`、`within_share=0.9859`。当前严格 claim 是：合法 memory operation value 存在不能被 benchmark-defined `question_type` 充分解释的样本级异质性；尚不能说 within-share 就是 evidence-state variance。P21 同时从 P11 locked confirm 404 中构建 H2 dev80 与 resealed confirm324 manifest，进入 Semantic State Prediction Ladder 之前的 split/protocol 已固定。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/run_p20_sequence36_model_state_trace.sh`：P20 新增 guarded 1.5B/H20 structural replication 脚本。不下载模型、不启动服务；要求已有 OpenAI-compatible endpoint 和 `MODEL_NAME`，默认 `AMC_TRACE_STATE_TEXT=1`。
- `runs/longmemeval_thresholded_heterogeneity_p20_v1/`：P20 thresholded within-type heterogeneity、type variance decomposition 与 operation-level overview。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/build_p21_h2_dev_confirm_split.py`：P21 新增 LongMemEval H2 dev/confirm split 构建脚本，只使用 qid 与 question_type metadata，不复制 gold answers 或 outcome labels。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/run_p21_h2_dev80_matrix.sh`：P21 新增 s05m dev80 前置 matrix 脚本，运行 STOP/REFINE/EXPAND/EXPAND-Q control，默认 0.5B reader、top20、temperature 0、`AMC_TRACE_STATE_TEXT=1`。
- `runs/longmemeval_thresholded_heterogeneity_p21_stat_repair_v2/`：P21 统计修正版 heterogeneity 输出，主表区分 `legal_online_operation` 与 `privileged_upper_bound`。
- `runs/longmemeval_manifests_p21_h2/`：P21 H2 split 输出，dev80 canonical hash 为 `722a49db963fcfde8662d9e0ed31c7c05f22b898f647c6d6324c35559d59d772`，resealed confirm324 canonical hash 为 `7e8d0095902cb6b9da02a74ce10c796d4d73290b7481002f42ece0b13be84a28`。
- `runs/longmemeval_p21_h2_dev80_0p5b_top20/`：P21 dev80 前置 matrix 已在 s05m 完成并拉回本地。四条线 stop/refine/expand/expand_q 均为 80/80。0.5B+surrogate 下 `expand_q` 的 all-group reward/utility 均高于当前 state-conditioned `expand`，说明当前 GROW 实现不能直接作为核心方法贡献，必须进入 H2 semantic state prediction 与更强 GROW 设计。
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/build_h2_state_prediction_table.py` 与 `tools/train_h2_state_prediction_ladder.py`：P21 新增 H2 state-prediction table 与 lightweight hashing ladder trainer。dev80 smoke 使用 LOOCV + inner 5-fold alpha selection，比较 `M_Q / M_QE / M_QEStats / M_QtildeE`。当前结果未通过 H2：naive evidence hashing 没有稳定超过 query-only，也没有超过 shuffled-evidence control；policy 基本退化为选择 best fixed `expand_q`。下一步应换成 frozen semantic encoder / cross-encoder score，并优先预测 `refine` helpful/harmful tail。

## P22 更新：H20 scale sanity + retrieval pressure

新增复制包：

- Codex outputs：`/Users/xinman/Documents/Codex/2026-08-30/agent-memory-code-research/outputs/THIS_ROUND_P22_H20_SCALE_SANITY_PREP_20260831`
- zip：`/Users/xinman/Documents/Codex/2026-08-30/agent-memory-code-research/outputs/THIS_ROUND_P22_H20_SCALE_SANITY_PREP_20260831.zip`

推荐阅读：

1. `refine-logs/P22_H20_SCALE_SANITY_AND_RETRIEVAL_PRESSURE_20260831.md`
2. `refine-logs/H20_P22_COMMANDS_20260831.sh`
3. `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/run_p22_h20_scale_sanity.sh`
4. `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/summarize_p22_scale_sanity.py`
5. `runs/p22_scale_sanity_comparison/p22_scale_sanity_key_metrics_0p5b_k5_k10_k20_growq.csv`
6. `runs/p22_scale_sanity_comparison/p22_scale_sanity_0p5b_k5_k10_k20_growq.csv`

当前一句话结论：

> P22 暂不进入 RL。当前最优动作是用 H20 做一个很小的 7B inference-only structural replication，审判 0.5B + surrogate F1 下看到的 operation heterogeneity、path dependence、sparse useful computation 是否是 tiny-reader artifact。0.5B 的 K=5/K=10/K=20 pressure sanity 已显示：K=5 是更强 memory pressure regime，reward-side path shift 最大；K=10 在 sequence36 上 reward/utility 更好，提示 retrieval pressure 同时影响 evidence recall 与 long-context distraction。这个结果只支持“值得上 H20 做 scale/pressure sanity”，不支持“已经可以训练 RL”。

当前下一步优先级更新为：

1. H20 7B structural replication：sequence36，K=20 与 K=5，inference-only。
2. 若 1.5B 能在 3090 上跑通，补 1.5B K=20/K=5。
3. official LongMemEval judge 一有 key/endpoint 立刻补，不用 surrogate F1 做正式 claim。
4. 若 7B + retrieval pressure 下 heterogeneity/path shift/oracle gap 仍存在，再做 frozen semantic encoder / cross-encoder state signal。
5. 再做 capacity-matched Static-Online vs Sequential-Online。
6. 最后才考虑 RL。

## P22-H20 更新：7B scale sanity reported summary

新增记录：

- `refine-logs/P22_H20_7B_SCALE_SANITY_REPORTED_SUMMARY_20260831.md`

H20 专用执行 task 已汇报：7B + K=20/K=5 均完成，archive 位于 H20 `/data/cw/memagent_work/p22_h20_scale_sanity_results_20260831_043559.tgz`，SHA256 为 `40fa85d215e71beee9d38de4e960b35774db1b8f777d0b0333adc95bf32bb26e`。GitHub HTTPS 出口仍阻塞，archive 尚未本地回传。

当前 reported exploratory 判断：

> 7B 下 `refine` 的平均 reward 增益仍为 K20 `+0.1618`、K5 `+0.1547`；mean absolute path shift 为 K20 `0.1243`、K5 `0.1185`；非平凡 tail 与 strict reversal 仍存在。因此当前结果不支持“现象仅是 0.5B tiny-reader artifact”。但 n=36、metric 为 surrogate F1、archive 未本地复核，不能写成正式论文主结论。

## P23 更新：ANSWER vs SHRINK 模型尺度裁决

新增记录：

- `refine-logs/P23_ANSWER_VS_SHRINK_MODEL_CONDITIONING_PLAN_AND_0P5B_AUDIT_20260831.md`
- `refine-logs/P23_H20_7B_ANSWER_VS_SHRINK_REPORTED_SUMMARY_20260831.md`
- `refine-logs/H20_P23_COMMANDS_20260831.sh`
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/analyze_p23_answer_vs_shrink.py`
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/run_p23_answer_vs_shrink_h20.sh`
- `runs/longmemeval_p23_answer_vs_shrink_0p5b_dev80_k20/`
- `runs/longmemeval_p23_answer_vs_shrink_model_scale_reported/`

当前 0.5B dev80 K20 surrogate 结果：

> `REFINE` 相比 `STOP` 平均 reward 小幅提升 `+0.0165`，但因平均多耗费 `+1.6072` context-kchars，utility 下降 `-0.0156`。在 `eps=0.1` 下，`REFINE` 为 `12 win / 61 tie / 7 loss`，说明它不是 universally beneficial；非平凡 positive/negative tail 仍存在。77/80 个样本初始证据已完整，wins/losses 主要发生在 evidence-complete regime，提示 SHRINK 的核心风险不是单纯 evidence recall，而是压缩/保留/重排对 reader 的影响。

P23 下一步：

H20 已 reported 完成 7B dev80 K20 的 `stop/refine` 两条冻结路径，无 ERROR_API_CALL，exact commit `b5a4590e7ec270f64cc5edf5b15f32328ad47ea2`，raw SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`。7B 结果为 mean ΔR `+0.0285`、mean Δutility `-0.0043`、eps=0.1 下 `13 win / 56 tie / 11 loss`。因此 P23 的裁决是：

> 7B 并未把 0.5B 上看到的 ANSWER-vs-SHRINK heterogeneity 抹掉。REFINE 对更强 reader 的平均 reward 仍有小幅正增益，但不是无条件有益；single-session 类任务倾向受伤，knowledge-update / multi-session 更受益。当前主张应收紧为 model-conditioned and evidence-structure-conditioned memory operation value。由于仍是 surrogate F1 与 dev exploratory，不能直接宣称 official benchmark 或 RL necessity。

下一步：

1. P24 做 ANSWER-vs-SHRINK 的 legal semantic state identification，不让 `question_type` 或 gold labels 进入 online policy。
2. 用 `M_Q / M_QE / M_QEStats` 预测 `ΔR_refine` 的正负 tail，并做 0.5B/7B cross-scale consistency。
3. 若 semantic state 能稳定预测 tail，再进入 capacity-matched Static-online vs Sequential-online；否则继续停在机制诊断，不上 RL。

## P24 更新：state prediction smoke 与 model-conditioned label

新增记录：

- `refine-logs/P24_ANSWER_SHRINK_STATE_PREDICTION_SMOKE_20260831.md`
- `runs/longmemeval_p24_answer_shrink_state_prediction_0p5b_dev80_k20_reward/`
- `runs/longmemeval_p24_answer_shrink_state_prediction_0p5b_dev80_k20_utility/`
- `runs/longmemeval_p23_answer_vs_shrink_model_scale_reported/p23_0p5b_7b_per_qid_joined_reported.csv`
- `runs/longmemeval_p23_answer_vs_shrink_model_scale_reported/p23_0p5b_7b_cross_scale_summary_reported.json`

P24 结论：

> 第一轮 lightweight state-prediction 没有通过。`M_QE/M_QEStats` 没有稳定超过 query-only 或 shuffled-evidence control；reward policy 退化为 always-REFINE，utility policy 基本退化为 always-STOP。与此同时，0.5B 与 H20-reported 7B 的 per-qid `ΔR_refine` Pearson 约为 `-0.0018`，eps=0.1 下 strict non-tie overlap 只有 5 个。也就是说，heterogeneity 的存在跨模型尺度稳定，但具体样本标签强烈 model-conditioned。下一步方法必须学习 `Q(s,a;m)`，不能把 0.5B per-example oracle label 当作 7B 的固定监督。

## P25 更新：P23/P24 full code audit 与 protocol repair

新增记录：

- `refine-logs/P25_P23_P24_FULL_CODE_AUDIT_AND_PROTOCOL_REPAIR_20260831.md`
- `refine-logs/H20_P23_RAW_COMPACT_RESPONSE_EXPORT_COMMANDS_20260831.sh`
- `runs/longmemeval_p23_answer_vs_shrink_0p5b_dev80_k20_p25_audited/`

代码修正：

> `tools/analyze_p23_answer_vs_shrink.py` 已把 `initial_retrieval_features()` 从“allowed operation 中取最后一个 record”改为只取 first `RETRIEVE/RETRIEVE_RECENT`，并新增 response/trace duplicate 与 missing audit。默认 fail-closed；只有显式 `--allow-missing-trace` 才允许 legacy analysis 继续。

重新审计 0.5B dev80 K20 后，主结果不变，但 canonical definition 更清楚：

- response rows = 80，mapped qids = 80，duplicate qid/hash = 0；
- QA trace rows = 80，duplicate trace hash = 0；
- missing response/trace/first retrieve = 0；
- matrix-level complete = 77/80；
- canonical first-retrieve `Complete(E0)` = 76/80。

P24 训练脚本审计：

> `train_h2_state_prediction_ladder.py` 的 target sign、LOOCV、inner alpha selection、qid/prediction assembly、shuffled evidence 与 privileged metadata 使用路径已复核；未发现明显顺序错位或反号。负 Spearman 当前应解释为 naive hashed evidence representation failure，而不是 evidence state 不可预测。

后续所有 review bundle 固定包含源码本体、runner、metric、analysis、training script、manifest/hash、per-qid 表、summary 与 config；README 不再替代源码。

P25 补充更新：

> H20 P23 7B compact raw responses 已由 H20 task 回贴并本地保存。校验结果为 160 行，stop/refine 各 80 行，80 个 qid 全部成对，operation+qid 无重复，SHA1 格式问题为 0，stop/refine 的 query hash 与 expected-answer hash 均无 mismatch，并且 qid set 与 7B compact per-qid score table 完全一致。

新增文件：

- `runs/longmemeval_p23_answer_vs_shrink_model_scale_reported/p23_h20_7b_raw_compact_responses_reported.csv`
- `runs/longmemeval_p23_answer_vs_shrink_model_scale_reported/p23_h20_7b_raw_compact_response_audit.json`

## P25.5 更新：SHRINK 机制审计与 trace semantics repair

新增记录：

- `refine-logs/P25_5_SHRINK_MECHANISM_AUDIT_AND_TRACE_SEMANTICS_REPAIR_20260831.md`
- `runs/longmemeval_p23_answer_vs_shrink_0p5b_dev80_k20_p25_5_legacy_aware/`
- `runs/longmemeval_p25_5_shrink_mechanism_audit_dev80_k20/`

代码修正：

> `selected_indices` 现在只作为 legacy 兼容字段保留；新 trace schema 明确区分 `retrieved_source_indices` 与 `admitted_source_indices`。P23 analyzer 只有在 trace 中存在真实 `admitted_source_indices` 时才计算 canonical `Complete(E0)`，旧 trace 会标为 `selected_indices_legacy_fallback`。

重新审计旧 0.5B P23 trace 后：

- `initial_index_source_counts = {"selected_indices_legacy_fallback": 80}`；
- `canonical_initial_complete_available_count = 0`；
- `canonical_initial_complete_e0_rate = NaN`；
- `legacy_initial_complete_e0_rate = 0.95`。

因此旧的 76/80 只能叫 legacy selected-index completeness，不能再叫 canonical admitted-evidence completeness。

P25.5 机制审计使用已有 0.5B 本地表和 H20 reported 7B compact 表，不训练、不生成。7B 上机制方向更清楚：

- high integration：mean ΔR `+0.0693`；
- low integration：mean ΔR `-0.1541`；
- high preservation risk：mean ΔR `-0.1541`；
- low preservation risk：mean ΔR `+0.0978`；
- `offline_integration_score` 与 ΔR：Pearson `+0.2853`；
- `offline_preservation_risk_score` 与 ΔR：Pearson `-0.3131`。

当前主线收紧为：

> memory transformation trades off integration benefit against preservation risk; operation value is evidence-structure-dependent and reader-relative.

P25.5 H20 new schema 补充：

> H20 已在 exact commit `b8b2cfb7ef6ed1b88e5360129fa598b244a9b1fb` 上完成 new trace-schema rerun。protocol parity 全部通过，`trace_schema_counts = {retrieved_vs_admitted_v1: 160}`，`initial_index_source_counts = {admitted_source_indices: 80}`，`canonical_initial_complete_available_count = 80`，`legacy_selected_fallback_count = 0`。

关键结果：

- canonical initial `Complete(E0)` rate = `0.3375`；
- mean initial trace evidence recall = `0.60875`；
- eps=0.1 W/T/L = `12/57/11`。

因此旧的 `legacy_initial_complete_e0_rate = 0.95` 不能用于论文机制 claim。下一步必须把 `retrieved ≠ admitted ≠ visible` 作为核心 protocol，并在 P26 前先做 admission-aware mechanism audit。

新增记录：

- `refine-logs/P25_5_H20_NEW_SCHEMA_RESULT_UPDATE_20260831.md`
- `runs/longmemeval_p25_5_shrink_mechanism_audit_dev80_k20/p25_5_h20_new_schema_reported_summary.json`

## P26 更新：Admission vs Transformation Factorization

新增记录：

- `refine-logs/P26_ADMISSION_TRANSFORMATION_FACTORIZATION_PLAN_20260831.md`
- `refine-logs/H20_P26_ADMISSION_TRANSFORMATION_COMMANDS_20260831.sh`
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/analyze_p26_admission_transformation_factorization.py`
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/run_p26_admission_transformation_h20.sh`

P26 是一次 operation contract reset，不是新增漂亮 claim。最新 source of truth：

- `VALID`：P25.5 H20 new trace schema / parity / alignment 审计；
- `EXPLORATORY`：legacy `refine` 相对 `stop` 的 win/tie/loss heterogeneity；
- `CONFOUNDED / NEEDS RERUN`：把 legacy `refine` 直接解释成 strict visible-state `SHRINK(W0)`，或直接解释成 integration-vs-preservation mechanism；
- `INVALID`：声称旧 `refine` 严格只作用于 reader-visible `W0`。

核心状态定义更新为：

```text
M  = long-term memory
C0 = retrieved candidate pool before context-budget admission
W0 = admitted working memory actually visible to the reader
```

旧 `AMC_SEQUENCE=refine` 保留为 backward-compatible legacy action，不静默改变含义。新实验使用两个显式 action：

- `AMC_SEQUENCE=shrink_visible`：只对 `W0` 做 transformation，不引入新来源；
- `AMC_SEQUENCE=repack_candidates`：在固定 `C0` 内重新分配 admitted working memory。

P26 第一张表必须按三类分解：

- Group A：`Gold ⊄ C0`，retrieval incomplete；
- Group B：`Gold ⊆ C0` 且 `Gold ⊄ W0`，retrieved-complete but admission-incomplete；
- Group C：`Gold ⊆ W0`，admitted complete。

在 P26 H20 结果出来前，不能再写 `SHRINK` 已有效、integration/preservation mechanism 已成立、或 RL necessity 已成立。

### P26 H20 reported result

新增记录：

- `refine-logs/P26_H20_ADMISSION_TRANSFORMATION_REPORTED_RESULT_20260831.md`

H20 已在 exact commit `674a306ad23a6661b0791c2e59fabef41b1e6dd8` 完成 P26 dev80 K20，三条路径 `stop / shrink_visible / repack_candidates` 均为 80/80，无 `ERROR_API_CALL`。compact parity 240 行，missing qid / prompt / schema / admitted hash / retrieved hash 均为 0，schema 仅为 `retrieved_vs_admitted_v1`。

Admission decomposition：

- Group A：`Gold ⊄ C0` retrieval incomplete，3/80；
- Group B：`Gold ⊆ C0` 且 `Gold ⊄ W0` admission incomplete，50/80；
- Group C：`Gold ⊆ W0` admitted complete，27/80。

核心结果：

- `shrink_visible` all-group mean ΔR `+0.01038`，mean ΔU `+0.02428`，W/T/L@0.1 = `3/73/4`，admission rescue `0/80`；
- `repack_candidates` all-group mean ΔR `+0.02852`，mean ΔU `-0.00431`，W/T/L@0.1 = `13/56/11`，admission rescue `50/80 = 62.5%`；
- Group B 中 `repack_candidates` mean ΔR `+0.10661`，W/T/L = `10/36/4`；
- Group C 中 `repack_candidates` mean ΔR `-0.11694`，W/T/L = `2/19/6`；
- Group C 中 `shrink_visible` mean ΔR `-0.00468`，W/T/L = `0/26/1`。

当前 source of truth：

> legacy `refine` 的主要正收益来自 admission rescue / working-memory reallocation，而不是 strict visible-state SHRINK。论文主问题应从 “SHRINK/REFINE 是否有用” 收紧为 “有限 working-memory budget 下，如何把 retrieved candidate pool `C0` 合法分配进 reader-visible `W0`，同时救 admission failure、避免 disruption”。

下一步 P27 应做 strong online admission baselines，不进入 RL。

## P27 更新：Strong Online Admission Baselines

新增记录：

- `refine-logs/P27_STRONG_ONLINE_ADMISSION_BASELINES_PLAN_20260831.md`
- `refine-logs/H20_P27_ADMISSION_POLICY_BASELINES_COMMANDS_20260831.sh`
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/run_p27_admission_policy_baselines_h20.sh`

P27 固定 operator contract 为 `REPACK_CANDIDATES: fixed C0 -> W1`，只改变 legal admission/repack policy：

- `lexical_bm25`
- `tfidf_jaccard`
- `graph_bridge`
- `temporal_session`

P27 是 RL 前的 strong static admission baseline，不训练、不调 prompt、不使用 `question_type` 或 gold labels 作为 online feature。它要回答的是：

> P26 发现的 Group B admission rescue 与 Group C disruption tradeoff，能否被一个强的合法 online admission policy 解决？

若强 static admission baseline 已能解决，则不支持 RL necessity。只有当 strong static planner 和 supervised sequential baseline 都不能解决，且 official judge/7B/confirm split 都保留该现象时，才进入 RL。

## P27.5 更新：empty-admitted bug 与 session-level rescue 口径修复

新增记录：

- `refine-logs/P27_5_EMPTY_ADMITTED_AND_SESSION_LEVEL_RESCUE_REPAIR_20260831.md`

代码修复：

- `memory_sequence_agent.py` 修复 operator 后 `admitted = ∅` 时内部 state 不清空的问题；
- `SHRINK_VISIBLE` contract violation 现在 fail-closed，直接抛错，不再只记录后继续；
- trace 新增 `input_source_indices`、`dropped_source_indices`、`admitted_content_sha1_by_source` 等字段；
- compact parity exporter 新增 `final_admitted_empty` 与 per-operation empty-admitted count；
- matrix builder 新增 `proxy_utility_context`，旧 `utility` 只作为兼容 alias。

状态收紧：

- P26 reward comparison 仍为 `EXPLORATORY / USABLE`；
- P26 Group A/B/C 只能称为 `EXPLORATORY / SESSION-LEVEL VALID`；
- `50/50 admission rescue` 改称 `50/50 gold-session-level rescue`；
- `preservation loss = 0` 暂标 `NEEDS REPAIR / NOT YET TRUSTED`；
- P27 只能称为 `heuristic static admission baseline stress test`，不能称最终 strong static planner。

### H20 P27.5 当前阻塞

新增记录：

- `refine-logs/P27_5_H20_BLOCKED_GITHUB_TIMEOUT_20260831.md`

H20 尚未 clone，尚未开始 P27 generation。固定 commit `b6792a6c3d79ce6e4620a8ad87a58db5a431c2e6` 的 `git ls-remote` 被 GitHub HTTPS timeout 终止，exit code = 124。当前没有回退旧 commit，也没有产生 P27 结果污染。

后续必须等 GitHub HTTPS 恢复后继续固定 `b6792a6...`，不允许回退 `7c7cd45` 或 `674a306` 跑 P27。

### H20 P27.5 heuristic baselines reported result

新增记录：

- `refine-logs/P27_5_H20_HEURISTIC_STATIC_ADMISSION_REPORTED_RESULT_20260831.md`

H20 已在 fixed commit `b6792a6c3d79ce6e4620a8ad87a58db5a431c2e6` 完成 P27.5，五条路径均 80/80，无 `ERROR_API_CALL`。compact parity 400 行，missing qid / prompt / schema / admitted hash / retrieved hash 均为 0，所有 policy final admitted empty count = 0，contract violations = 0。因此 P27.5 empty-admitted bug 没有实际污染这轮 H20 P27 结果。

Group A/B/C = 3/50/27。

核心结果：

- `repack_graph_bridge`：ALL ΔR `+0.0283`，Group B `+0.1063`，Group C `-0.1169`；
- `repack_lexical_bm25`：ALL ΔR `+0.0220`，Group B `+0.0865`，Group C `-0.0914`；
- `repack_tfidf_jaccard`：ALL ΔR `+0.0203`，Group B `+0.0837`，Group C `-0.0914`；
- `repack_temporal_session`：ALL ΔR `+0.0135`，Group B `+0.0232`，Group C `0.0000`。

当前裁决：

> 没有一个 heuristic legal static admission policy 同时实现实质 gold-session-level Group-B rescue 且避免 Group-C reward harm。graph/lexical/tfidf 能 rescue B 但伤 C；temporal 不伤 C 但几乎不 rescue。rescue precision 仅 0.18 到 0.22，说明 gold-session-level admission 远不足以保证 task reward 改善。

下一步不是 RL，而是 static admission oracle gap 与 learned semantic static admission planner。

## P28 准备：static admission oracle gap

新增：

- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/analyze_p28_static_admission_oracle_gap.py`
- `refine-logs/H20_P28_STATIC_ADMISSION_ORACLE_GAP_COMMANDS_20260831.sh`

P28 不重跑 generation，只读 P27 wide matrix，计算：

- best fixed heuristic admission policy；
- per-group best fixed policy；
- raw per-example static oracle over `stop + four repack policies`；
- margin-aware static oracle，`margin_eps = 0.05 / 0.10`；
- `oracle - best fixed` gap；
- positive-margin sample count、oracle policy entropy、bootstrap CI；
- Group B rescue 与 Group C disruption frontier。

Static oracle 是 offline upper bound，不是 online deployable policy。P28 gate 看 margin-aware oracle，不看 raw max oracle。若 margin-aware oracle gap 仍大，只能说明 conditional static admission headroom，不支持 RL necessity。若 raw gap 大但 `margin_eps=0.10` 后消失，则不能进入 learned admission method gate。

### P28 margin-aware oracle repair

新增记录：

- `refine-logs/P28_MARGIN_AWARE_ORACLE_REPAIR_20260831.md`

P28 脚本已修正到 commit `394e6ae4d16f71b838fd73f45258451184da626c`。旧 raw per-example max oracle 只作为 exploratory reference，不作为 gate。修正版新增 `margin_eps=0.05/0.10`、`tie_eps=0.01`、cost fail-closed、positive-margin sample count、oracle policy entropy、bootstrap CI 与 Group A/B/C separate gap。

H20 已被要求使用修正版 P28；如果旧 P28 已经跑过，旧结果只标为 `RAW EXPLORATORY ORACLE ONLY`。

### H20 P28 margin-aware oracle reported result

新增记录：

- `refine-logs/P28_H20_MARGIN_AWARE_STATIC_ORACLE_REPORTED_RESULT_20260831.md`

H20 已在 fixed commit `394e6ae4d16f71b838fd73f45258451184da626c` 完成 P28 read-only analysis。输入是 P27.5 frozen wide matrix，不重跑 generation，不改 prompt/operator/protocol。

Reward 下，tie/cost-aware best fixed policy 是 `repack_tfidf_jaccard`，mean stop `0.34428`，mean best fixed `0.36453`。Margin-aware static oracle 在 ε=`0.05` 时 gap vs best fixed 为 `+0.09636`，bootstrap 95% CI `[0.04469, 0.15413]`；ε=`0.10` 时 gap 为 `+0.09391`，CI `[0.04600, 0.15218]`。

Proxy utility 下，best fixed 同样是 `repack_tfidf_jaccard`，mean stop `-0.14072`，mean best fixed `-0.00800`。Margin-aware oracle 在 ε=`0.05` 时 gap 为 `+0.08116`，CI `[0.03653, 0.13146]`；ε=`0.10` 时 gap 为 `+0.07486`，CI `[0.03327, 0.12806]`。

当前裁决：

> P28 通过的是 conditional static admission headroom gate：heuristic fixed policy 之后仍有非平凡 per-example selection headroom。它不证明 learned policy 可学，也不证明 RL necessity。下一步进入 P29 learned static admission gate，而不是继续刷 heuristic 或直接上 RL。

## P29 准备：learned static admission gate

新增：

- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/train_p29_selective_admission_gate.py`
- `experiments/UMA-BudgetedEvidenceMemory-20260830/tools/run_p29_selective_admission_gate_h20.sh`
- `refine-logs/P29_LEARNED_STATIC_ADMISSION_GATE_PLAN_20260831.md`
- `refine-logs/H20_P29_SELECTIVE_ADMISSION_GATE_COMMANDS_20260831.sh`

P29 是第一轮 learned static admission method gate，但仍不进入 RL。它只读 P27 frozen outputs，训练 nested-LOOCV selector 来比较：

- `stop`；
- tie/cost-aware best fixed heuristic；
- binary KEEP-vs-READMIT gates；
- multiclass static admission selector；
- offline static oracle upper bound。

默认 online feature 只允许 query text、first-retrieval C0/W0 trace statistics、W0 state text 与 budget/context metadata。`question_type`、gold evidence、answer、response text、judge label 与 operation outcome 均不得作为 online feature。脚本提供 `--include-question-type` 作为 privileged diagnostic control，但正式主表默认 false。

P29 的通过标准不是“看起来有一点提升”，而是 online-only learned selector 相对 best fixed heuristic 有正 gap，且不退化成 always-stop 或 always-one-readmit。即便 P29 通过，也只能支持 static admission learnability，不能跳到 RL necessity。

### P29 s05m 0.5B admission baselines and learned gate

新增记录：

- `refine-logs/P29_S05M_0P5B_ADMISSION_BASELINES_AND_GATE_RESULT_20260831.md`

本轮补做了真正的 s05m 小卡侧实验。s05m 上已有 manxin 的 `Qwen2.5-0.5B-Instruct` vLLM endpoint，运行在 3090 GPU1；本轮没有新启动大模型服务，也没有覆盖旧 UMA 实验目录，而是在：

`/data/manxin/agent_memory_rl/code/UMA-P29-S05M-AdmissionBaselines`

新建独立 clone，固定 commit：

`32b5a59775e6a2cb3748aa5994ee846de043f1ce`

完成 dev80/K20 的 P27-style 五策略 generation：

- `stop`：80/80；
- `repack_lexical_bm25`：80/80；
- `repack_tfidf_jaccard`：80/80；
- `repack_graph_bridge`：80/80；
- `repack_temporal_session`：80/80。

Parity rows = `400`，missing qid / prompt / trace schema 均为 `0`，所有 operation 的 final admitted empty count = `0`，contract violation = `0`。

P27-style 0.5B baseline 的结构结果：

- ALL 下 `repack_lexical_bm25` 是 reward/proxy utility 的 best fixed；reward ΔR `+0.02359`，proxy ΔU `+0.13654`；
- Group B 中 lexical/tfidf 都有正收益，lexical ΔR `+0.06449`，tfidf ΔR `+0.04684`；
- Group C 中 lexical/tfidf/graph 都有 reward harm，lexical ΔR `-0.05389`，tfidf ΔR `-0.07587`，graph ΔR `-0.05023`；
- temporal 不做 gold-session rescue，但在 Group C 不伤，ΔR `+0.04359`，W/T/L = `2/24/0`。

随后在同一 0.5B frozen matrix 上运行 P29 learned static admission gate。feature audit 通过：online-only、include_question_type=False、missing trace=0、missing first retrieve=0、W0 state text=80/80。

P29-s05m 结果：

- reward 下 best fixed 是 `repack_lexical_bm25`，mean best fixed `0.26993`，multiclass learned selector `0.26854`，learned-best fixed = `-0.00139`，CI `[-0.05734, 0.05248]`；
- proxy utility 下 best fixed 是 `repack_lexical_bm25`，mean best fixed `-0.10280`，multiclass learned selector `-0.11022`，learned-best fixed = `-0.00741`，CI `[-0.02987, 0.01100]`。

当前裁决：

> P29-s05m 0.5B 没有通过 “learned static admission clearly beats best fixed” 的门。它可以明显赢 STOP，尤其在 proxy utility 下，但没有稳定赢 best fixed lexical。说明 P28/P27 的 oracle headroom 还没有被当前 stats+hashed query/W0 online representation 捕捉到。下一步应等 H20 7B P29 adjudication，并优先补 semantic state representation / answer-bearing audit，而不是直接上 RL。

### P29 H20 7B learned static admission gate

新增记录：

- `refine-logs/P29_H20_7B_LEARNED_STATIC_ADMISSION_GATE_REPORTED_RESULT_20260831.md`
- `refine-logs/P29_CROSS_SCALE_STATIC_ADMISSION_SYNTHESIS_20260831.md`

H20 P29 修复版已完成，fixed commit：

`32b5a59775e6a2cb3748aa5994ee846de043f1ce`

H20 run root：

`/data/cw/memagent_work/code/UMA-P29-SelectiveAdmissionGate-H20/runs/p29_selective_admission_gate_from_p27_dev80_k20_fixed_32b5a597`

H20 feature audit：n=`80`，missing trace=`0`，missing first retrieve=`0`，W0 state text=`80`，include_question_type=`false`，nested LOOCV，feature count=`283`。输入只读 P27.5 frozen artifacts，没有重跑 generation，没有改 prompt/operator/protocol，没有打开 confirm324。

Reward 下 best fixed 是 `repack_tfidf_jaccard`。Multiclass static selector mean learned `0.35315`，learned - stop `+0.00887`，learned - best fixed `-0.01138`，CI `[-0.05819, 0.02811]`。

Proxy utility 下 best fixed 也是 `repack_tfidf_jaccard`。Multiclass static selector mean learned `-0.00998`，learned - stop `+0.13074`，learned - best fixed `-0.00199`，CI `[-0.00596, 0.00000]`，行为几乎退化为选择 tfidf `78/80`。

P29 cross-scale 裁决：

> 0.5B/s05m 与 7B/H20 都没有让当前 `stats_text` online selector 稳定超过 best fixed heuristic。P28 oracle headroom 仍然存在，但 P29 显示当前 representation/model family 没有把这个 headroom 转成可部署 policy。下一步不是 RL，而是 semantic-state selector、answer-bearing audit 与 official judge calibration。

## P29.5：source-index 删除与 W0 state_text 合约验证

新增记录：

- `refine-logs/P29_5_STAGE_CONSOLIDATED_DELIVERY_20260901.md`
- `refine-logs/P29_5_SOURCE_INDEX_AND_STATE_TEXT_CONTRACT_REPAIR_20260831.md`
- `refine-logs/P29_5_S05M_NO_SOURCE_INDEX_RESULT_20260831.md`

P29 代码审查后确认：原始 online feature 默认包含 `retrieved_source_index_*` 与 `admitted_source_index_*` 统计。这不是 gold / answer / judge leakage，但属于 dataset/session ordering artifact，不能作为正式 online-state method feature。P29.5 已修复为：

- 默认删除 raw source-index statistics；
- 仅在显式打开 `--include-source-index-features` 时作为 diagnostic control 加入，且字段前缀改为 `diagnostic_source_index_*`；
- report / feature audit 明确记录 `include_source_index_features`；
- 新增 `tools/verify_trace_state_text_contract.py`，在 runner 中训练前验证 STOP trace 的 W0 `state_text` 是否与 producer 记录的 context hash、final context hash、prompt hash 对齐。

本次修复固定 commit：

`41e14dd5457eecd41711ff43124b2ab033a2b5bf`

该 commit 已 push 到：

`Gufrannn/MemAgent-Research@research/uma-p25-5-trace-semantics-h20-20260831`

### P29.5 s05m 0.5B no-source-index result

s05m 已完成 P29.5 formal no-source-index gate。输入只读已有 P27-style 0.5B frozen artifacts；没有重跑 generation，没有改 prompt/operator/split/metric，没有打开 confirm324。

运行位置：

`/data/manxin/agent_memory_rl/runs/p29_5_selective_admission_gate_s05m_0p5b_no_source_index_from_p27_dev80_k20`

W0 contract verifier：

- status：`VALID_TRACE_STATE_TEXT_CONTRACT`
- checked：`80/80`
- violation rows：`0`

Reward：

- best fixed：`repack_lexical_bm25`
- multiclass learned selector mean：`0.271259`
- learned - stop：`+0.024928`
- learned - best fixed：`+0.001333`
- 95% CI vs best fixed：`[-0.065159, 0.066487]`

Proxy utility：

- best fixed：`repack_lexical_bm25`
- multiclass learned selector mean：`-0.112210`
- learned - stop：`+0.127133`
- learned - best fixed：`-0.009407`
- 95% CI vs best fixed：`[-0.032395, 0.010066]`

当前裁决：

> P29.5-s05m 是 `VALID EXPLORATORY HASHED-LEXICAL ONLINE-STATE BASELINE WITHOUT SOURCE-INDEX`。删除 source-index 后，结论没有实质改变：当前 cheap stats+hashed W0 representation 可以赢 STOP，但没有稳定赢 best fixed heuristic。它是有效的弱基线负结果，不是 semantic admission 结论，也不支持 RL necessity。

H20 P29.5 已单独派发到 H20 task：固定同一 commit `41e14dd5457eecd41711ff43124b2ab033a2b5bf`，只读 H20 P27.5 frozen artifacts，运行 no-source-index gate + W0 contract verifier。等待 H20 返回后再登记 7B 裁决。

### P29.5 H20 7B no-source-index result

新增记录：

- `refine-logs/P29_5_H20_NO_SOURCE_INDEX_RESULT_20260831.md`
- `refine-logs/P29_5_CROSS_SCALE_NO_SOURCE_INDEX_SYNTHESIS_20260901.md`

H20 已完成 P29.5 formal no-source-index gate。固定 commit：

`41e14dd5457eecd41711ff43124b2ab033a2b5bf`

H20 run root：

`/data/cw/memagent_work/code/UMA-P29_5-SelectiveAdmissionGate-H20/runs/p29_5_selective_admission_gate_no_source_index_from_p27_dev80_k20_41e14dd`

Contract verifier 通过：

- status：`VALID_TRACE_STATE_TEXT_CONTRACT`
- violation rows：`0`

Feature audit：

- `include_question_type=false`
- `include_source_index_features=false`
- `feature_count=273`
- `n=80`
- `missing_trace=0`
- `missing_first_retrieve=0`
- W0 state text rows：`80`

Reward 下 best fixed 是 `repack_tfidf_jaccard`。Multiclass learned selector mean `0.340438`，learned - stop `-0.003838`，learned - best fixed `-0.024093`，CI `[-0.079499, 0.021808]`。

Proxy utility 下 best fixed 同样是 `repack_tfidf_jaccard`。Multiclass learned selector mean `-0.009985`，learned - stop `+0.130740`，learned - best fixed `-0.001986`，CI `[-0.005958, 0]`，几乎退化为选择 tfidf `78/80`。

P29.5 cross-scale 裁决：

> 删除 raw source-index statistics 后，0.5B/s05m 与 7B/H20 都没有让当前 cheap `stats_text + hashed W0` online selector 稳定超过 best fixed heuristic。P29.5 是 `VALID EXPLORATORY HASHED-LEXICAL ONLINE-STATE BASELINE WITHOUT SOURCE-INDEX` 的负结果。它说明当前 representation 不足以实现 P28 offline oracle headroom；它不证明 semantic admission 不可学，也不支持 RL necessity。

下一步应进入 P30 semantic-state admission gate，而不是继续在 P29.5 的 hashed feature family 上调参，也不是直接进入 RL。

### P30 Semantic Selective Admission v1

新增记录：

- `refine-logs/P30_SSA_V1_METHOD_AND_S05M_RESULT_20260901.md`
- `refine-logs/P30_H20_STATUS_AND_HANDOFF_20260901.md`

P30 固定 commit：

`17bd632fb3480851a96fd35f34a31adff259f69d`

本轮严格遵守“不要偷偷改东西”的要求。P30 没有改 prompt、operator、split、metric，没有重跑 generation，没有打开 confirm324，没有下载大模型或数据。代码改动只围绕 Semantic Selective Admission v1 与报告字段修正。

核心代码：

- `tools/train_p30_semantic_selective_admission.py`
- `tools/run_p30_semantic_selective_admission_from_p27.sh`
- `tools/verify_trace_state_text_contract.py`

s05m 已完成 P30 0.5B/Qwen-mean exploratory run：

`/data/manxin/agent_memory_rl/runs/p30_ssa_v1_s05m_0p5b_qwen_mean_dev80_k20_17bd632`

W0 contract：

- status：`VALID_TRACE_STATE_TEXT_CONTRACT`
- checked：`80/80`
- violation rows：`0`

Reward 下，SSA v1 mean `0.226151`，比 STOP 低 `-0.020180`，比 full-dev descriptive best fixed `repack_lexical_bm25` 低 `-0.043774`。Proxy utility context 下，SSA v1 mean `-0.163936`，比 STOP 高 `+0.075408`，但比 full-dev descriptive best fixed `repack_lexical_bm25` 低 `-0.061133`。

当前裁决：

> P30 s05m 是 `VALID EXPLORATORY NEGATIVE FOR SSA_V1_QWEN_MEAN_ON_DEV80`。它说明当前 Qwen2.5-0.5B mean-pooled semantic features + tiny static value head 没有把 P28 oracle headroom 转化为超过 strong fixed heuristic 的可学习 policy。它不证明 semantic admission 不可学，也不支持 RL necessity。

H20 P30 已完成，新增记录：

- `refine-logs/P30_H20_7B_SSA_V1_RESULT_20260901.md`
- `refine-logs/P30_CROSS_SCALE_SSA_V1_SYNTHESIS_20260901.md`

H20 固定同一 commit `17bd632fb3480851a96fd35f34a31adff259f69d`，只读 frozen P27.5 artifacts，没有 generation，没有改 prompt/operator/split/metric/protocol，没有打开 confirm324，没有 install/download/fallback。encoder 为 `/data/cw/memagent_work/models/Qwen2.5-7B-Instruct`，backend 为 `transformers_mean`。

H20 W0 contract：

- status：`VALID_TRACE_STATE_TEXT_CONTRACT`
- checked：`80/80`
- violation rows：`0`

H20 forbidden feature audit：

- include_question_type：`0`
- include_source_index_features：`0`
- uses_gold：`0`
- uses_answer：`0`
- uses_judge：`0`
- uses_outcome：`0`
- source_indices_used_as：`pointers_only_not_features`

H20 reward 下，SSA v1 mean `0.347463`，比 STOP 高 `+0.003187`，但比 full best fixed `TFIDF` 低 `-0.017069`，CI `[-0.062859, 0.023799]`。Proxy utility context 下，SSA v1 mean `0.004753`，比 full best fixed 高 `+0.012752`，但 CI `[-0.004111, 0.039631]` 跨 0，不能覆盖 reward gate 失败。

Cross-scale 裁决：

> P30 是当前 SSA v1 的 cross-scale exploratory negative。0.5B/s05m 与 7B/H20 都没有在 reward gate 上超过 strong fixed heuristic。当前 Qwen mean-pooled semantic features + tiny Default/Override head 没有实现 P28 oracle headroom。下一步应做 candidate-level answer-bearing / admission opportunity audit，而不是直接进入 RL。

最终可复制整理包：

`/Users/xinman/Documents/Codex/2026-08-30/agent-memory-code-research/outputs/P30_SSA_V1_S05M_H20_COMPLETE_20260901`

### P30.1 Baseline Consistency Audit

新增记录：

- `refine-logs/P30_1_BASELINE_CONSISTENCY_AUDIT_20260901.md`

新增代码：

- `tools/audit_p30_baseline_consistency.py`

P30.1 起因是 P30 method table 中 `Full_Best_Fixed` 命名与表内 fixed policy raw mean 最高项不一致。例如 H20 reward 中 Graph mean `0.372584` 高于 TFIDF mean `0.364531`，但旧表把 TFIDF 写成 full best fixed。

审计结论：

> 当前可见表支持 `reporting/name semantics bug`，不是 matrix mixing。P30 代码使用 `tie_eps=0.01` 的 tie/cost-aware selection，因此一个 raw mean 略低但更便宜的 policy 会被选为 tie/cost-aware fixed default。旧名 `Full_Best_Fixed` 不严谨，未来应分成 `Raw_Best_Fixed` 与 `Full_TieCost_Fixed`。

四个表的状态：

- s05m reward：`NAMING_INCONSISTENCY_TIECOST_WITHIN_TIE_EPS`
- s05m proxy：`CONSISTENT_RAW_BEST`
- H20 reward：`NAMING_INCONSISTENCY_TIECOST_WITHIN_TIE_EPS`
- H20 proxy：`NAMING_INCONSISTENCY_TIECOST_WITHIN_TIE_EPS`

P30 qualitative negative 不变。相对 raw best fixed，reward 下 s05m SSA v1 差 `-0.049624`，H20 SSA v1 差 `-0.025121`。下一步 P31 的 oracle exception audit 必须明确区分 raw-best-fixed exception、tie/cost-default exception、fold-local default exception。

### P31 Oracle Exception Audit

新增代码：

- `tools/audit_p31_oracle_exceptions.py`

P31 修复 commit：

`954b956a9ec1813c539483bd7d40544577edd26d`

P31 是只读 offline audit，不训练、不 generation、不重评分、不改 prompt/operator/split/metric/protocol，不打开 confirm324。它使用 `answer_session_ids` 作为 offline audit label，检查 oracle exception 到底来自 answer-bearing admission、candidate failure、还是 organization/interface/surrogate。

s05m 0.5B 已完成，新增记录：

- `refine-logs/P31_S05M_ORACLE_EXCEPTION_AUDIT_20260901.md`

s05m reward 下，raw fold best 与 tie/cost fold default 都有 `24/80` 个 >0.1 exception。C0 answer-bearing session present rate 为 `0.775`，STOP W0 answer-bearing session present rate 为 `0.4125`。但 exception 主类不是“winning policy 成功 rescue answer-bearing session”，而是：

- `no_candidate_opportunity_answer_not_in_C0`
- `organization_or_reader_interface_or_surrogate`
- 以及少量 `answer_bearing_admission_deficit_not_rescued`

当前 s05m 裁决：

> P31-s05m 不支持“P28/P30 oracle headroom 主要来自简单 answer-bearing session rescue”。它提示下一步要区分 retrieval/candidate construction failure、working-memory organization/interface，以及 surrogate F1 wording noise。

H20 P31 已派发到专用 H20 task，固定同一 commit `954b956a9ec1813c539483bd7d40544577edd26d`，等待 7B 结果后再做 cross-scale P31 synthesis。

H20 7B P31 已完成，新增记录：

- `refine-logs/P31_H20_ORACLE_EXCEPTION_AUDIT_20260901.md`
- `refine-logs/P31_CROSS_SCALE_ORACLE_EXCEPTION_SYNTHESIS_20260901.md`

H20 7B reward 下：

- raw fold best：15/80 exception，C0 answer present `0.9875`，W0 answer present `0.9125`。
- tie/cost fold default：15/80 exception，C0 answer present `0.9875`，W0 answer present `0.9125`。
- dominant exception class：`organization_or_reader_interface_or_surrogate`，raw 下 13/15，tie/cost 下 13/15。

P31 cross-scale 裁决：

> 7B 下 substantive oracle exceptions 主要不是 session-level answer-bearing admission rescue。answer-bearing session 通常已经在 C0 和 W0 中。下一步不应继续救 SSA v1，也不应直接把 candidate-level answer-bearing admission scorer 当主方法；更合理的是先做 P32 Organization-vs-Surrogate Audit，判断 oracle win 到底来自 evidence organization/reader interface，还是 surrogate F1 wording artifact。

P31 可复制整理包：

`/Users/xinman/Documents/Codex/2026-08-30/agent-memory-code-research/outputs/P31_ORACLE_EXCEPTION_AUDIT_COMPLETE_20260901`

### P31.5 Gold-Session Completeness Taxonomy Repair

新增代码修复：

- `tools/audit_p31_oracle_exceptions.py`

修复 commit：

`545bf00e31756890beeab2c2013ac867cc0b1004`

P31.5 起因：P31 旧代码中的 `has_answer_session` 实际是 `ANY gold session`，不是 `ALL gold sessions`。因此旧 P31 中 “organization/interface/surrogate” 类 exception 可能被高估。P31.5 不改 prompt、operator、split、metric、generation protocol，只把 offline audit taxonomy 修成 `any / all / recall` 三口径。

新增记录：

- `refine-logs/P31_5_S05M_GOLD_SESSION_COMPLETENESS_AUDIT_20260901.md`
- `refine-logs/P31_5_CROSS_SCALE_SYNTHESIS_PENDING_H20_20260901.md`

s05m repaired taxonomy 已完成：24/80 substantive exceptions 中，`retrieval_or_candidate_missing_gold_sessions=17`，`admission_incomplete_not_rescued=5`，`gold_session_complete_organization_interface_or_surrogate_candidate=2`。这说明 0.5B regime 下旧 P31 的 organization/interface 解释需要大幅降级，主要仍是 retrieval/candidate/admission completeness 问题。

H20 repaired taxonomy 已派发到专用 H20 task，固定同一 commit `545bf00e31756890beeab2c2013ac867cc0b1004`。在 H20 结果返回前，不能再引用旧 P31 的 “7B 13/15 organization/interface/surrogate” 作为严格结论。

H20 repaired taxonomy 主 summary 已返回，新增记录：

- `refine-logs/P31_5_H20_GOLD_SESSION_COMPLETENESS_AUDIT_20260901.md`
- `refine-logs/P31_5_CROSS_SCALE_SYNTHESIS_20260901.md`

H20/7B repaired taxonomy 下，15/80 substantive exceptions 中，raw fold best 为 `admission_incomplete_not_rescued=8`、`gold_session_complete_organization_interface_or_surrogate_candidate=6`、`retrieval_or_candidate_missing_gold_sessions=1`；tie/cost fold default 为 `admission_incomplete_not_rescued=7`、`gold_session_complete_organization_interface_or_surrogate_candidate=7`、`retrieval_or_candidate_missing_gold_sessions=1`。因此旧 P31 的 “13/15 organization/interface/surrogate” 结论被 P31.5 修正为：7B oracle exceptions 同时包含 admission incomplete 与 gold-session-complete organization/interface/surrogate candidates。下一步 P32 应做 turn/span/content-level failure-mode identification。

H20 P31.5 输出 SHA256：by-type `594370a8517b8bd3e5fdc75c0f92b8a87cbc4b48cdf22b82468620798a4c5a85`，per-qid `769d22daf1a94b3eebffb9e6be63fa2dcba9c3b4c47d9e87d39c72077f3cadd5`，report `28bba7b75a60459031e13633b884cc28068eeb3115478e172c87022235ef49da`，summary `5540b438997c3572f77e7e52c55c89eb37addeba65497b455dc938022c0b2981`。

### P31.6 Final Visible Working Memory Repair

新增代码修复：

- `tools/audit_p31_oracle_exceptions.py`

修复 commit：

`72a3ac0027a5ea96f16368569a096b91acbc5da8`

P31.6 起因：P31.5 虽然修复了 ANY/ALL/Recall，但 base/winner W completeness 对所有 policy 使用了 `first RETRIEVE.admitted_source_indices`。trace 抽样显示，`repack_*` 的真实 final reader-visible W 在 `REPACK_CANDIDATES.admitted_source_indices`。因此 P31.5 结果先降级，P31.6 改为 C0 使用 first RETRIEVE retrieved，policy W 使用 final visible op record。

s05m / 0.5B P31.6 已完成，新增记录：

- `refine-logs/P31_6_FINAL_VISIBLE_W_REPAIR_AND_S05M_RESULT_20260901.md`

s05m P31.6 raw fold best：24/80 exceptions，`retrieval_or_candidate_missing_gold_sessions=17`，`admission_incomplete_complete_rescue=3`，`admission_incomplete_not_rescued=1`，`gold_session_complete_organization_interface_or_surrogate_candidate=3`。tie/cost fold default：24/80 exceptions，`retrieval_or_candidate_missing_gold_sessions=17`，`admission_incomplete_complete_rescue=3`，`gold_session_complete_organization_interface_or_surrogate_candidate=4`。

H20 P31.6 已完成，P31.6 正式覆盖 P31.5。H20/7B final-visible-W 口径下，raw fold best 与 tie/cost fold default 均为 15/80 exceptions，其中 14/15 是 `gold_session_complete_organization_interface_or_surrogate_candidate`，1/15 是 `retrieval_or_candidate_missing_gold_sessions`。P31.5 中的 admission-incomplete-not-rescued 分类在 final-visible-W 口径下消失。

因此 P32 进入 GO，但范围严格限制为 H20/7B 的 15 个 substantive exceptions。P32 只做 turn/span/content-level failure-mode identification，不扩大样本、不训练、不 generation、不重评分、不进入 RL。

### P32 Exception Content/Path Audit

新增脚本：

- `tools/audit_p32_exception_content_paths.py`

Commit：

`0f3e2070ae3a93db6f2c940e19b46225e5684058`

P32 是 P31.6 后的最后一轮 failure-mode identification。它只读取 frozen P27.5 artifacts、P31.6 per-qid、raw LongMemEval 和 dev80 manifest；不训练、不 generation、不 rescoring、不调用 official judge，只生成 official judge queue。

s05m / 0.5B smoke 已完成，新增记录：

- `refine-logs/P32_EXCEPTION_CONTENT_PATH_AUDIT_STATUS_20260901.md`

s05m P32 仅用于验证脚本能运行，不作为论文机制主依据。H20 / 7B P32 已派发并开始执行，等待 15 个 substantive exceptions 的 content/path audit 结果。

H20 / 7B P32 主 summary 已返回。raw 15 exceptions 中：retrieval missing 1，winner-content-surface 1，B-or-C content-present 9，B-or-C answer-string-absent 3，C surrogate/format 1。tie/cost 15 exceptions 中：retrieval missing 1，winner-content-surface 2，B-or-C content-present 9，B-or-C answer-string-absent 2，C surrogate/format 1。当前无法在 Route B working-memory organization 与 Route C surrogate artifact 之间做最终选择。P32 仍是 exploratory surrogate F1 + string probe，不是 official judge，也不是 turn/span annotation。

H20 P32 SHA256：per-exception `41647d800263bcfac2b8308b30037baa7a1d5c760324a63732719d608824480c`，summary `8f83368feec07e6e3d1fd8b1938e120f97206b7b78b1a3316a1d0a323d2f2172`，by-type `b6afad9efd43d582e4aa955d11573de7a0edf167b499852a190c59a5c48aac0a`，judge queue `523734d8e17a80a99186520a4a3302b0ccf7fbdfda61979f1397e03d2bf7fb3e`，report `d52643b0fc208783d4e1e5c6ded73f9f560405d8e2b6366422a2f6d889b13486`。

P32 H20 final extraction 已完成。official judge queue 只生成未调用，header present，60 data rows / 61 file lines；base counts raw 30、tiecost 30；role counts base 30、winner 30；policy counts tfidf 14、graph 18、stop 19、lexical 5、temporal 4。

P32 final route 裁决：当前不能在 Route B working-memory organization 与 Route C surrogate artifact 之间做最终选择。多数 H20 exceptions 仍是 B-or-C unresolved。下一步最小信息增益是对 P32 的 60-row judge queue 做 official judge 或人工 turn/span/content 判读，而不是进入 Method v2 或 RL。

### P33 UMA Official Baseline Reset

P33 开始，主线从旧 LongMemEval admission / packing 诊断线收回，切换为：

`Official UMA baseline → UMA RL anatomy → 找真实 failure → 改 UMA → matched training comparison`

新增记录：

- `refine-logs/P33_UMA_BASELINE_RESET_AND_RL_ANATOMY_PLAN_20260901.md`
- `handoffs/P33_H20_UMA_OFFICIAL_BASELINE_PRECHECK_20260901.md`

本轮只读审计官方 UMA 本地副本：

- repo：`code_research/repos/unified-memory-agent`
- commit：`768f9620231bae11264771f59e43a4839506cf94`
- official upstream：`https://github.com/ictnlp/unified-memory-agent`
- UMA-4B checkpoint page：`https://huggingface.co/ICTNLP/UMA-4B`

初步代码锚点显示：官方 UMA 的 GRPO advantage 分组使用 `uid_is_final`，因此 memory phase 与 QA phase 被分开归一化；但 memory phase 内尚未看到按 chunk/session/tool op/target/content 的细粒度下游归因。`agent_loop.py` 会把 final QA outcome 平均传回 non-final memory conversations，再叠加 tool execution reward；`core_algos.py` 的 GRPO 将 scalar score 乘到整段 generated-token mask。该发现目前状态是 `HYPOTHESIS-CANDIDATE / NEEDS INSTRUMENTED RUN`，不能直接写成最终机制结论。

P33 pre-registered hypotheses：

1. H1：UMA memory credit 不能区分同一 trajectory 内不同 memory decisions。
2. H2：flat token-level credit 混合 operation / target / content 三层错误。
3. H3：global memory-group normalization 与异质 CRUD operation 分布不匹配。

下一步 P33.1：在 3090/s05m 做 UMA Advantage Anatomy smoke，只加默认关闭的 logging instrumentation，不改 prompt/operator/split/metric/generation protocol；H20 先只做官方 UMA readiness precheck，不训练、不 generation、不下载、不安装。

### P33.1 UMA Credit-Flow Sanity

新增脚本：

- `experiments/UMA-Official-Anatomy-20260901/tools/audit_p33_1_uma_credit_flow_sanity.py`

新增记录：

- `refine-logs/P33_1_UMA_CREDIT_FLOW_SANITY_20260901.md`
- `results/P33_1_UMA_CREDIT_SANITY_20260901/p33_1_uma_credit_flow_sanity.json`
- `results/P33_1_UMA_CREDIT_SANITY_20260901/p33_1_uma_credit_flow_sanity.md`

P33.1 是 CPU-only / read-only 的官方 UMA credit-flow anatomy，不训练、不 generation、不下载、不改官方代码，也不改 prompt/operator/split/metric/generation protocol。

审计官方 UMA commit：

`768f9620231bae11264771f59e43a4839506cf94`

本轮结论：

- `agent_loop.py:523-645` 显示 non-final memory conversation 的 reward 为 shared downstream QA outcome 加 local tool reward；同一 trajectory 内 memory chunks 共享 QA component。
- `hotpotqa.py:149-157` 与 `mem_tool.py` 显示 tool reward 是执行成功信号，不是 semantic memory quality signal。
- `ray_trainer.py:229-240` 显示 GRPO group 使用 `uid_is_final`，memory phase 与 QA phase 分开归一化，但 memory phase 内仍不是 chunk/session/tool-target/content 级 credit。
- `core_algos.py:298-325` 显示 response scalar advantage 被广播到 generated-token mask。
- `agent_loop.py:871-904` 的 `ray.wait` completion-order mapping pattern 在合成测试中可复现错配风险；当前只标为工程风险，尚不能宣称官方训练已污染。

状态标签：

`CODE AUDITED / SYNTHETIC UNIT SANITY / NO TRAINING / RL NECESSITY NOT ESTABLISHED`

下一步最小信息增益实验是 P33.2：真实 rollout credit dump，只加默认关闭 logging instrumentation，先检查真实训练张量里的 reward/advantage 对齐，再决定是否允许设计 credit repair。

### P33.2 UMA Credit Trace Instrumentation Prep

新增脚本：

- `experiments/UMA-Official-Anatomy-20260901/tools/prepare_p33_2_uma_credit_instrumentation.py`
- `experiments/UMA-Official-Anatomy-20260901/tools/summarize_p33_2_credit_trace.py`

新增记录：

- `refine-logs/P33_2_UMA_CREDIT_TRACE_INSTRUMENTATION_PREP_20260901.md`
- `results/P33_2_UMA_CREDIT_TRACE_PREP_20260901.json`
- `handoffs/P33_2_H20_UMA_CREDIT_TRACE_REAL_ROLLOUT_HANDOFF_20260901.md`

本地正式 instrumented worktree：

- `experiments/UMA-Official-Anatomy-20260901/instrumented_worktree_p33_2_v2`

本轮严格边界：不改官方 UMA baseline 副本，不训练、不 generation、不 rescore、不 official judge、不下载模型/数据，也不改 prompt/operator/split/metric/generation protocol/reward formula/trainer objective。

P33.2 instrumentation 默认关闭，只有显式设置：

`UMA_CREDIT_TRACE=1` 与 `UMA_CREDIT_TRACE_PATH=/path/to/uma_credit_trace.jsonl`

才写 JSONL。关闭时 trace 工具为 no-op。

本地 smoke：AST parse、`git diff --check`、default-off trace、enabled trace write 均通过。正式 instrumented file SHA256：

- `agent_loop.py`：`3ab729f15fa9efb97a9906510f394e3bd25d3dcb058fb003b0d869dbe4d35487`
- `ray_trainer.py`：`b4fb96fab8532c8c802fa23d1cee35ba2f95efa051e8bd13aaa9d600631f42c4`
- `uma_credit_trace.py`：`87f0a41145535004ba7e8dff38b8941728a9fcbea8f374cef20d49fd6dbe8a1a`

s05m smoke：已从远端 dirty UMA experiment 的 Git 对象创建 detached clean official worktree `/data/manxin/agent_memory_rl/p33_uma_official_anatomy/uma_official_768f962`，HEAD 为 `768f9620231bae11264771f59e43a4839506cf94`；再生成 `/data/manxin/agent_memory_rl/p33_uma_official_anatomy/uma_instrumented_p33_2_v2`。远端 Python 为 `/data/manxin/miniconda3/envs/newMamba310/bin/python`，版本 3.10.13。远端 AST、diff check、default-off 和 enabled trace write 均通过。远端 modified file SHA 与本地一致。

当前未在 s05m 启动真实 rollout。原因：官方 `external/verl/run_1node.sh` 是 8×GPU/Qwen3-4B 训练入口，而 s05m precheck 时只有 GPU0 空闲、GPU1 占用。本轮 s05m 状态只能标为 `INSTRUMENTATION SMOKE PASS / REAL ROLLOUT NOT RUN`。

P33.2-H20 handoff 已准备，但 H20 只能从 GitHub 拉取。P33.2 代码尚未在本轮 push 到 GitHub exact commit，因此 H20 real rollout anatomy 当前等待用户授权代码同步。H20 后续只应做 official 4B real rollout anatomy dump，不应进入正式训练或 method 改造。

### P33.2-v3 UMA Credit Trace Identity Repair

新增记录：

- `refine-logs/P33_2_V3_UMA_CREDIT_TRACE_REPAIR_20260901.md`
- `results/P33_2_V3_UMA_CREDIT_TRACE_REPAIR_20260901/p33_2_v3_status.json`
- `h20_handoffs/P33_2_V3_H20_REAL_ROLLOUT_HANDOFF_20260901.md`

本轮是 P33.2 的 instrumentation repair，不是新实验结论。根据代码审查意见，v3 只修以下内容：

- 不再依赖可能缺失的 `trajectory_id` 做 trajectory-level 分组；
- 从 official `get_trajectory_info(step, sample_index, rollout_n, validate)` 派生 `p33_2_trajectory_key`；
- 在 `tool_mem_agent_loop.py` 创建 `AgentLoopOutput.extra_fields` 时写入真实 `p33_2_conversation_index`、`p33_2_memory_step_index`、`p33_2_final_query_index`；
- summarizer 禁止根据最终 row order 重建 chunk index，缺 identity/index 字段直接 fail closed；
- trace logging 改成默认关闭、per-host/pid JSONL 文件，不使用共享文件锁，不逐条 `fsync`；
- summarizer 增加 `reward_rank_within_grpo_group` 与 `trajectory_mean_memory_advantage`。

明确未改 prompt、operator、split、metric、generation protocol、reward formula、trainer objective 或 optimizer 行为。

本地 v3 worktree：

- `experiments/UMA-Official-Anatomy-20260901/instrumented_worktree_p33_2_v3`

本地 smoke 结果：

- Python compile：PASS；
- `git diff --check`：PASS；
- default-off trace：PASS；
- enabled per-process trace：PASS；
- summarizer synthetic valid trace：PASS；
- summarizer missing `trajectory_key` fail-closed：PASS；
- reward-rank tie-break：PASS。

本轮 s05m v3 未运行：向 s05m 同步 v3 脚本时被本地安全审查拒绝，因此旧的 s05m v2 smoke 不能当作 v3 证据。若继续 s05m，需要用户明确授权将 P33.2-v3 两个脚本传到 s05m 并运行只读 smoke。

状态标签：

`CODE REPAIRED / LOCAL SMOKE PASSED / S05M V3 NOT RUN / H20 REAL ROLLOUT NOT RUN / NO TRAINING / NO GENERATION`

### P33.2-v3.1 UMA Credit Trace Primary Identity Repair

新增记录：

- `refine-logs/P33_2_V3_1_UMA_CREDIT_TRACE_PRIMARY_ID_REPAIR_20260901.md`
- `results/P33_2_V3_1_UMA_CREDIT_TRACE_IDENTITY_REPAIR_20260901/p33_2_v3_1_status.json`
- `h20_handoffs/P33_2_V3_1_H20_REAL_ROLLOUT_HANDOFF_20260901.md`

P33.2-v3.1 是一个 summarizer-only tiny repair。根据代码审查，官方 `ToolMemoryAgentLoop` 已经生成 per-agent-run UUID `trajectory_id`，它应作为 primary trajectory identity；v3 派生的 `(step, sample_index, rollout_n, validate)` 只保留为 provenance / collision cross-check。

本轮只改：

- `experiments/UMA-Official-Anatomy-20260901/tools/summarize_p33_2_credit_trace.py`

本轮未改：

- instrumentation diff；
- `prepare_p33_2_uma_credit_instrumentation.py`；
- prompt / operator / split / metric / generation protocol / reward formula / trainer objective / optimizer behavior。

修复内容：

- `trajectory_id` 变为 REQUIRED primary grouping key；
- `trajectory_key` 仍 REQUIRED，但只做 provenance；
- 检查 `trajectory_id -> trajectory_key` 必须唯一，违反则 fail closed；
- 允许并报告 `trajectory_key -> multiple trajectory_id`；
- reward rank 改为 tie-aware dense rank；
- 新增 `reward_dense_rank_within_grpo_group`、`reward_tie_size_within_grpo_group`、`n_unique_rewards_in_grpo_group`。

本地 smoke：

- AST parse PASS；
- valid synthetic trace PASS；
- UUID primary grouping PASS；
- missing `trajectory_id` fail-closed PASS；
- `trajectory_id -> multiple trajectory_key` fail-closed PASS；
- tie-aware dense rank PASS。

状态标签：

`CODE REPAIRED / LOCAL SMOKE PASSED / INSTRUMENTATION DIFF UNCHANGED / H20 NOT RUN`

H20 必须使用包含 v3.1 summarizer 的 exact commit；旧 v3 commit 不再作为 H20 real rollout 入口。

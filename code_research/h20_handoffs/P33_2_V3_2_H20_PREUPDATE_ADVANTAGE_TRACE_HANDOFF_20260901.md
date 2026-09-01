# P33.2-v3.2 H20 官方 UMA pre-update credit trace 交接

状态：主线修复包，供 H20 对话执行。v3.1 因 H20 缺官方 UMA source、缺明确 4B model/data/env/runner 而 BLOCKED；v3.2 不允许用旧派生 repo、7B/14B 或手工 patch 代替。

## 目标

在官方 UMA source exact commit 上做一次最小真实 rollout anatomy：

- 读取官方 UMA `ToolMemoryAgentLoop` 真实产生的 `trajectory_id`。
- 读取真实 reward postprocess。
- 读取真实 GRPO group / raw reward / advantage。
- 在 advantage trace 写完后立即退出，禁止进入 critic/actor optimizer step。

这不是训练，不是复现完整 paper，不生成 checkpoint，不改 prompt/operator/split/metric/reward/generation protocol。

## 必须固定

- GitHub repo：`Gufrannn/MemAgent-Research`
- Branch：`research/uma-p33-2-credit-trace-v3-20260901`
- 使用主线提供的最新 v3.2 commit（不要用 v3.1 的 `2b3b3e0...`）。
- 官方 UMA source repo：`https://github.com/ictnlp/unified-memory-agent.git`
- 官方 UMA exact commit：`768f9620231bae11264771f59e43a4839506cf94`
- H20 工作根目录：`/data/cw/memagent_work`

## H20 执行原则

1. 不使用 H20 上已有的 `MemAgent`、`MemAgent-Research`、`MemAgent-control`、`MemAgent-MIC` 等派生 repo 作为 official source，除非它们能精确证明是官方 `768f962...` 且 layout 完整；否则由 runner 从官方 HTTPS repo 拉取。
2. 不在 dirty repo 上 patch。
3. 不安装依赖。
4. 不下载模型或数据。
5. 不替换模型：没有本地官方 4B 路径就 BLOCK。
6. 不替换数据：没有本地官方 UMA native train/val parquet 路径就 BLOCK。
7. 不调用 optimizer：必须设置 `UMA_CREDIT_TRACE_EXIT_AFTER_ADVANTAGE=1`，runner 会在 advantage 后退出。

## 需要 H20 先确认/填入的本地路径

这些由 H20 对话只读扫描后填入；没有就停：

```bash
export UMA_MODEL_PATH="/absolute/local/path/to/Qwen3-4B-Instruct-2507"
export UMA_TRAIN_FILES_LITERAL="['/absolute/local/path/to/train.parquet']"
export UMA_VAL_FILES_LITERAL="['/absolute/local/path/to/dev.parquet']"
export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"
```

如果 H20 只有 Qwen2.5-7B/14B 或 RL-MemoryAgent 7B/14B，不允许替代。

## 第一阶段：只做 precheck

在 H20 上进入从 GitHub 拉下来的 `Gufrannn/MemAgent-Research` worktree 后：

```bash
export WORK_ROOT="/data/cw/memagent_work"
export RUN_MODE="precheck"
export UMA_MODEL_PATH="/absolute/local/path/to/Qwen3-4B-Instruct-2507"
export UMA_TRAIN_FILES_LITERAL="['/absolute/local/path/to/train.parquet']"
export UMA_VAL_FILES_LITERAL="['/absolute/local/path/to/dev.parquet']"
export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"

bash code_research/experiments/UMA-Official-Anatomy-20260901/tools/run_p33_2_v3_2_h20_preupdate_advantage_trace.sh
```

必须看到：

```text
P33_2_V3_2_PRECHECK_OK
RUN_MODE_PRECHECK_COMPLETE_NO_ROLLOUT_NO_TRAINING_NO_OPTIMIZER
```

如果 precheck 失败，回传日志，不要手工修。

## 第二阶段：one-step dry-run

precheck 通过后才运行：

```bash
export WORK_ROOT="/data/cw/memagent_work"
export RUN_MODE="one_step"
export RUN_TAG="p33_2_v3_2_h20_4b_preupdate_$(date +%Y%m%d_%H%M%S)"
export UMA_MODEL_PATH="/absolute/local/path/to/Qwen3-4B-Instruct-2507"
export UMA_TRAIN_FILES_LITERAL="['/absolute/local/path/to/train.parquet']"
export UMA_VAL_FILES_LITERAL="['/absolute/local/path/to/dev.parquet']"
export EMBEDDING_SERVICE_ENDPOINT="http://localhost:8080/embeddings"

export UMA_TRAIN_BATCH_SIZE="8"
export UMA_ROLLOUT_N="4"
export UMA_N_GPUS="8"

bash code_research/experiments/UMA-Official-Anatomy-20260901/tools/run_p33_2_v3_2_h20_preupdate_advantage_trace.sh
```

必须看到：

```text
P33_2_EXIT_AFTER_ADVANTAGE_BEFORE_OPTIMIZER
P33_2_V3_2_ONE_STEP_COMPLETE_EXITED_BEFORE_OPTIMIZER
```

如果 trainer 在这之前失败，回传 `RUN_TAG`、stdout/stderr、env inventory、`nvidia-smi` 文件路径，不要换模型/数据/runner。

## 完成后回传格式

请按以下 compact 格式给主线：

```text
P33.2-v3.2 H20 STATUS: COMPLETE/BLOCKED
commit: <MemAgent-Research exact commit>
official_uma_commit: 768f9620231bae11264771f59e43a4839506cf94
RUN_MODE: precheck/one_step
RUN_TAG: ...
OFFICIAL_SRC_DIR: ...
INSTRUMENTED_DIR: ...
TRACE_DIR: ...
SUMMARY_DIR: ...
model_path: ...
train_files_literal_sha1: ...
val_files_literal_sha1: ...
env_inventory_path: ...
nvidia_smi_path: ...
exit_marker_seen: yes/no
trace_files_count: ...
summary_files: ...
sha256:
  prepare_manifest_stdout: ...
  instrumentation_manifest: ...
  instrumentation_diff: ...
  trace_summary_csv/json/md if produced: ...
boundary:
  no_generation_protocol_change: yes
  no_prompt_operator_split_metric_change: yes
  no_optimizer_step: yes/no/unknown
  no_model_data_download: yes
  no_dependency_install: yes
```

## 结果标签

- 如果缺官方 4B model/data/env：`BLOCKED_SOURCE_ENV`
- 如果 precheck 过但 one-step 启动失败：`BLOCKED_RUNTIME`
- 如果看到 advantage trace 和 safe-exit marker：`COMPLETE_PREUPDATE_TRACE`

任何没有 safe-exit marker 的 one-step 结果都不能声称“无 optimizer step”。

# Control-Sufficient Memory: paired headroom test

This branch tests a narrow claim before training: a memory representation that
preserves action preconditions and observed state transitions should improve
future tool selection beyond a same-budget state/fact memory.

## Causal comparison

The official Mem2ActBench-small files are used unchanged. For every benchmark
item, each referenced historical session is compiled before the future query is
shown. The writer never receives `query`, `evolution_chain`, the target schema,
or the gold tool call. The frozen memory is then paired with the official query
and target schema for vLLM execution.

Conditions are `no_memory`, `full_history`, generic `summary`, current `state`,
and `control`. `state` and `control` share the same output-token budget. The key
contrast is therefore `control - state`, not control versus an artificially weak
no-context baseline.

Primary metric: exact tool call (correct name, exact argument keys, values, and
types). The summary also reports JSON validity, tool-name accuracy, expected
argument accuracy, L1--L4 strata, paired wins, and paired bootstrap intervals.

## One-time remote setup

```bash
export WORK_ROOT=/data/cw/memagent_work
source "$WORK_ROOT/.venv/bin/activate"

cd "$WORK_ROOT/datasets"
git clone https://github.com/Cantaloupe-M/Mem2ActBench.git
```

Start the existing Qwen2.5-7B-Instruct vLLM service on an available GPU. The
experiment expects the OpenAI-compatible endpoint at port 8001 and never uses a
Hugging Face generation path.

```bash
export CUDA_VISIBLE_DEVICES=4
vllm serve "$WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --served-model-name qwen25-7b \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85
```

## Run

First run 16 examples as a protocol smoke test, then run at least 100 examples
for an initial decision. The JSONL is resumable and per-session memories are
cached, so rerunning the same command does not regenerate completed work.

```bash
export WORK_ROOT=/data/cw/memagent_work
export NUM_SAMPLES=16
bash experiments/control_memory/run_mem2act_control_headroom.sh

export NUM_SAMPLES=100
bash experiments/control_memory/run_mem2act_control_headroom.sh
```

Results are written under:

```text
/data/cw/memagent_work/logs/control_memory/mem2act_headroom/
```

Proceed to predictive/control auxiliary RL only if `control_minus_state` and
`control_minus_summary` improve exact tool calls, with the effect present on
L2/L3/L4 rather than only L1. If control and state tie, this benchmark does not
identify action-conditioned structure; move the same query-unknown protocol to
workflow/gotcha subsets of LongMemEval-V2 or MemoryArena rather than training on
an unsupported objective.

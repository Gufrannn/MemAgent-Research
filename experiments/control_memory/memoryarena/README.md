# MemoryArena reward-contrastive control memory

This is the next falsification test after the Mem2ActBench pilots. It uses the
official MemoryArena loop and its official action/observation/reward chunks.
It does not construct a dataset.

The paired backends share the same lexical retriever and grounded raw episodes:

- `control_raw`: retrieves raw official episodes only.
- `control_boundary`: writes query-unknown decision cards, then deliberately
  retrieves both successful and failed relevant episodes. The downstream agent
  receives the learned failure boundary together with unmodified raw evidence.

This differs from descriptive transition memory: the proposed unit is the
success/failure boundary for an action under state preconditions. The key test
is whether reward-contrastive pairing improves later actions over the identical
raw retrieval substrate.

## Install into the official preview

```bash
git clone --depth 1 https://github.com/ZexueHe/MemoryArena.git \
  /data/cw/memagent_work/code/MemoryArena-official

python experiments/control_memory/memoryarena/install_backend.py \
  /data/cw/memagent_work/code/MemoryArena-official
```

The installer is idempotent and fails closed if the official source layout has
changed. It adds `control_raw` and `control_boundary` to the official memory
server. Point the writer to the existing vLLM endpoint:

```bash
export CONTROL_MEMORY_BASE_URL=http://127.0.0.1:8001
export CONTROL_MEMORY_MODEL=qwen25-7b
export CONTROL_MEMORY_TOP_K=4
export CONTROL_MEMORY_MAX_CONTEXT_CHARS=14000
```

Run matched official tasks once with each backend. Start with WebShop because
its official memory entry contains the full turn trace, final observation, and
reward. Do not compare different task IDs or model settings across conditions.

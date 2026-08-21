# COMMIT/RETAIN capture — legacy GPU45 branch identity

This profile is an execution-only clone of the frozen COMMIT/RETAIN capture
contract. It changes no sample, model, prompt, seed, intervention, decoding,
metric, or claim boundary. Despite the historical `gpu45` branch/profile name,
the physical devices are now mandatory runtime input through
`MEMAGENT_COMMIT_RETAIN_GPU_PAIR=A,B` and may be any two authenticated H20s,
including a non-contiguous pair such as `2,4`.

The profile remains independent in code identity only:

- physical devices: explicit canonical ascending `A,B`;
- project locks: `locks/memagent_h20_gpu_A.lock` and
  `locks/memagent_h20_gpu_B.lock`, the exact Original S128 curve namespace,
  simultaneously held in ascending order;
- output root: `logs/commit_retain_capture_frozen_20260821/<run-id>_gpuA_B`;
- branch: `h20/qwen25-7b-commit-retain-capture-gpu45-20260821`;
- P0 and capture entrypoints carry the `_gpu45` suffix.

The original GPU67 and historical GPU45 wrappers resolve the same explicit
pair contract. Neither wrapper implies a physical pair. Pair slug, indices,
UUIDs, and device names are frozen into P0, the parent credential, every pair,
the run receipt, supervisor ledger, and final audit.

`run_id` is globally one-time across both historical profiles and every GPU
pair. P0 rejects legacy unsuffixed default/GPU45 roots, any existing suffixed
pair root, and the atomic run-ID tombstone. A failed r3 cannot be restarted by
changing the pair slug; use a new r4 run ID.

The inherited backend contract freezes vLLM 0.8.2 tensor-parallel workers to
the `spawn` start method. P0 verifies both the environment and the method seen
by the installed vLLM runtime; capture receipts record the selected context and
whether the coordinator had initialized CUDA before constructing the engine.
This prevents CUDA state inherited through `fork` from producing a one-worker
startup followed by a distributed-store timeout.

Failed run roots are immutable evidence. In particular, a run that contains a
PASS P0 certificate plus a capture credential/log but no four-pair capture,
run receipt, or final report must be archived as a whole and must never be
continued. Recovery uses a new run ID and starts again at P0.

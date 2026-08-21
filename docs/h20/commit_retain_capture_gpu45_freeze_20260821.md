# COMMIT/RETAIN capture — physical GPU 4,5 profile

This profile is an execution-only clone of the frozen COMMIT/RETAIN capture
contract. It changes no sample, model, prompt, seed, intervention, decoding,
metric, or claim boundary. It binds the single two-way tensor-parallel vLLM
engine to physical NVIDIA H20 devices 4 and 5.

The GPU45 profile is independent from the original GPU67 profile:

- physical devices: `4,5`;
- project lock: `locks/memagent_gate_a_gpu_4_5.lock`;
- output root: `logs/commit_retain_capture_gpu45_frozen_20260821/<run-id>`;
- branch: `h20/qwen25-7b-commit-retain-capture-gpu45-20260821`;
- P0 and capture entrypoints carry the `_gpu45` suffix.

The original GPU67 manifest, wrappers, lock, and output paths remain the
default profile and are not redirected by this variant.

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

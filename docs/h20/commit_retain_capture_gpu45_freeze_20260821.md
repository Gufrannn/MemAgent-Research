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

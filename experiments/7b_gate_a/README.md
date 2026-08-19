# MemAgent Qwen2.5-7B Gate A

This branch migrates only the infrastructure needed to make later mechanism
claims meaningful:

- independent, deterministic per-trajectory and per-turn vLLM request seeds;
- append-only `rollout_seed_audit.jsonl`;
- stable validation identity fields kept out of reward aggregation;
- dense HotpotQA *training* reward (evaluation remains separate EM/F1);
- strict vLLM 2-step smoke followed by explicit resume from step 2 to step 3.

It deliberately excludes legacy Path interventions, pair rewards, NCR and all
other candidate methods.

The launcher supports explicit 2/4/8-GPU allocations with matching FSDP size.
A 2-GPU H20 run is an infrastructure smoke only; it is not comparable to the
official training shape or admissible as a method/baseline performance result.

## Gate A pass condition

`verify_gate_a.py` must pass all of the following:

1. independent seed records exist and are unique within each global step;
2. step 2 and step 3 each contain model/optimizer/extra/data checkpoint files;
3. fresh log reaches step 2 and exposes actor backward/update;
4. resume log explicitly loads `global_step_2` and reaches step 3;
5. logs expose FSDP-to-vLLM weight synchronization;
6. actor model shards change from step 2 to step 3.

Passing Gate A proves the execution chain only. It does not prove improved QA,
useful memory, restored learning signal, or any scientific mechanism.

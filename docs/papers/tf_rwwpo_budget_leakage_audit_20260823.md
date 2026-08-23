# TF-RWWPO budget, leakage, and adaptive-use audit

## Evidence status

This note separates static facts proven by the release source from cardinalities
that require the H20 evidence files. It does not infer disjointness from the
names `train` and `dev`.

| Scientific conclusion | Direct leakage | Adaptive benchmark risk | Paper wording | Remaining blocker |
|---|---|---|---|---|
| T25 is a single-seed early-budget pilot, not sufficient training or convergence. The static maximum is 25 global proposals, 100 prompt groups, 200 trajectories, one PPO epoch and one global minibatch per step. GRPO disables the critic; no reward/auxiliary model is fit. | `PENDING_H20_CONTENT_HASH_AUDIT`. The certificate joins canonical `(question hash, context hash, ground-truth hash)` content keys and `(question hash, context hash)` root keys for the exact first 100 production-eligible actor rows against all 128 S128 rows. Dataset-local integer IDs are diagnostic only. | `CONFIRMED`. T5/10/15/20/25 S128 results were viewed and explicitly motivated the controller pivot. Therefore `selection ∩ S128 = 128`; this is adaptive reuse, not gradient leakage. | S128 is now called an adaptive development benchmark. T25 thresholds are development-screen gates only. Claims of blind final testing, sufficient training, convergence, or confirmed superiority are forbidden. | Run the read-only H20 certificate. Threshold-selection provenance is still pending. A confirmatory set must be preregistered, content/root-disjoint, and untouched until code, controller, thresholds, and selection rules are frozen. |

## Frozen data and optimizer inventory

- Actor source: `/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_train_32k.parquet`, SHA256 `798b7a2a9ece4f40884e2a9d02d165d7352df7763d1569ceaf402b45f76896f8`.
- Actor manifest: `manifests/h20/qwen25_7b_tf_rwwpo_seed2026.json`, SHA256 `640d4d48500f8ddbeceb3b3b7fe306a4ad372a78b4c4104352458fc5cecf0e9e`. It freezes seed 2026, batch 4, rollout `n=2`, mini-batch 4, no shuffle, and T25.
- Runtime filtering caveat: the manifest requests `data.max_prompt_length=8192`, but `MemoryDataset` mutates the live Hydra data config to `max_chunks * chunk_size = 40000` before the base dataset filters prompts (`recurrent/impls/memory.py`). The overlap audit uses the effective 40000-token predicate and reports how many source rows were skipped. Protocol evidence must state both requested and effective values; a post-mutation value must never be presented as the original immutable manifest value.
- Critic fit: empty, zero optimizer updates. `algorithm.adv_estimator=grpo` makes `use_critic=False`.
- Reference policy fit: empty, zero optimizer updates. Frozen reference log probabilities are evaluated on actor trajectories.
- Reward/auxiliary fit: empty, zero optimizer updates. The dense HotpotQA reward is a fixed scorer using actor-training labels, not a learned performance model.
- Performance early stopping: none. `trainer.val_before_train=False` and `trainer.test_freq=-1`; the T5 health gate uses on-policy mechanism evidence, not S128 scores.
- S128 data: `/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_dev.parquet`, SHA256 `54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6`.
- S128 resolved identity: `/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json`, SHA256 `6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411`, identity hash `351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a`.
- Hyperparameter/controller selection: S128 was used for controller-form selection. The inspected receipts are under `/data/cw/memagent_work/logs/rwwpo/rwwpo_seed2026_metricfix1_b485f2e/diagnostic_c07bc26_r2/certificates_3ad7c75/` with names `t5_s128_diagnostic.json`, `t10_s128_diagnostic.json`, `t15_s128_diagnostic.json`, `t20_s128_diagnostic.json`, and `t25_s128_diagnostic.json`; their exact file SHAs are `PENDING_H20_READBACK`. Whether `q_min`, the log-ratio cap, or the alpha grid were tuned from S128 is `PENDING_DOCUMENTARY_EVIDENCE`; absence of a recorded tuning event is not proof of non-use.

## Read-only H20 certificate

Run after checking out the exact clean release commit. Omit the optional ledger
arguments before a completed TF run; in that mode exact writer-turn/token and
commit counts remain `PENDING_TF_RUN`.

```bash
export TF_RWWPO_AUDIT_ROOT="$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates"

"$RWWPO_WORK_ROOT/.venv/bin/python" \
  tools/h20/audit_tf_rwwpo_budget_leakage.py \
  --manifest manifests/h20/qwen25_7b_tf_rwwpo_seed2026.json \
  --train "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  --tokenizer-root "$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --s128-data "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  --s128-resolved "$RWWPO_WORK_ROOT/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json" \
  --s128-resolved-sha256 6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411 \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --actual-ledger-dir "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/actual_loss" \
  --rollout-seed-audit "$RWWPO_OUTPUT/rollout_seed_audit.jsonl" \
  --execution-ledger "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/execution.jsonl" \
  --output "$TF_RWWPO_AUDIT_ROOT/budget_leakage.json"
```

The command verifies every input SHA, reconstructs the S128 manifest self-hash,
replays the production prompt filter, uses the sequential first-100 actor rows,
and independently validates the two rank actual-loss ledgers. It outputs hashes
and counts only; no question, context, or answer text is serialized.

## Untouched confirmation plan

No existing artifact is currently certified as an untouched confirmatory set.
Before any further method decision, preregister a candidate source population
and a deterministic hash-selection salt, subtract all actor-training, S128,
Capture32, and other previously inspected root/content inventories, and freeze
at least 512 remaining examples if the eligible population permits. Commit the
source file SHA, row-key inventory, code, controller, thresholds, and checkpoint
rule before generation. Evaluate it once after development and disclose every
access. Until the overlap certificate and access ledger pass, its status remains
`PENDING_PREREGISTRATION_AND_MATERIALIZATION`, not “untouched.”

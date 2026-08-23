# PRD-MemRL evidence-level self-check

## Budget interpretation

T25 is an **early-budget pilot**, not evidence of sufficient training or convergence. Per capacity the
frozen launcher schedules 25 global updates, 4 prompt groups/update, 2 independent rollouts/group:
100 rollout groups and 200 trajectories. PPO uses one epoch and one rollout-expanded minibatch per
global update, hence 25 actor optimizer updates. GRPO with `grpo_use_adv=False` has no learned critic,
so critic-fit updates are zero. The history-blind prior and projected dual each update once per global
update (25 each). Exact writer turns and writer/response token totals are data-dependent and remain
`PENDING_H20_EXECUTION_LEDGER`; they must be summed from authenticated Method rows, not estimated from
maximum lengths. “30 epochs” in the inherited launcher is inert because `total_training_steps=25`
terminates first. No result may be described as converged, fully trained, or a proof of superiority.

## Data roles and overlap

Actor and prior fit use `/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_train_32k.parquet` in
no-shuffle order; the prior sees stop-gradient writer samples from those same trajectories. There is no
critic. Capacity points and stopping rules are preregistered in the manifest; T5 checks only numerical
health. Fixed S128 uses `/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_dev.parquet`, SHA-256
`54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6`, plus stable resolved SHA-256
`6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411` and eval-manifest hash
`351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a`.

Direct train/S128 overlap is `PENDING_H20_EVIDENCE`, not presumed zero from filenames. Run
`tools/h20/audit_prd_data_overlap.py` on H20. It compares both canonical content triples
`(question hash, context hash, ground-truth hash)` and semantic root IDs. A nonzero train-pool overlap
is hard NO-GO. Critic-fit∩S128 is structurally zero because no critic exists. Selection∩S128 is
conservatively **128**: the entire certified Original S128 curve was inspected and used during program
development.

## Adaptive benchmark status

S128 is a development benchmark, not a blind final test. It has already supported Original curve
inspection, engineering canaries, route framing, and planned Method comparison. It may support paired
descriptive development results only. Capacity and hyperparameters must not be retuned from Method
S128 outcomes. A paper-level final test requires a separately frozen, previously untouched HotpotQA
split or external long-memory QA benchmark, with its manifest/content hashes committed before any
predictions are opened; proposed primary choice is an untouched held-out HotpotQA partition, with an
external query-blind LongMemEval subset as external-validity replication.

## Final five-column disposition

| Scientific conclusion | Direct leakage | Adaptive benchmark risk | Wording correction | Remaining blocker |
|---|---|---|---|---|
| T25 is an early-budget, same-compute pilot; it can estimate a local five-anchor frontier but not convergence. | Train∩S128 is PENDING until the H20 content/root audit; critic∩S128=0; prior follows train. | High: selection∩S128=128, so S128 is dev, not blind test. | Use “paired descriptive fixed-S128 pilot”; never “fully trained”, “converged”, “proved superior”, or “blind final test”. | H20 overlap certificate, actual writer-turn/token totals, Method execution, posthoc E1, and a preregistered untouched final test. |

Reproducible H20 command:

```bash
PYTHONPATH=/data/cw/memagent_work/code/MemAgent-Research-prd-memrl \
/data/cw/memagent_work/.venv/bin/python \
  /data/cw/memagent_work/code/MemAgent-Research-prd-memrl/tools/h20/audit_prd_data_overlap.py \
  --train /data/cw/memagent_work/datasets/hotpotqa/hotpotqa_train_32k.parquet \
  --validation /data/cw/memagent_work/datasets/hotpotqa/hotpotqa_dev.parquet \
  --stable-resolved /data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json \
  --output /data/cw/memagent_work/logs/prd_memrl/data_overlap_certificate.json
```

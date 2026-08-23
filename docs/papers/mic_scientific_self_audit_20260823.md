# MIC scientific self-audit (2026-08-23)

Inspected code commit: `ecfb3a45c0056fa133338ede7e3f693b90727316`.
Recorded H20 training commit: `6013dcc7afb24b91afbb31b37e14a7c54702b066`.
Run ID: `mic-main-v1-seed2026-runtimefix3`.

This report distinguishes configured facts, authenticated runtime facts, and
pending H20-only facts. A path name such as `train` or `dev` is never treated as
proof of row disjointness.

## 1. Training budget and claim boundary

The frozen launcher expands four prompt groups by `rollout_n=2`, yielding eight
trajectories per actor update. The PPO prompt minibatch is four, so each update
has one global rollout-expanded minibatch; the actor default is one PPO epoch.
The run has 25 fixed optimizer updates and no validation-driven stop. Therefore:

| Quantity | Configured/derivable count | Runtime certificate requirement |
|---|---:|---|
| Actor optimizer updates | 25 | 25 gradient-delivery receipts and T5/10/15/20/25 health audits |
| Rollout groups | 100 | 25 × 4 groups in `rollout_seed_audit.jsonl` |
| Trajectories | 200 | 25 × 4 × 2 unique stable trajectory IDs |
| PPO epochs | 1/update | frozen trainer configuration |
| Global PPO minibatches | 1/update; 25 total | prompt minibatch 4 equals train batch 4 |
| Writer turns | `PENDING_H20_CERTIFICATE` | count non-prewrite states in T25 critic checkpoint |
| Writer/answer tokens | `PENDING_H20_CERTIFICATE` | sum exact delivery token receipts over 25 updates |
| Neural critic optimizer updates | 0 | GRPO actor run has `use_critic=false` |
| MIC closed-form critic refits | 25 | one cumulative OOF checkpoint per update |
| Fold ridge solves | expected 100 | sum actual occupied-fold receipts from all critic checkpoints |
| Separate prior/auxiliary fits | 0 | no such model is connected to the training path |

Code evidence: `experiments/7b_gate_a/run_gate_a.sh:92-106,118-120,126-145,175-185`,
`verl/trainer/config/ppo_trainer.yaml:32-53`, and
`verl/trainer/ppo/ray_trainer.py:2182-2220,2445-2617`.

**Conclusion:** T25 is an early-budget, single-seed, same-budget pilot. Twenty-five
updates over 200 trajectories, with no learning-curve plateau test, no longer
schedule, and no multi-seed replication, cannot establish convergence or
"sufficient training." A negative result rejects the frozen pilot criterion; it
does not prove that MIC is ineffective after adequate optimization.

## 2. Data roots and authenticated identities

| Consumer | Root/path | Authority and SHA status |
|---|---|---|
| Actor optimizer | `/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_train_32k.parquet` | accepted-protocol SHA `798b7a2a9ece4f40884e2a9d02d165d7352df7763d1569ceaf402b45f76896f8`; certificate recomputes bytes |
| MIC critic | exact on-policy subset reconstructed from actor `rollout_seed_audit.jsonl`; cumulative state/outcome rows in T25 `critic.json` | seed-audit, critic-file, and critic-payload SHAs emitted by certificate |
| Prior/auxiliary model | none | no fit data |
| Hyperparameter selection | frozen MIC manifest and E0 synthetic toy; pre-run human provenance incomplete | manifest SHA + `PENDING_EXTERNAL_LAB_NOTE` |
| Early stopping | none (`val_before_train=false`, `test_freq=-1`, total updates fixed) | five health-audit SHAs |
| Fixed S128 | `/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_dev.parquet` | accepted-protocol SHA `54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6`; resolved identity manifest and certificate both bind bytes |
| S128 identity | `/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821/certificates/p0_resolved_manifest.json` | `75167a3179698ce8edf0a2da6164795bc495952c66b9e1ea09b2cda91c5b0bb6` |

The audit computes two intersections independently of dataset names and semantic
indices:

1. root content = SHA-256(canonical question hash, context hash);
2. full example = SHA-256(root content hash, canonical ground-truth hash).

It reconstructs the exact actor/critic source subset from stable trajectory
records and reports both root and full-example intersections. Until the H20
certificate is produced, the following must remain honest `PENDING` values:

| Intersection | Count |
|---|---:|
| actor train ∩ S128, content root | `PENDING_H20_CERTIFICATE` |
| actor train ∩ S128, full example | `PENDING_H20_CERTIFICATE` |
| actor-consumed 100 groups ∩ S128, content root/full example | `PENDING_H20_CERTIFICATE` |
| critic fit ∩ S128, content root | `PENDING_H20_CERTIFICATE` |
| critic fit ∩ S128, full example | `PENDING_H20_CERTIFICATE` |
| automated early-stop/checkpoint selection ∩ S128 | 0 by frozen execution contract |
| human adaptive exposure ∩ S128 | 128 |

## 3. Adaptive use

All five MIC S128 anchor aggregates and the certified Original curve were
inspected. This is adaptive benchmark exposure even though no performance-based
early stop occurred and no retraining followed the revealed metrics in the
recorded run. S128 is therefore a **development benchmark**, not a blind final
test. The frozen, certificate-hashed disclosure is represented by
`docs/papers/mic_adaptive_use_disclosure_20260823.json` and is hashed into the
H20 certificate.

The replacement final test must be frozen before revealing any output:

1. Independently select at least 512 previously untouched rows.
2. Content-hash deduplicate against actor training, the exact critic subset,
   S128, and every human/model-selection example.
3. Commit the identity manifest and SHA before generation.
4. Evaluate only the already-frozen T25 checkpoint once; do not reveal T5--T20.
5. Freeze separate MuSiQue and 2WikiMultiHopQA manifests for external validity.

## 4. Five-column final status

| Scientific conclusion | Direct leakage | Adaptive benchmark risk | Paper wording correction | Remaining blocker |
|---|---|---|---|---|
| Early-budget single-seed pilot only; no convergence/sufficient-training claim. The frozen S128 performance criterion is descriptive development evidence, not a final paper result. | `PENDING_H20_CERTIFICATE` for full-pool and actually-consumed content intersections. Direct-leakage NO-GO is based on consumed actor/critic rows; unused-pool overlap is reported separately as source-corpus risk. | High: 128/128 S128 rows are in the revealed benchmark; S128 is downgraded to dev. | Abstract/Introduction now explicitly forbid convergence, superiority, and blind-test language; lines 3-5, 34-38, 121-128, 138-142, and 310-343 of the framing document. | Produce scientific certificate; run E1; freeze untouched test; one-shot T25 evaluation; multi-seed and external-validity confirmation; close pre-run human-selection lab note. |

## 5. Reproducible read-only H20 entry

Run from the exact current MIC audit checkout after setting the same five MIC
environment variables used by the evaluation entry. This command does not take
GPU locks or launch a model:

```bash
export MEMAGENT_MIC_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_MIC_REPO_DIR=/data/cw/memagent_work/code/MemAgent-MIC-Audit
export MEMAGENT_MIC_EXPECTED_COMMIT=REPLACE_WITH_THIS_AUDIT_COMMIT
export MEMAGENT_MIC_GPU_PAIR=6,7
export MEMAGENT_MIC_RUN_ID=mic-main-v1-seed2026-runtimefix3

cd "$MEMAGENT_MIC_REPO_DIR"
bash scripts/h20/audit_scientific_evidence_qwen25_7b_mic.sh
```

Expected immutable output:

`/data/cw/memagent_work/logs/mic_frozen_20260822/mic-main-v1-seed2026-runtimefix3/certificates/scientific_evidence_audit.json`

The entry refuses overwrite. For a rerun, set
`MEMAGENT_MIC_SCIENCE_OUTPUT` to a new append-only certificate path and compare
the resulting file SHA; do not delete or replace the first certificate.

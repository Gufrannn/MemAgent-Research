# RWWPO-2 evidence-level self-check

**Scope.** This audit covers only Paper I / RWWPO. It distinguishes the completed
K1 T25 diagnostic, the passed RWWPO-2 pre-R50 evidence at commit `2d036904...`,
the first B/seed-2026 R50 attempt that failed before any optimizer commit, and
the later `5e9e60d...` release-test attempt that failed before GPU work because
two subprocess tests selected a non-project Python without Torch. That
124/126 attempt is `NO_GO` and cannot be reused. No valid R50 mechanism result
and no R400 performance result currently exist.

## 1. Training-budget interpretation

The completed K1 T25 run is an **early-budget, single-seed pilot**. It is not a
convergence or sufficient-training result. Its authenticated actual-loss receipt
reported 25 behavior rounds, one actor transaction per round, one full optimizer
minibatch per transaction, 50 rank receipts, eight accepted transactions but
only seven nonzero parameter commits (the accepted round-1 proposal had zero
movement), and no accepted proposal at T5/T10/T15/T20/T25. The static maximum
budget was 100 prompt groups and 200 rollout trajectories. Exact writer-turn,
writer-token, answer-token, and active-token counts remain `PENDING_H20_READONLY`
until the original tensor ledger is re-audited; they must not be guessed from
aggregate reward or S128 rows. Critic and auxiliary-fit optimizer updates were
zero.

The planned RWWPO-2 R400 budget is a **medium-budget controlled study**, not a
convergence target. Per cell and training seed it contains exactly 400 rollout
rounds, 1,600 prompt groups, at most 3,200 trajectories, two full-batch actor
transactions per round, one optimizer minibatch/step per transaction, and at
most 800 proposals. Writer turns/tokens are data-dependent and are reported only
by the post-run attempt auditor. Critic, prior, reward-model, and auxiliary-fit
optimizer updates remain zero.

The wording firewall is explicit in
`docs/papers/rwwpo2_r400_k2_preregistration_20260823.md`: R400 is not converged
or sufficiently trained, K1 is controller-diagnostic only, and no unrun RWWPO-2
performance claim is authorized. The legacy working draft is subordinate to
that preregistration and is being revised; any T25 performance paragraph is
historical diagnostic context only.

## 2. Data roots, hashes, and canonical overlap audit

| Use | Absolute H20 path | Frozen identity | Optimizer use |
|---|---|---|---|
| Actor training | `/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_train_32k.parquet` | SHA256 `798b7a2a9ece4f40884e2a9d02d165d7352df7763d1569ceaf402b45f76896f8` | actor policy only |
| Critic fit | none | no dataset; zero updates | none |
| Prior fit | none | no dataset; zero updates | none |
| Auxiliary fit | none | no dataset; zero updates | none |
| Hyperparameter/method selection | fixed S128 was inspected during the K1-to-K2 pivot | see next row | no optimizer use, but adaptive use |
| S128 development | `/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_dev.parquet`; resolved `/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json` | data SHA256 `54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6`; resolved SHA256 `6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411`; eval-manifest hash `351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a` | forbidden during R50/R400 training |
| Confirmatory set | not yet materialized | `PENDING_SEALED_DISJOINT_AT_LEAST_512` | forbidden for training/selection |

`tools/h20/audit_rwwpo2_data_boundary.py` reconstructs the exact sequential
actor-training prefix after the runtime prompt-length filter, constructs
`root_key = SHA256(canonical(question_hash, context_hash))` and
`content_key = SHA256(canonical(question_hash, context_hash,
ground_truth_hash))`, and joins those sets to the 128 identities authenticated
by the frozen S128 resolved manifest. It emits counts and hashes only—never raw
questions, contexts, or outcomes.

The authenticated H20 data-boundary report is
`/data/cw/memagent_work/logs/rwwpo2_evidence/rwwpo2_evidence_2d03690_r1/data_boundary.json`.
It reports:

- `actor-train ∩ S128`: content 0, canonical root 0;
- `critic-fit ∩ S128`: exactly 0 because no critic is fit;
- `prior/auxiliary-fit ∩ S128`: exactly 0 because neither fit exists;
- `selection ∩ S128`: 128 by design, because all S128 results were viewed.

The last item is adaptive benchmark use, not direct optimizer leakage. The
dataset-local semantic-ID diagnostic intersects in 73 values, but canonical
content/root hashes intersect in zero; it is not treated as identity evidence.
Any nonzero canonical actor root/content intersection makes preflight fail.

## 3. Adaptive use and next untouched test

S128 T5/T10/T15/T20/T25 values were viewed, used to diagnose hard rollback, and
informed the RWWPO-2 controller/geometry pivot. S128 is therefore an
**adaptive development benchmark**. It must never be called blind, held-out, or
final, and it cannot select R50/R400 hyperparameters, seeds, stopping, or
checkpoints.

The next confirmatory test is a separately sourced, at-least-512-root set that
has never been generated or scored by this method family. Before R400,
`tools/h20/seal_rwwpo2_confirmation_set.py` must bind its file SHA, canonical
root/content inventory digests, zero overlap with the exact actor-training
prefix and S128, greedy generation seed/decoding, model/tokenizer revisions,
lengths, stable join, and missingness rule. Preflight rejects R400 without that
seal. The set is opened once only after all eight paired B/D/E training seeds,
attempt audits, mechanism analyses, and code are frozen.

## 4. Five-field scientific status

| Scientific conclusion | Direct leakage | Adaptive benchmark risk | Paper wording | Remaining blocker |
|---|---|---|---|---|
| K1 whole-path/per-write/tokenwise objectives are single-pass degenerate under the proposition's complete-state assumptions; old T25 identifies controller dynamics only. RWWPO-2 remains scientifically KEEP, but the first R50 attempt is implementation-failure evidence only and contains zero valid commits. | Canonical actor-train/S128 content and root intersections are both 0; critic/prior/aux are 0 by construction. | High and acknowledged: S128 is development-only and cannot support confirmation. | No superiority, sufficient-training, convergence, or blind-test claim; R400 is a medium-budget conditional test. | Release tests, boundary, base protocol and old numeric oracle passed at `2d036904...`; live/oracle gradient-sketch implementation drift caused first-update NO_GO. Release and rerun the shared registered projection plus long-context streaming oracle under a new commit/root, then restart B/seed2026 from fresh base. |

## 5. Reproducible read-only H20 entry

Run this only from the exact clean commit approved by the independent release
review; replace the output root with a new one-use evidence directory. It does
not start training or use a GPU:

```bash
export RWWPO2_MANIFEST="$PWD/manifests/h20/qwen25_7b_rwwpo2_r400_k2_seed2026.json"
export RWWPO2_MANIFEST_SHA256="$(sha256sum "$RWWPO2_MANIFEST" | awk '{print $1}')"
export RWWPO2_AUDIT_ROOT=/data/cw/memagent_work/logs/rwwpo2_preflight/<NEW_ONE_USE_ID>
mkdir "$RWWPO2_AUDIT_ROOT"

/data/cw/memagent_work/.venv/bin/python \
  tools/h20/audit_rwwpo2_data_boundary.py \
  --manifest "$RWWPO2_MANIFEST" \
  --manifest-sha256 "$RWWPO2_MANIFEST_SHA256" \
  --train /data/cw/memagent_work/datasets/hotpotqa/hotpotqa_train_32k.parquet \
  --tokenizer-root /data/cw/memagent_work/models/Qwen2.5-7B-Instruct \
  --s128-data /data/cw/memagent_work/datasets/hotpotqa/hotpotqa_dev.parquet \
  --s128-resolved /data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json \
  --s128-resolved-sha256 6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411 \
  --expected-commit "$(git rev-parse HEAD)" \
  --output "$RWWPO2_AUDIT_ROOT/data_boundary.json"
```

Required report location is the absolute path printed by the command. Its
`git_commit`, file SHA, signed `report_sha256`, intersection counts, and PASS or
NO-GO decision must be copied into the run's P0. The command is retained as a
reproducibility entry. The `2d036904...` report is historical authenticated
pre-R50 evidence only; a source change requires a new commit-bound report and
may not inherit its PASS.

# Paired-effect candidate: capture32 preregistration and claim boundary

This branch prepares a read-only evidence pipeline for the paired-effect
credit candidate. It does not attach to the trainer, select a method, authorize
GPU work, or report a paper result.

## Scientific question

For one materialized writer candidate, the capture runs two matched futures:

```text
paired effect = COMMIT final-reader token F1 - RETAIN final-reader token F1
```

COMMIT loads the candidate state; RETAIN loads the exact pre-candidate state.
Both arms must share the same candidate, pre-state, future chunks and horizon,
reader checkpoint/decode, future writer RNG, reader RNG, cache contract,
configured cost budget, and one strict-vLLM process. Outcomes are independently
recomputed from final-reader text and the frozen ground truth. A stored target,
score, UUID, or pair ID is never accepted as a substitute.

The diagnostic scorer uses only pre-branch candidate/state features. Four-fold
cross-fit is grouped by stable example. Each held-out prediction is generated
by a standardized ridge fit over the other 24 examples; its target cannot
enter its model, standardization, or fold-mean baseline. A full-capture refit
is retained only for diagnostics and carries `deployment_use_authorized=false`.

## Frozen capture32 cohort

The preregistration is
`manifests/h20/qwen25_7b_paired_effect_capture32_preregistration.json`.
It binds the existing project S128 parquet and Qwen2.5-7B tokenizer:

- validation parquet SHA-256:
  `54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6`
- interface-neutral S128 manifest hash:
  `351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a`
- tokenizer manifest SHA-256:
  `1567e178abe4f245846c6bd59e7e6f3b7e842fde92200ddfc74851559a402023`
- complete model-file manifest SHA-256:
  `0b5381a2d40dfcad3d72be1f9cfc335433c6b7e3012042f1b1cc768447139fc7`
- committed 128-row Stable-I authority file SHA-256:
  `8c7c34cf884972325f5f42c0541f2f8b12ff2c60eb81aeb71b5b59555c899396`

Before any capture32 outcome is observed, all 128 rows are sorted by:

```text
(writer_turn0_prompt_token_length ASC, source_order_index ASC)
```

The selected zero-based sorted positions are:

```text
1, 5, 9, 13, 17, 21, 25, 29,
33, 37, 41, 45, 49, 53, 57, 61,
65, 69, 73, 77, 81, 85, 89, 93,
97, 101, 105, 109, 113, 117, 121, 125
```

This is one fixed offset from each four-row prompt-length stratum. The earlier
observed capture4 positions `15,47,79,111` are disjoint and explicitly
excluded. Selection may use only prompt length and source order. Arm outcomes,
reader answers, rewards, F1/EM, candidate output, actual cost, scorer output,
runtime UUID, pair ID, ground truth, and ground-truth hash are forbidden.

The preregistration contains the complete 128-row prompt ranking and literal
32-row identity inventory. Stable example/root/write IDs, request seeds,
source/token hashes, row hashes, inventory hashes, and the self-hash are all
recomputed by the auditor. Key frozen digests are:

- preregistration: `4dad37906f60131b04756b79f2cf0bafc978833a0bbeb6f0566ec4aa7e7482f5`
- selected inventory: `0434147f4b6d7878b31662d70b7fce0ee263b989e2106028d7f6ac8b3bb97d87`
- fold membership: `6d78a2bebe2f86b962d6c633c215575d283c73c8e0d25f2d823971f065d4c75a`

The literal inventory is not accepted merely because its own hashes agree.
The auditor first authenticates
`manifests/h20/qwen25_7b_paired_effect_s128_authority.json`, binds every
selected and unselected stable identity to its complete 128-row Stable-I
payload, and independently recomputes trajectory/request seeds and writer-turn
schedules. For a completed capture it additionally rereads the exact parquet,
hashes every model/tokenizer file, applies the frozen tokenizer and writer
template to all 128 rows, and reproduces the ranking and selected inventory
byte-for-byte.

Fold assignment is frozen as sorted stable-example ID round-robin, producing
exactly four folds of eight held-out examples and 24 fit examples per fold.
Runtime UUIDs, run IDs, ledger order, and pair IDs never define membership.

## Attrition and artifact state

Before a valid capture32 P0 commitment exists, absent artifacts yield
`PENDING`. The P0 certificate is the commitment point and must be written
before the first generation. After it exists, any missing pair, arm, turn,
reader outcome, ledger, resolved manifest, or final report is
`FAIL/PAIRED_EFFECT_NO_GO:CAPTURE32_ATTRITION`.

The exact inventory is 32/32. There is no complete-case analysis, replacement,
nearest-row substitution, run stitching, or capture4 padding. A restart uses a
new run ID and executes the full 32 again. A legal empty model answer is an
observed zero-score outcome; a missing receipt is attrition.

The capture32 producer closure emits exactly these run-root artifacts plus two
local export candidates under `WORK_ROOT/provenance/commit_retain_capture32`:

```text
certificates/p0_preflight.json
certificates/p0_resolved_manifest.json
commit_retain_capture32_execution_ledger.jsonl
credentials/capture_child.json
credentials/capture_child_consumed.json
captures/commit_retain_pairs.jsonl
captures/run_receipt.json
certificates/commit_retain_capture32_final_report.json
```

The consumer authenticates their exact paths, schemas, SHA chain, Git blobs,
32 identities, canonical pair envelopes, prompts, state/candidate relation,
RNG/cost contracts, one-process vLLM receipt, and contiguous global generation
calls. The separately named capture32 producer now lives in this Git closure;
it does not modify the trainer or authorize a GPU run. It requires an explicit
ascending pair of two physical H20 indices, per-device locks plus intersecting
legacy aggregate locks, one parent-issued credential, one process/engine, and
exactly 353 generation calls. P0, all three bindings, expected pair binding,
and every pair must agree on the selected physical pair, checkpoint, strict
vLLM 0.8.2, templates, decode, engine config, multiprocessing mode, lock
inodes, credential consumption, and the exact call count. The local anchors
remain `PENDING_EXTERNAL_SIGNATURE` and cannot authorize training.

## Frozen admissibility thresholds

Capture32 must have all 32 unique stable examples, at least eight effects with
absolute delta F1 at least `0.01`, at least three distinct `1e-6` effect bins,
mean absolute effect at least `0.02`, and target variance at least `0.0001`.
The held-out scorer must improve on the fit-fold-mean baseline MSE by at least
5%, have Pearson correlation at least `0.20`, and improve in at least three of
four folds. Non-finite scores or any fold other than 8 held-out/24 fit fail.

These thresholds, feature order, ridge value (`1.0`), fold rule, baseline, and
standardization are frozen before capture32. They cannot be tuned on its
outcomes. Capture4 cannot satisfy them even if duplicated or if its diagnostic
metrics look favorable.

## Report decisions

- no capture32 and no valid pilot: `PENDING / PAIRED_EFFECT_CAPTURE_PENDING`
- valid capture4 only: `PENDING / PAIRED_EFFECT_CAPTURE4_PILOT_ONLY`
- partial capture32 after P0: `FAIL / PAIRED_EFFECT_NO_GO:CAPTURE32_ATTRITION`
- invalid/tampered evidence: `FAIL / PAIRED_EFFECT_NO_GO:INVALID_CAPTURE_OR_PIPELINE`
- complete capture32 but failed signal: `FAIL / PAIRED_EFFECT_NO_GO:SCORER_ADMISSIBILITY`
- complete internally valid capture32 with acceptable signal but no external
  provenance anchor: `PENDING / PAIRED_EFFECT_CAPTURE_PROVENANCE_PENDING`

Report schema v1 intentionally permits only `PENDING` or `FAIL`. Internal Git
blobs and SHA chains are tamper-detecting, but an attacker controlling the
whole run directory can rebuild them. A later reviewed commit must freeze an
external immutable/signed anchor before a top-level PASS is possible. Even an
admissibility PASS would only mean evidence is ready for method review; it
would not select the method, set lambda, connect a live scorer, or authorize
training.

## Credit-routing safety contract

The offline tensor helper routes only within a stable question group when all
QA rewards tie, every row is eligible, and the group is not fully exact-correct.
Centered score bonuses have trajectory-total semantics and are divided over
all valid non-final writer tokens. Final-answer rows and non-target groups are
asserted bitwise unchanged. The helper emits
`trainer_integration_authorized=false` and is not called from `trainer.fit()`.

## Development-set warning and next gate

These 32 examples are drawn from the fixed S128 and therefore constitute
development evidence. If their outcomes fit or select the method, they cannot
also be treated as untouched confirmatory evaluation rows in the paper. A
later performance protocol must use a disjoint untouched evaluation cohort or
state this exclusion explicitly.

Once a real capture32 exists, this read-only audit should take minutes. The GPU
capture itself is expected to take roughly 3.5–8 hours on two H20s, extrapolated
from the four-pair pilot; that estimate is not a run authorization.

Only after capture32 scientific and provenance review may a separate branch
preregister an authenticated score-receipt join, lambda, raw Qwen2.5-7B fresh
start, update-1 enablement, T5/T10/T15/T20/T25 anchors, strict-vLLM weight-sync
audits, and exact Original-matched configuration.

## Read-only consumer command after this branch is reviewed

```bash
export MEMAGENT_PAIRED_EFFECT_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_PAIRED_EFFECT_REPO_DIR=/data/cw/memagent_work/code/MemAgent-Research
export MEMAGENT_PAIRED_EFFECT_EXPECTED_COMMIT=<REVIEWED_PIPELINE_COMMIT>
export MEMAGENT_PAIRED_EFFECT_RUN_ID=paired_effect_capture32_audit_r1
export MEMAGENT_PAIRED_EFFECT_CAPTURE_RUN_ID=<EXACT_CAPTURE32_RUN_ID>

cd "$MEMAGENT_PAIRED_EFFECT_REPO_DIR"
/data/cw/memagent_work/.venv/bin/python \
  tools/h20/audit_qwen25_7b_paired_effect_candidate.py \
  --manifest manifests/h20/qwen25_7b_paired_effect_candidate_seed2026.json \
  --write
```

Exit code `2` is expected while the result is PENDING. This command is
read-only with respect to evidence and launches no GPU process.

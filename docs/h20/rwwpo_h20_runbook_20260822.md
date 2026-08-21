# RWWPO H20 frozen runbook

## Scientific state

`REFRAME / PENDING_ACTUAL_LOSS_LEDGER`. The code is train-capable but the GPU
entry is deliberately locked until independently generated E0 and frozen
Original actual-loss E1 receipts both PASS. Existing Original S128 curve and
Capture32 evidence do not contain the required current-policy loss tensors and
cannot be used to manufacture E1.

## Required checkout and immutable bindings

```bash
git fetch origin h20/qwen25-7b-rwwpo-t25-frozen-20260822
git checkout h20/qwen25-7b-rwwpo-t25-frozen-20260822
test "$(git rev-parse HEAD)" = <RELEASE_SHA>
export RWWPO_REPO_DIR=$PWD
export RWWPO_EXPECTED_COMMIT=<RELEASE_SHA>
export RWWPO_WORK_ROOT=/data/cw/memagent_work
export GPU_PAIR=2,5                 # any two idle H20s, distinct and ascending
export RWWPO_RUN_ID=rwwpo_seed2026_primary
export RWWPO_ORIGINAL_RESOLVED_MANIFEST=/readonly/original/accepted_resolved_manifest.json
export RWWPO_ORIGINAL_RESOLVED_SHA256=<ACCEPTED_RESOLVED_MANIFEST_SHA256>
```

Never select an output that already exists. The run ID is semantic and explicit;
it is not a random UUID. Each physical card is locked independently at
`locks/memagent_h20_gpu_N.lock`. Occupied cards or lock conflicts cause NO-GO;
the scripts never kill a process.

## Read-only baseline import, E0, and E1

The baseline bundle is supplied by the experiment authority and contains
absolute read-only paths, per-file SHA-256 values, stable-key rows, and certified
aggregates. The importer copies no predictions.

```bash
mkdir -p "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates"
"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/import_rwwpo_original_baseline.py \
  --bundle /readonly/original_curve/baseline_bundle.json \
  --bundle-sha256 <AUTHORITY_PUBLISHED_BUNDLE_SHA256> \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/baseline_import.json"

"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/run_rwwpo_e0.py \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/e0.json"

"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/run_rwwpo_e1.py \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --original-ledger /readonly/original_actual_loss/actual_loss_rank0.jsonl \
  --original-ledger /readonly/original_actual_loss/actual_loss_rank1.jsonl \
  --output "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/e1.json"
```

If those Original actual-loss ledgers do not exist, stop at
`PENDING_ACTUAL_LOSS_LEDGER`. A future collection-only Original audit may be run
under separate authorization, but the certified Original curve must not be
rerun.

The collection closure is already wired into the accepted trainer entry: set
`RWWPO_COLLECT_ORIGINAL=1`, an absolute `RWWPO_LEDGER_DIR`, `RWWPO_Q_MIN`, and a
semantic `RWWPO_ATTEMPT_ID`. This mode retains Original tokenwise PPO loss and
only appends actual-loss tensors; it is mutually exclusive with
`RWWPO_ENABLE=1`. It is intentionally not invoked by this runbook because the
accepted Original curve is read-only and no new collection run is authorized.

## Fresh-base T5 and audit

```bash
RWWPO_PHASE=t5 bash scripts/h20/run_qwen25_7b_rwwpo.sh

"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/audit_rwwpo_run.py \
  --run-root "$RWWPO_WORK_ROOT/logs/memory_agent/qwen25_7b_rwwpo_seed2026_${RWWPO_RUN_ID}" \
  --actual-ledger-dir "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/actual_loss" \
  --execution-ledger "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/execution.jsonl" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" --target-step 5 \
  --output "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/t5_health.json"
```

Method starts at the same fresh Qwen2.5-7B base and RWWPO is enabled at update
1. No Original step-3 checkpoint is accepted. Run the repository's frozen S128
evaluator on Method-T5, recompute row metrics, and compare only to the imported
certified Original-T5. Continue only when health and the `-0.02` token-F1
non-inferiority gate pass.

The concrete fixed-S128 entry and read-only metric audit are:

```bash
export RWWPO_EVAL_RESOLVED_MANIFEST=/readonly/stable_s128/p0_resolved_manifest.json
export RWWPO_EVAL_RESOLVED_SHA256=<SHA256>
export RWWPO_EVAL_MANIFEST_HASH=<EVAL_MANIFEST_HASH>
RWWPO_EVAL_STEP=5 bash scripts/h20/run_rwwpo_s128_anchor.sh
"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/audit_rwwpo_s128.py \
  --eval-root "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/s128_t5" --step 5 \
  --validation "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  --resolved-manifest "$RWWPO_EVAL_RESOLVED_MANIFEST" \
  --expected-manifest-sha256 "$RWWPO_EVAL_RESOLVED_SHA256" \
  --output "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/t5_s128.json"
"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/compare_rwwpo_anchor.py \
  --method "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/t5_s128.json" \
  --baseline-import "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/baseline_import.json" \
  --step 5 --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/t5_compare.json"
```

## T10/T15/T20/T25 continuation

Repeat the following for `5→10`, `10→15`, `15→20`, and `20→25`; do not skip an
anchor. Each resume verifies the complete prior checkpoint and reuses the same
run identity and append-only ledgers.

```bash
RWWPO_PHASE=continue RWWPO_RESUME_STEP=5 RWWPO_TARGET_STEP=10 \
  bash scripts/h20/run_qwen25_7b_rwwpo.sh
RWWPO_PHASE=continue RWWPO_RESUME_STEP=10 RWWPO_TARGET_STEP=15 \
  bash scripts/h20/run_qwen25_7b_rwwpo.sh
RWWPO_PHASE=continue RWWPO_RESUME_STEP=15 RWWPO_TARGET_STEP=20 \
  bash scripts/h20/run_qwen25_7b_rwwpo.sh
RWWPO_PHASE=continue RWWPO_RESUME_STEP=20 RWWPO_TARGET_STEP=25 \
  bash scripts/h20/run_qwen25_7b_rwwpo.sh
```

After every anchor, run `audit_rwwpo_run.py` with the corresponding target,
then the frozen S128 evaluator. The final comparison joins stable IDs across all
five Method anchors and their imported Original counterparts. Required headline
gates are T25 token-F1 `+0.02`, five-anchor mean `+0.01`, and no anchor below
Original by more than `0.02`.

## Recovery and variants

- A constraint or infrastructure failure aborts before the next checkpoint.
  Preserve the ledger and log. Resume only from the last complete audited anchor.
- For an authorized recovery set a new semantic `RWWPO_ATTEMPT_ID`, e.g.
  `resume_10_to_15_recovery1`; never truncate a ledger.
- The single permitted T5 revision must use a new `RWWPO_RUN_ID`, manifest,
  experiment, output, and commit, and must name the preregistered failure class.
- Do not overwrite, delete, or reuse failed outputs.

Expected wall time is hardware- and queue-dependent. Based on the accepted
two-H20 configuration, reserve roughly 1–2 hours for preflight/E0/E1 and T5
health, then several hours per five-update continuation plus fixed-S128
evaluation. These are planning estimates, not claims of a local H20 run.

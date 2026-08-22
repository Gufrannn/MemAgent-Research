# RWWPO H20 frozen runbook

## Scientific state

`PAPER-FRAMING GO / PENDING_BASELINE_BUNDLE_MATERIALIZATION`. Original
actual-loss is an optional, separately authorized diagnostic and is not a Method
training gate. The main experiment uses the existing read-only Original curve,
starts Method once from the same fresh base, trains continuously through T25,
and saves T5/T10/T15/T20/T25.

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
export RWWPO_ORIGINAL_RESOLVED_MANIFEST=/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json
# Compute this file's SHA read-only on H20 and freeze the exact output; do not guess.
export RWWPO_ORIGINAL_RESOLVED_SHA256=$(sha256sum "$RWWPO_ORIGINAL_RESOLVED_MANIFEST" | awk '{print $1}')
```

Never select an output that already exists. The run ID is semantic and explicit;
it is not a random UUID. Each physical card is locked independently at
`locks/memagent_h20_gpu_N.lock`. Occupied cards or lock conflicts cause NO-GO;
the scripts never kill a process.

## Read-only baseline import and E0

The unified read-only authority is
`manifests/h20/rwwpo_original_evidence_authority_20260822.json`. The certified
Original curve is rooted at
`/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821`; its
canonical row digests are not standalone file SHA values. No certified
`baseline_bundle.json` currently exists. It must be materialized from artifact
paths and SHA values authenticated by that curve's PASS final report, with
EM/token-F1/format independently recomputed. Until then this gate is
`PENDING_BASELINE_BUNDLE_MATERIALIZATION`.

Once supplied, the baseline bundle contains
absolute read-only paths, per-file SHA-256 values, stable-key rows, and certified
aggregates. The importer copies no predictions.

```bash
mkdir -p "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates"
export RWWPO_CURVE_ROOT=/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821
export RWWPO_CURVE_FINAL=$RWWPO_CURVE_ROOT/certificates/original_s128_curve_final_report.json
export RWWPO_CURVE_RESOLVED=$RWWPO_CURVE_ROOT/certificates/p0_resolved_manifest.json
export RWWPO_CURVE_LEDGER=$RWWPO_CURVE_ROOT/original_s128_curve_execution_ledger.jsonl
export RWWPO_CURVE_FINAL_SHA256=$(sha256sum "$RWWPO_CURVE_FINAL" | awk '{print $1}')
export RWWPO_CURVE_RESOLVED_SHA256=$(sha256sum "$RWWPO_CURVE_RESOLVED" | awk '{print $1}')
export RWWPO_CURVE_LEDGER_SHA256=$(sha256sum "$RWWPO_CURVE_LEDGER" | awk '{print $1}')
export RWWPO_CURVE_LEDGER_TAIL_SHA256=$(tail -n 1 "$RWWPO_CURVE_LEDGER" | sha256sum | awk '{print $1}')
export RWWPO_BASELINE_ROOT=$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/baseline_materialization

"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/materialize_rwwpo_baseline_bundle.py \
  --final-report "$RWWPO_CURVE_FINAL" --final-report-sha256 "$RWWPO_CURVE_FINAL_SHA256" \
  --resolved-manifest "$RWWPO_CURVE_RESOLVED" --resolved-sha256 "$RWWPO_CURVE_RESOLVED_SHA256" \
  --execution-ledger "$RWWPO_CURVE_LEDGER" --ledger-sha256 "$RWWPO_CURVE_LEDGER_SHA256" \
  --ledger-tail-sha256 "$RWWPO_CURVE_LEDGER_TAIL_SHA256" \
  --validation "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  --validation-sha256 54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6 \
  --authority manifests/h20/rwwpo_original_evidence_authority_20260822.json \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" --output-root "$RWWPO_BASELINE_ROOT"

export RWWPO_BASELINE_BUNDLE=$RWWPO_BASELINE_ROOT/baseline_bundle.json
export RWWPO_BASELINE_BUNDLE_SHA256=$(sha256sum "$RWWPO_BASELINE_BUNDLE" | awk '{print $1}')
export RWWPO_BASELINE_BUNDLE_AUDIT=$RWWPO_BASELINE_ROOT/bundle_audit.json
"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/audit_rwwpo_baseline_bundle.py \
  --bundle "$RWWPO_BASELINE_BUNDLE" --bundle-sha256 "$RWWPO_BASELINE_BUNDLE_SHA256" \
  --materialization-report "$RWWPO_BASELINE_ROOT/materialization_report.json" \
  --final-report "$RWWPO_CURVE_FINAL" --resolved-manifest "$RWWPO_CURVE_RESOLVED" \
  --authority manifests/h20/rwwpo_original_evidence_authority_20260822.json \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" --output "$RWWPO_BASELINE_BUNDLE_AUDIT"

"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/import_rwwpo_original_baseline.py \
  --bundle "$RWWPO_BASELINE_BUNDLE" --bundle-sha256 "$RWWPO_BASELINE_BUNDLE_SHA256" \
  --bundle-audit "$RWWPO_BASELINE_BUNDLE_AUDIT" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/baseline_import.json"

"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/run_rwwpo_e0.py \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/e0.json"

```

The importer requires an independently audited bundle receipt. Original
actual-loss ledgers do not exist and are not required here. Never synthesize
them or rerun the certified Original curve.

The collection closure is already wired into the accepted trainer entry: set
`RWWPO_COLLECT_ORIGINAL=1`, an absolute `RWWPO_LEDGER_DIR`, `RWWPO_Q_MIN`, and a
semantic `RWWPO_ATTEMPT_ID`. This mode retains Original tokenwise PPO loss and
only appends actual-loss tensors; it is mutually exclusive with
`RWWPO_ENABLE=1`. It is intentionally not invoked by this runbook because the
accepted Original curve is read-only and no new collection run is authorized.

## One fresh-base Method run through T25

```bash
RWWPO_PHASE=full bash scripts/h20/run_qwen25_7b_rwwpo.sh

"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/audit_rwwpo_run.py \
  --run-root "$RWWPO_WORK_ROOT/logs/memory_agent/qwen25_7b_rwwpo_seed2026_${RWWPO_RUN_ID}" \
  --actual-ledger-dir "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/actual_loss" \
  --execution-ledger "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/execution.jsonl" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --expected-schema-version rwwpo-actual-loss-v1 \
  --expected-objective legacy --expected-controller legacy --target-step 5 \
  --output "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/t5_health.json"
```

Method starts at the same fresh Qwen2.5-7B base and RWWPO is enabled at update
1. No Original checkpoint is accepted. Runtime finite-loss/gradient, ledger,
checkpoint, and weight-sync assertions are the cheap T5 health gate. Unless a
numeric or contract failure aborts the process, training continues without an
S128 pause to T25 and retains all five anchors.

The concrete fixed-S128 entry and read-only metric audit are:

```bash
export RWWPO_EVAL_RESOLVED_MANIFEST=/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json
export RWWPO_EVAL_RESOLVED_SHA256=6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411
export RWWPO_EVAL_MANIFEST_HASH=351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a
RWWPO_EVAL_STEP=5 bash scripts/h20/run_rwwpo_s128_anchor.sh
"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/audit_rwwpo_s128.py \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
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

## Legacy recovery-only continuation

The commands below are recovery-only after an infrastructure interruption and
require a new authorized attempt identity. They are not the default main run.

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
two-H20 configuration, reserve several hours for the continuous T25 training,
then additional time for five fixed-S128 evaluations. These are planning
estimates, not claims of a local H20 run.

## Final five-anchor audit

```bash
"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/audit_rwwpo_five_anchor.py \
  --certificate-root "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates" \
  --baseline-import "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/baseline_import.json" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/certificates/final_five_anchor.json"
```

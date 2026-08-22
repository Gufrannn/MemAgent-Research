# TF-RWWPO H20 runbook (release-gated)

This runbook is not a claim that TF-RWWPO works.  Substitute only the exact
reviewed release SHA after PAPER GO, RELEASE GO, and a clean pushed branch.
Never reuse a run ID or output root.  Never kill another GPU process.

## Identity and evidence

```bash
cd /home/test001/memagent-rwwpo-release
git fetch origin h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822
git switch h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822 || \
  git switch --track -c h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822 \
    origin/h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822
git merge --ff-only origin/h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822
test "$(git rev-parse HEAD)" = '<REVIEWED_RELEASE_SHA>'
test -z "$(git status --porcelain)"

export RWWPO_REPO_DIR="$PWD"
export RWWPO_WORK_ROOT=/data/cw/memagent_work
export RWWPO_EXPECTED_COMMIT='<REVIEWED_RELEASE_SHA>'
export RWWPO_EXPECTED_BRANCH=h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822
export GPU_PAIR='<LOWER_IDLE_GPU>,<HIGHER_IDLE_GPU>'
export RWWPO_RUN_ID='tf_rwwpo_seed2026_primary_<SHORT_SHA>'
export RWWPO_PHASE=full
export RWWPO_ATTEMPT_ID=tf_rwwpo_primary
export RWWPO_MANIFEST="$RWWPO_REPO_DIR/manifests/h20/qwen25_7b_tf_rwwpo_seed2026.json"

export RWWPO_ORIGINAL_RESOLVED_MANIFEST=/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json
export RWWPO_ORIGINAL_RESOLVED_SHA256="$(sha256sum "$RWWPO_ORIGINAL_RESOLVED_MANIFEST" | awk '{print $1}')"

source scripts/h20/rwwpo_common.sh
```

Materialize new commit-bound E0 and baseline receipts from authenticated
read-only Original rows. Do not rerun Original:

```bash
mkdir -p "$RWWPO_CERT_ROOT"
"$RWWPO_PYTHON" tools/h20/run_tf_rwwpo_e0.py \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" --output "$RWWPO_E0"

export CURVE_ROOT=/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821
export CURVE_FINAL="$CURVE_ROOT/certificates/original_s128_curve_final_report.json"
export CURVE_RESOLVED="$CURVE_ROOT/certificates/p0_resolved_manifest.json"
export CURVE_LEDGER="$CURVE_ROOT/original_s128_curve_execution_ledger.jsonl"
export BASELINE_ROOT="$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/baseline_materialization"
export CURVE_FINAL_SHA="$(sha256sum "$CURVE_FINAL" | awk '{print $1}')"
export CURVE_RESOLVED_SHA="$(sha256sum "$CURVE_RESOLVED" | awk '{print $1}')"
export CURVE_LEDGER_SHA="$(sha256sum "$CURVE_LEDGER" | awk '{print $1}')"
export CURVE_LEDGER_TAIL="$("$RWWPO_PYTHON" -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])["record_sha256"])' "$CURVE_LEDGER")"

"$RWWPO_PYTHON" tools/h20/materialize_rwwpo_baseline_bundle.py \
  --final-report "$CURVE_FINAL" --final-report-sha256 "$CURVE_FINAL_SHA" \
  --resolved-manifest "$CURVE_RESOLVED" --resolved-sha256 "$CURVE_RESOLVED_SHA" \
  --execution-ledger "$CURVE_LEDGER" --ledger-sha256 "$CURVE_LEDGER_SHA" \
  --ledger-tail-sha256 "$CURVE_LEDGER_TAIL" \
  --validation "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  --validation-sha256 54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6 \
  --authority manifests/h20/rwwpo_original_evidence_authority_20260822.json \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" --output-root "$BASELINE_ROOT"

export BASELINE_BUNDLE="$BASELINE_ROOT/baseline_bundle.json"
export BASELINE_SHA="$(sha256sum "$BASELINE_BUNDLE" | awk '{print $1}')"
"$RWWPO_PYTHON" tools/h20/audit_rwwpo_baseline_bundle.py \
  --bundle "$BASELINE_BUNDLE" --bundle-sha256 "$BASELINE_SHA" \
  --materialization-report "$BASELINE_ROOT/materialization_report.json" \
  --final-report "$CURVE_FINAL" --resolved-manifest "$CURVE_RESOLVED" \
  --authority manifests/h20/rwwpo_original_evidence_authority_20260822.json \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" --output "$BASELINE_ROOT/bundle_audit.json"
"$RWWPO_PYTHON" tools/h20/import_rwwpo_original_baseline.py \
  --bundle "$BASELINE_BUNDLE" --bundle-sha256 "$BASELINE_SHA" \
  --bundle-audit "$BASELINE_ROOT/bundle_audit.json" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" --output "$RWWPO_BASELINE"
```

## GPU preflight and one continuous fresh run

```bash
nvidia-smi -i "$GPU_PAIR" --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
python3 tools/h20/audit_tf_rwwpo_source_firewall.py

screen -dmS tf-rwwpo-t25 bash -lc '
set -euo pipefail
cd "$RWWPO_REPO_DIR"
bash scripts/h20/run_qwen25_7b_tf_rwwpo.sh
'
```

The launcher takes both canonical per-GPU locks before CUDA initialization and
refuses occupied cards.  Training starts from the frozen fresh base, enables
the method at update 1, and targets T25 in one process.  At T5 it saves the
checkpoint and runs only the cheap mechanism gate: at least four nonzero
commits, median committed alpha >= 1/8, ESS/cap/behavior identity, rank
agreement, rollback closure, checkpoint inventory, and weight sync.  A failure
terminates the run and preserves its entire root.  A pass continues without
S128 evaluation and saves T10/T15/T20/T25.

## Monitoring and recovery

```bash
screen -r tf-rwwpo-t25
tail -F "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/train.log"
```

If the process fails, do not append or resume that root.  Preserve screen log,
ledger, locks, checkpoint, and tombstone.  After processes exit, locks release,
and both GPUs are empty, fix only on a new reviewed commit and use a new semantic
run ID from fresh P0.  An unmatched transaction intent makes audit fail closed.

## Post-training audit and evaluation

After T25, run `audit_rwwpo_actual_loss.py` and then:

```bash
"$RWWPO_WORK_ROOT/.venv/bin/python" tools/h20/audit_rwwpo_run.py \
  --run-root "$RWWPO_OUTPUT" --actual-ledger-dir "$RWWPO_LEDGER_DIR" \
  --execution-ledger "$RWWPO_EXECUTION_LEDGER" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --expected-schema-version rwwpo-actual-loss-v2 \
  --expected-objective whole_prefix --expected-controller feasible_backtracking \
  --target-step 25 --output "$RWWPO_CERT_ROOT/t25_health.json"
```

Then perform a second read-only re-audit. Only after both pass, evaluate the saved
T5/10/15/20/25 checkpoints using the same fixed-S128 manifest and strict-vLLM
launcher, import the certified Original five-anchor bundle read-only, and run
the per-anchor plus five-anchor auditors.  Performance rows remain separate
from the mechanism ledger.  B and C require different run IDs, manifests,
output roots, and release commits.

The complete formal five-anchor evaluation chain is:

```bash
export RWWPO_EVAL_RESOLVED_MANIFEST=/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json
export RWWPO_EVAL_RESOLVED_SHA256=6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411
export RWWPO_EVAL_MANIFEST_HASH=351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a
unset RWWPO_DIAGNOSTIC_ONLY RWWPO_EVAL_ATTEMPT_SUFFIX

for STEP in 5 10 15 20 25; do
  RWWPO_EVAL_STEP="$STEP" bash scripts/h20/run_rwwpo_s128_anchor.sh
  "$RWWPO_PYTHON" tools/h20/audit_rwwpo_s128.py \
    --expected-commit "$RWWPO_EXPECTED_COMMIT" \
    --eval-root "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/s128_t${STEP}" \
    --step "$STEP" --validation "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
    --resolved-manifest "$RWWPO_EVAL_RESOLVED_MANIFEST" \
    --expected-manifest-sha256 "$RWWPO_EVAL_RESOLVED_SHA256" \
    --output "$RWWPO_CERT_ROOT/t${STEP}_s128.json"
  "$RWWPO_PYTHON" tools/h20/compare_rwwpo_anchor.py \
    --method "$RWWPO_CERT_ROOT/t${STEP}_s128.json" --baseline-import "$RWWPO_BASELINE" \
    --step "$STEP" --expected-commit "$RWWPO_EXPECTED_COMMIT" \
    --output "$RWWPO_CERT_ROOT/t${STEP}_compare.json"
done

"$RWWPO_PYTHON" tools/h20/audit_rwwpo_five_anchor.py \
  --certificate-root "$RWWPO_CERT_ROOT" --baseline-import "$RWWPO_BASELINE" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_CERT_ROOT/five_anchor_final.json"
```

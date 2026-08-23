# RWWPO-2 K2 H20 runbook

## Release state

This runbook is executable only after the branch has an independently reviewed
release commit. Until that exact SHA is written below, no RWWPO-2 GPU run is
authorized. The first GPU experiment is the performance-free R50 mechanism
matrix. It never reads S128 and it does not report task performance.

```bash
export RWWPO_EXPECTED_COMMIT=<RELEASE_SHA_AFTER_REVIEW>
export RWWPO_EXPECTED_BRANCH=h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822
```

R400 is a later stage. It is fail-closed until the complete R50 matrix passes
and a disjoint confirmation set of at least 512 roots is sealed.

## 1. Exact checkout and immutable environment

Run each code block in a normal Bash shell. Do not paste the literal angle
brackets above; replace the release SHA first.

```bash
cd /home/test001/memagent-rwwpo-release
git fetch origin "$RWWPO_EXPECTED_BRANCH"
git switch -C "$RWWPO_EXPECTED_BRANCH" FETCH_HEAD
test "$(git rev-parse HEAD)" = "$RWWPO_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

export RWWPO_REPO_DIR="$PWD"
export RWWPO_WORK_ROOT=/data/cw/memagent_work
export GPU_PAIR=6,7
export RWWPO_MANIFEST="$RWWPO_REPO_DIR/manifests/h20/qwen25_7b_rwwpo2_r400_k2_seed2026.json"
export RWWPO_MANIFEST_SHA256="$(sha256sum "$RWWPO_MANIFEST" | awk '{print $1}')"
export RWWPO_SCHEMA="$RWWPO_REPO_DIR/rwwpo2_experiment_manifest.schema.json"
export RWWPO_SCHEMA_SHA256="$(sha256sum "$RWWPO_SCHEMA" | awk '{print $1}')"
export RWWPO_PYTHON="$RWWPO_WORK_ROOT/.venv/bin/python"

export RWWPO_ORIGINAL_RESOLVED_MANIFEST="$RWWPO_WORK_ROOT/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json"
export RWWPO_ORIGINAL_RESOLVED_SHA256="$(sha256sum "$RWWPO_ORIGINAL_RESOLVED_MANIFEST" | awk '{print $1}')"
export RWWPO_S128_DATA="$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet"
export RWWPO_S128_RESOLVED="$RWWPO_WORK_ROOT/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json"
export RWWPO_S128_RESOLVED_SHA256=6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411

test -x "$RWWPO_PYTHON"
test -f "$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct/config.json"
test -f "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet"
test -f "$RWWPO_S128_DATA"
test "$(sha256sum "$RWWPO_S128_DATA" | awk '{print $1}')" = 54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6
test "$(sha256sum "$RWWPO_S128_RESOLVED" | awk '{print $1}')" = "$RWWPO_S128_RESOLVED_SHA256"
```

`GPU_PAIR` may be any two completely idle H20 indices, but it must be explicit,
distinct, and ascending. Every producer obtains the common per-card locks under
`$RWWPO_WORK_ROOT/locks/memagent_h20_gpu_N.lock`, checks both cards twice, and
exits without killing any process if either card is busy.

## 2. Static release checks

```bash
python3 -m py_compile \
  recurrent/research/rwwpo_transaction.py \
  recurrent/research/rwwpo_ledger.py \
  verl/trainer/ppo/core_algos.py \
  verl/workers/actor/dp_actor.py \
  tools/h20/audit_rwwpo2_*.py \
  tools/h20/calibrate_rwwpo2_numeric_oracle.py \
  tools/h20/materialize_rwwpo2_resolved_contract.py \
  tools/h20/preflight_rwwpo2.py \
  tools/h20/run_rwwpo2_e0.py \
  tools/h20/seal_rwwpo2_confirmation_set.py \
  tools/h20/preflight_rwwpo2_confirmation.py \
  tools/h20/materialize_rwwpo2_confirmation_eval.py \
  tools/h20/audit_rwwpo2_confirmation_eval.py \
  tools/h20/finalize_rwwpo2_confirmation.py
bash -n scripts/h20/rwwpo2_common.sh \
  scripts/h20/run_rwwpo2_numeric_oracle.sh \
  scripts/h20/run_qwen25_7b_rwwpo2.sh \
  scripts/h20/run_rwwpo2_confirmation_eval.sh \
  experiments/7b_gate_a/run_gate_a.sh
git diff --check
test -z "$(git status --porcelain)"

"$RWWPO_PYTHON" -m pytest --version
"$RWWPO_PYTHON" -m pytest -q \
  tests/h20/test_rwwpo2_program.py \
  tests/h20/test_rwwpo2_tensor_ledger.py \
  tests/h20/test_rwwpo_core.py \
  tests/h20/test_rwwpo_entrypoints.py \
  tests/h20/test_rwwpo_transaction.py \
  tests/h20/test_tf_rwwpo_budget_leakage.py \
  recurrent/research/tests/test_actor_batch.py
```

The pytest command above is a mandatory related-regression gate before using
GPUs. A missing `pytest` or `torch` installation is a `NO_GO`; a later hand-written
`echo PASS` is not evidence that the suite ran.

## 3. One-time CPU evidence and data-boundary audit

Use a new evidence ID. Its output root is append-only.

```bash
export RWWPO_EVIDENCE_ID=rwwpo2_evidence_${RWWPO_EXPECTED_COMMIT:0:8}_r1
[[ $RWWPO_EVIDENCE_ID =~ ^rwwpo2_evidence_[a-z0-9_-]+$ ]] || {
  echo 'RWWPO2_EVIDENCE_NO_GO:empty or invalid evidence ID' >&2; exit 1;
}
export RWWPO_EVIDENCE_ROOT="$RWWPO_WORK_ROOT/logs/rwwpo2_evidence/$RWWPO_EVIDENCE_ID"
[[ $RWWPO_EVIDENCE_ROOT == "$RWWPO_WORK_ROOT/logs/rwwpo2_evidence/"?* ]] || {
  echo 'RWWPO2_EVIDENCE_NO_GO:evidence root is not a child root' >&2; exit 1;
}
test ! -e "$RWWPO_EVIDENCE_ROOT"
mkdir -p "$RWWPO_EVIDENCE_ROOT"

export RWWPO_E0="$RWWPO_EVIDENCE_ROOT/e0.json"
export RWWPO_DATA_BOUNDARY_AUDIT="$RWWPO_EVIDENCE_ROOT/data_boundary.json"
export RWWPO_BASE_PROTOCOL_AUDIT="$RWWPO_EVIDENCE_ROOT/base_protocol.json"
export RWWPO_SOURCE_FIREWALL="$RWWPO_EVIDENCE_ROOT/source_firewall.json"

"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_source_firewall.py \
  --manifest "$RWWPO_MANIFEST" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_SOURCE_FIREWALL"

"$RWWPO_PYTHON" tools/h20/run_rwwpo2_e0.py \
  --manifest "$RWWPO_MANIFEST" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_E0"

"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_data_boundary.py \
  --manifest "$RWWPO_MANIFEST" \
  --manifest-sha256 "$RWWPO_MANIFEST_SHA256" \
  --train "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  --tokenizer-root "$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --s128-data "$RWWPO_S128_DATA" \
  --s128-resolved "$RWWPO_S128_RESOLVED" \
  --s128-resolved-sha256 "$RWWPO_S128_RESOLVED_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_DATA_BOUNDARY_AUDIT"

"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_base_protocol.py \
  --original-resolved "$RWWPO_ORIGINAL_RESOLVED_MANIFEST" \
  --original-resolved-sha256 "$RWWPO_ORIGINAL_RESOLVED_SHA256" \
  --model-root "$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --train "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_BASE_PROTOCOL_AUDIT"
```

Expected decisions are `RWWPO2_SOURCE_FIREWALL_PASS`, `RWWPO2_E0_PASS`,
`RWWPO2_DATA_BOUNDARY_AUDIT_PASS`, and
`RWWPO2_BASE_PROTOCOL_AUDIT_PASS`. The data report must show exact zero
train–S128 content/root overlap; S128 remains labeled adaptive development.

## 4. Two-H20 numeric oracle and resolved contract

The numeric oracle consumes a separate one-use root and holds both GPU locks
through producer and audit. No training starts here.

```bash
export RWWPO_NUMERIC_ID=rwwpo2_numeric_${RWWPO_EXPECTED_COMMIT:0:8}_r1
export RWWPO_NUMERIC_ORACLE_ROOT="$RWWPO_WORK_ROOT/logs/rwwpo2_numeric/$RWWPO_NUMERIC_ID"
export RWWPO_NUMERIC_AUDIT="$RWWPO_EVIDENCE_ROOT/numeric_oracle_audit.json"
test ! -e "$RWWPO_NUMERIC_ORACLE_ROOT"

bash scripts/h20/run_rwwpo2_numeric_oracle.sh

export RWWPO_NUMERIC_REPORT="$RWWPO_NUMERIC_ORACLE_ROOT/numeric_oracle.json"
export RWWPO_NUMERIC_REPORT_SHA256="$(sha256sum "$RWWPO_NUMERIC_REPORT" | awk '{print $1}')"
"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_numeric_oracle.py \
  --oracle-root "$RWWPO_NUMERIC_ORACLE_ROOT" \
  --oracle-report-sha256 "$RWWPO_NUMERIC_REPORT_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_NUMERIC_AUDIT"

export RWWPO_NUMERIC_AUDIT_SHA256="$(sha256sum "$RWWPO_NUMERIC_AUDIT" | awk '{print $1}')"
export RWWPO_RESOLVED_CONTRACT="$RWWPO_EVIDENCE_ROOT/resolved_contract.json"
"$RWWPO_PYTHON" tools/h20/materialize_rwwpo2_resolved_contract.py \
  --manifest "$RWWPO_MANIFEST" \
  --manifest-sha256 "$RWWPO_MANIFEST_SHA256" \
  --manifest-schema "$RWWPO_SCHEMA" \
  --manifest-schema-sha256 "$RWWPO_SCHEMA_SHA256" \
  --numeric-oracle "$RWWPO_NUMERIC_REPORT" \
  --numeric-oracle-sha256 "$RWWPO_NUMERIC_REPORT_SHA256" \
  --numeric-oracle-audit "$RWWPO_NUMERIC_AUDIT" \
  --numeric-oracle-audit-sha256 "$RWWPO_NUMERIC_AUDIT_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_RESOLVED_CONTRACT"
export RWWPO_RESOLVED_CONTRACT_SHA256="$(sha256sum "$RWWPO_RESOLVED_CONTRACT" | awk '{print $1}')"
```

## 5. One fresh R50 assignment

Use this block once for each `(cell, seed)` in
`A/B/C/D/E × 2026/2027/2028`. Every `RWWPO_RUN_ID` is globally one-use.

```bash
export RWWPO_CELL=B
export RWWPO_EXPERIMENT_SEED=2026
export RWWPO_TARGET_ROUND=50
export RWWPO_PHASE=fresh
export RWWPO_RUN_ID="rwwpo2_r50_${RWWPO_CELL,,}_seed${RWWPO_EXPERIMENT_SEED}_${RWWPO_EXPECTED_COMMIT:0:8}_r1"
unset RWWPO_RESUME_ROUND RWWPO_LINEAGE_PARENT_RECEIPT RWWPO_PARENT_OUTPUT_ROOT

test ! -e "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID"
nvidia-smi -i "$GPU_PAIR" \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader

screen -L \
  -Logfile "$RWWPO_WORK_ROOT/logs/rwwpo2_${RWWPO_CELL}_${RWWPO_EXPERIMENT_SEED}.screen.log" \
  -dmS "rwwpo2-${RWWPO_CELL}-${RWWPO_EXPERIMENT_SEED}" \
  bash scripts/h20/run_qwen25_7b_rwwpo2.sh
screen -ls
```

The healthy endpoint contains 100 rank-paired transaction receipts, full
recovery checkpoints at 10/20/30/40/50 with only 40 and 50 retained, and
immutable actor anchors at 5/10/15/20/25/50. Training never invokes validation.

## 6. Attempt audit and failure recovery

After a screen exits, reconstruct the deterministic output root and audit it.

```bash
source scripts/h20/rwwpo2_common.sh
export RWWPO_ATTEMPT_AUDIT="$RWWPO_CERT_ROOT/attempt_audit_r50.json"
"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_attempt.py \
  --attempt-root "$RWWPO_ATTEMPT_ROOT" \
  --output-root "$RWWPO_OUTPUT" \
  --resolved-contract "$RWWPO_RESOLVED_CONTRACT" \
  --resolved-contract-sha256 "$RWWPO_RESOLVED_CONTRACT_SHA256" \
  --preflight "$RWWPO_PREFLIGHT" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --cell "$RWWPO_CELL" \
  --experiment-seed "$RWWPO_EXPERIMENT_SEED" \
  --target-round 50 \
  --output "$RWWPO_ATTEMPT_AUDIT"
```

If a run fails after a fully audited round-10 multiple, preserve the failed
root. Create a signed parent receipt from the last usable recovery checkpoint:

```bash
export RWWPO_RESUME_ROUND=20
export RWWPO_PARENT_SEGMENT_AUDIT="$RWWPO_CERT_ROOT/attempt_audit_r${RWWPO_RESUME_ROUND}.json"
"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_attempt.py \
  --attempt-root "$RWWPO_ATTEMPT_ROOT" \
  --output-root "$RWWPO_OUTPUT" \
  --resolved-contract "$RWWPO_RESOLVED_CONTRACT" \
  --resolved-contract-sha256 "$RWWPO_RESOLVED_CONTRACT_SHA256" \
  --preflight "$RWWPO_PREFLIGHT" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --cell "$RWWPO_CELL" \
  --experiment-seed "$RWWPO_EXPERIMENT_SEED" \
  --target-round "$RWWPO_RESUME_ROUND" \
  --output "$RWWPO_PARENT_SEGMENT_AUDIT"

export RWWPO_LINEAGE_PARENT_RECEIPT="$RWWPO_CERT_ROOT/lineage_parent_r${RWWPO_RESUME_ROUND}.json"
"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_lineage_parent.py \
  --parent-attempt-root "$RWWPO_ATTEMPT_ROOT" \
  --parent-output-root "$RWWPO_OUTPUT" \
  --checkpoint-round "$RWWPO_RESUME_ROUND" \
  --cell "$RWWPO_CELL" \
  --experiment-seed "$RWWPO_EXPERIMENT_SEED" \
  --resolved-contract "$RWWPO_RESOLVED_CONTRACT" \
  --resolved-contract-sha256 "$RWWPO_RESOLVED_CONTRACT_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_LINEAGE_PARENT_RECEIPT"

export RWWPO_PARENT_OUTPUT_ROOT="$RWWPO_OUTPUT"
export RWWPO_PHASE=resume
export RWWPO_RUN_ID="rwwpo2_r50_${RWWPO_CELL,,}_seed${RWWPO_EXPERIMENT_SEED}_${RWWPO_EXPECTED_COMMIT:0:8}_resume_r${RWWPO_RESUME_ROUND}_r1"
bash scripts/h20/run_qwen25_7b_rwwpo2.sh
```

The new attempt reproduces logical proposal clocks and seeds from round 21. Its
attempt ID changes only the evidence path. The authenticated parent prefix joins
the canonical DAG; the failed suffix does not. After the resumed attempt reaches
round 50, audit the resumed root as `attempt_audit_r50.json`. Both the parent's
`attempt_audit_r20.json` and the resumed segment's `attempt_audit_r50.json` must
be supplied to the R50 finalizer.

## 7. Complete R50 mechanism gate

After all 15 cell/seed assignments have complete attempt-audit receipts, pass
every receipt (including multiple lineage segments, if any) to the finalizer:

```bash
R50_ARGS=(
  --attempt-audit /absolute/path/to/A_seed2026/attempt_audit_r50.json
  --attempt-audit /absolute/path/to/A_seed2027/attempt_audit_r50.json
  # ...list every authenticated segment for all 15 assignments explicitly...
  # A resumed assignment contributes both its parent-prefix and resume receipts.
  --attempt-audit /absolute/path/to/resumed_parent/attempt_audit_r20.json
  --attempt-audit /absolute/path/to/resumed_suffix/attempt_audit_r50.json
)

export RWWPO_R50_PROGRAM_GATE="$RWWPO_EVIDENCE_ROOT/r50_program_gate.json"
"$RWWPO_PYTHON" tools/h20/audit_rwwpo2_r50_program.py \
  "${R50_ARGS[@]}" \
  --resolved-contract "$RWWPO_RESOLVED_CONTRACT" \
  --resolved-contract-sha256 "$RWWPO_RESOLVED_CONTRACT_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_R50_PROGRAM_GATE"
```

Do not use an unfiltered `find` command: the work root may contain failed,
superseded, or another RWWPO-2 matrix. The explicit receipt list is part of the
scientific decision. `PASS` means only that
the K2 geometry was exposed, numerically separated, controller support was
stable under prompt-root deletion, shared-proposal aperture existed, and all
distributed/recovery evidence closed. It does not mean the method improved
task performance.

## 8. R400 and confirmation boundary

Only after the R50 program gate passes, bind a previously untouched parquet.
The sealing entry emits no row content, but it freezes every row identity and
must reject any filter loss, duplicate root/content identity, or overlap.

```bash
export RWWPO_CONFIRM_DATA=/absolute/path/to/untouched_confirmation.parquet
export RWWPO_CONFIRM_DATA_SHA256="$(sha256sum "$RWWPO_CONFIRM_DATA" | awk '{print $1}')"
export RWWPO_CONFIRM_ROOT="$RWWPO_WORK_ROOT/logs/rwwpo2_confirmation/seal_${RWWPO_EXPECTED_COMMIT:0:8}_r1"
export RWWPO_CONFIRM_BASE_RESOLVED="$RWWPO_CONFIRM_ROOT/resolved_identity.json"
export RWWPO_CONFIRMATION_SEAL="$RWWPO_CONFIRM_ROOT/seal.json"
test ! -e "$RWWPO_CONFIRM_ROOT"
mkdir -p "$RWWPO_CONFIRM_ROOT"

"$RWWPO_PYTHON" tools/h20/seal_rwwpo2_confirmation_set.py \
  --manifest "$RWWPO_MANIFEST" --manifest-sha256 "$RWWPO_MANIFEST_SHA256" \
  --train "$RWWPO_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  --tokenizer-root "$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --s128-resolved "$RWWPO_S128_RESOLVED" \
  --s128-resolved-sha256 "$RWWPO_S128_RESOLVED_SHA256" \
  --confirmation-data "$RWWPO_CONFIRM_DATA" \
  --confirmation-data-sha256 "$RWWPO_CONFIRM_DATA_SHA256" \
  --seal-id "rwwpo2-confirm-${RWWPO_EXPECTED_COMMIT:0:8}-r1" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --resolved-output "$RWWPO_CONFIRM_BASE_RESOLVED" \
  --output "$RWWPO_CONFIRMATION_SEAL"

export RWWPO_CONFIRM_BASE_RESOLVED_SHA256="$(sha256sum "$RWWPO_CONFIRM_BASE_RESOLVED" | awk '{print $1}')"
export RWWPO_CONFIRMATION_SEAL_SHA256="$(sha256sum "$RWWPO_CONFIRMATION_SEAL" | awk '{print $1}')"
```

Then fresh-base train B/D/E for all seeds 2026–2033 and C for the explanatory
seeds 2026–2028. Each assignment uses the same launch block as Section 5 with
`RWWPO_TARGET_ROUND=400`, a new one-use run ID, and these two added bindings:

```bash
export RWWPO_R50_PROGRAM_GATE=/absolute/path/to/r50_program_gate.json
export RWWPO_R50_PROGRAM_GATE_SHA256="$(sha256sum "$RWWPO_R50_PROGRAM_GATE" | awk '{print $1}')"
export RWWPO_CONFIRMATION_SEAL
export RWWPO_CONFIRMATION_SEAL_SHA256
```

Do not resume an R50 model into R400: every R400 assignment starts from the same
fresh Qwen2.5-7B base. A hardware failure may resume only from that assignment's
own authenticated newest round-10 recovery checkpoint. Audit every complete
R400 lineage using `audit_rwwpo2_attempt.py --target-round 400`. Freeze an
explicit TSV with exactly 24 confirmatory assignments before opening outcomes:

```text
# cell<TAB>seed<TAB>attempt_audit_json<TAB>global_step_400
B 2026 /absolute/.../attempt_audit_r400.json /absolute/.../global_step_400
# ...exactly B/D/E x seeds 2026..2033; no C/A rows...
```

Evaluate those 24 entries sequentially on the same explicit idle GPU pair.
The following loop assumes the four fields above are tab/space separated and
that `RWWPO_CONFIRM_REGISTRY` has been made read-only after review:

```bash
export RWWPO_CONFIRM_REGISTRY=/absolute/path/to/frozen_rwwpo2_confirm_registry.tsv
export RWWPO_CONFIRM_EVAL_PARENT="$RWWPO_WORK_ROOT/logs/rwwpo2_confirmation/evals_${RWWPO_EXPECTED_COMMIT:0:8}_r1"
export RWWPO_CONFIRM_CERT_PARENT="$RWWPO_WORK_ROOT/logs/rwwpo2_confirmation/certs_${RWWPO_EXPECTED_COMMIT:0:8}_r1"
test ! -e "$RWWPO_CONFIRM_EVAL_PARENT"
test ! -e "$RWWPO_CONFIRM_CERT_PARENT"
mkdir -p "$RWWPO_CONFIRM_EVAL_PARENT" "$RWWPO_CONFIRM_CERT_PARENT"

while read -r CELL SEED ATTEMPT_AUDIT CHECKPOINT; do
  [[ -n $CELL && $CELL != \#* ]] || continue
  export RWWPO_CONFIRM_INTERFACE_ID="RWWPO2_${CELL}_seed${SEED}_R400"
  export RWWPO_CONFIRM_ATTEMPT_ID="rwwpo2-confirm-eval-${CELL,,}-seed${SEED}-${RWWPO_EXPECTED_COMMIT:0:8}-r1"
  export RWWPO_CONFIRM_EVAL_ROOT="$RWWPO_CONFIRM_EVAL_PARENT/${CELL}_seed${SEED}"
  export RWWPO_CONFIRM_RESOLVED="$RWWPO_CONFIRM_CERT_PARENT/${CELL}_seed${SEED}_resolved.json"
  export RWWPO_CONFIRM_CHECKPOINT="$CHECKPOINT"
  export RWWPO_CONFIRM_MODEL="$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct"

  "$RWWPO_PYTHON" tools/h20/materialize_rwwpo2_confirmation_eval.py \
    --seal "$RWWPO_CONFIRMATION_SEAL" \
    --seal-sha256 "$RWWPO_CONFIRMATION_SEAL_SHA256" \
    --base-resolved "$RWWPO_CONFIRM_BASE_RESOLVED" \
    --base-resolved-sha256 "$RWWPO_CONFIRM_BASE_RESOLVED_SHA256" \
    --attempt-audit "$ATTEMPT_AUDIT" --checkpoint "$CHECKPOINT" \
    --validation "$RWWPO_CONFIRM_DATA" --model "$RWWPO_CONFIRM_MODEL" \
    --eval-root "$RWWPO_CONFIRM_EVAL_ROOT" \
    --interface-id "$RWWPO_CONFIRM_INTERFACE_ID" \
    --attempt-id "$RWWPO_CONFIRM_ATTEMPT_ID" \
    --cell "$CELL" --experiment-seed "$SEED" \
    --expected-commit "$RWWPO_EXPECTED_COMMIT" \
    --output "$RWWPO_CONFIRM_RESOLVED"
  export RWWPO_CONFIRM_RESOLVED_SHA256="$(sha256sum "$RWWPO_CONFIRM_RESOLVED" | awk '{print $1}')"

  bash scripts/h20/run_rwwpo2_confirmation_eval.sh

  "$RWWPO_PYTHON" tools/h20/audit_rwwpo2_confirmation_eval.py \
    --eval-root "$RWWPO_CONFIRM_EVAL_ROOT" \
    --resolved-manifest "$RWWPO_CONFIRM_RESOLVED" \
    --resolved-manifest-sha256 "$RWWPO_CONFIRM_RESOLVED_SHA256" \
    --validation "$RWWPO_CONFIRM_DATA" --checkpoint "$CHECKPOINT" \
    --model "$RWWPO_CONFIRM_MODEL" \
    --interface-id "$RWWPO_CONFIRM_INTERFACE_ID" \
    --attempt-id "$RWWPO_CONFIRM_ATTEMPT_ID" \
    --expected-commit "$RWWPO_EXPECTED_COMMIT" \
    --metric-rows-output "$RWWPO_CONFIRM_CERT_PARENT/${CELL}_seed${SEED}_metric_rows.jsonl" \
    --output "$RWWPO_CONFIRM_CERT_PARENT/${CELL}_seed${SEED}_audit.json"
done < "$RWWPO_CONFIRM_REGISTRY"
```

Finally pass all 24 audit receipts explicitly; do not use a broad filesystem
search. The finalizer independently reopens all raw terminal outputs and the
sealed parquet, reconstructs metric rows/aggregates, requires one generation
protocol SHA and stable-key inventory, then performs the two margin-centered
exact tests and Holm correction:

```bash
CONFIRM_ARGS=(
  --evaluation-report "$RWWPO_CONFIRM_CERT_PARENT/B_seed2026_audit.json"
  # ...all and only B/D/E x 2026..2033...
)
"$RWWPO_PYTHON" tools/h20/finalize_rwwpo2_confirmation.py \
  "${CONFIRM_ARGS[@]}" \
  --seal "$RWWPO_CONFIRMATION_SEAL" \
  --seal-sha256 "$RWWPO_CONFIRMATION_SEAL_SHA256" \
  --base-resolved "$RWWPO_CONFIRM_BASE_RESOLVED" \
  --base-resolved-sha256 "$RWWPO_CONFIRM_BASE_RESOLVED_SHA256" \
  --validation "$RWWPO_CONFIRM_DATA" --manifest "$RWWPO_MANIFEST" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output "$RWWPO_CONFIRM_CERT_PARENT/final_confirmation.json"
```

Both co-primary tests must exceed their frozen +0.02/+0.01 margins and pass
Holm family-wise correction. A scientifically valid negative result writes a
`NO_GO` final report and exits nonzero; it is never converted to PASS by EM,
precision/recall, format, or historical SubEM.

R400 is a medium-budget experiment, not evidence of convergence. S128 remains
adaptive development and is never a blind final test.

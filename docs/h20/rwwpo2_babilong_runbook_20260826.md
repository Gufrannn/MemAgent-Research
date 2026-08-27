# RWWPO-2 BABILong evaluation runbook (frozen 2026-08-26)

## Scientific scope

This program evaluates whether the RWWPO-2 variants improve long-context state
tracking as reasoning depth and context length increase.  It does not train on
BABILong and it does not reinterpret the R50 mechanism gate as a performance
result.

- Official source: `RMT-team/babilong` at exact revision
  `e3a924b6686759422257925a695cbbb4b2684936`.
- Length labels: `32k` and `128k`.
- Registered reasoning-depth proxy: `qa1=1`, `qa2=2`, `qa3=3`.
- Development membership: eight SHA-ranked rows per length/task cell, 48 rows
  total.  It is adaptive and descriptive.
- Confirmation membership: the complementary 92 rows per cell, 552 rows total.
  It is procedurally forbidden from model evaluation or metric inspection until
  R400 training is complete.  BABILong is a public dataset, so this is a
  procedural holdout rather than a secret test set.
- Primary metric: official case-insensitive target-substring accuracy.
- Key secondary: strict normalized exact match from the final box.
- Secondary: macro token F1, precision, and recall.
- Safety: boxed-format success.  Format is not a performance claim.
- Contexts are never silently truncated.  The executable capacity is 40,000
  Qwen tokens for 32k (`5000 x 8`) and 160,000 for 128k (`5000 x 32`).

The official target sentence is unchanged.  The only prompt adaptation is the
MemAgent final-answer wrapper `\\boxed{<answer sentence>}`.  QA depth is a
registered reasoning-depth proxy, not direct causal credit-location evidence.

## Frozen execution order

1. Exact-commit source firewall and authenticated related-regression suite.
2. Six manually constructed fixtures: 32k/128k x qa1/qa2/qa3.
3. Pinned official source bundle, development materialization, and independent
   bundle reconstruction.  Confirmation membership is frozen in code but is not
   materialized/evaluated yet.
4. B-R20 then D-R20 on the identical 48 development examples, followed by an
   exact source-identity paired descriptive difference.
5. E-R20 is added only after its checkpoint appears; compare B-E under the same
   protocol.
6. B/D/E R50 endpoints are evaluated on the same development rows only after
   each endpoint has a PASS training-attempt audit.
7. B/D/E R400 endpoints are evaluated on the 552-row confirmation complement
   only after each endpoint has a PASS training-attempt audit.  The confirmation
   partition must not be used for checkpoint selection or method changes.

## One-shot steps 1-4 entry

Run only from a clean checkout of the exact adapter commit.  Every ID below is
one-use, including a failed attempt.

```bash
export RWWPO_WORK_ROOT=/data/cw/memagent_work
export RWWPO_REPO_DIR=/home/test001/memagent-rwwpo-release
export RWWPO_EXPECTED_COMMIT=<EXACT_ADAPTER_COMMIT>
export RWWPO_TRAINING_COMMIT=4a6a72ef51aa9e8bba2b9c2efe22dc4c98b54dfe
export GPU_PAIR=0,1
export RWWPO_BABILONG_RUN_ID=rwwpo2_babilong_<SHORT_COMMIT>_bd_r20_r1
export RWWPO_BABILONG_MODEL="$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct"
export RWWPO_BABILONG_B_CHECKPOINT="$RWWPO_WORK_ROOT/logs/memory_agent/rwwpo2_B_seed2026_rwwpo2_r50_b_seed2026_4a6a72ef_r3/global_step_20"
export RWWPO_BABILONG_D_CHECKPOINT="$RWWPO_WORK_ROOT/logs/memory_agent/rwwpo2_D_seed2026_rwwpo2_r50_d_seed2026_4a6a72ef_r1/global_step_20"

cd "$RWWPO_REPO_DIR"
test "$(git rev-parse HEAD)" = "$RWWPO_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

screen -L \
  -Logfile "$RWWPO_WORK_ROOT/logs/${RWWPO_BABILONG_RUN_ID}.screen.log" \
  -dmS "$RWWPO_BABILONG_RUN_ID" \
  bash scripts/h20/run_rwwpo2_babilong_prepare_and_bd_t20.sh
```

Monitor without changing the run:

```bash
tail -F "$RWWPO_WORK_ROOT/logs/${RWWPO_BABILONG_RUN_ID}.screen.log"
```

The terminal success marker is:

```text
RWWPO2 BABILONG PREPARE + B/D R20 PASS
```

The paired report is under:

```text
$RWWPO_WORK_ROOT/logs/rwwpo2_babilong/$RWWPO_BABILONG_RUN_ID/bd_r20/certificates/B_minus_D_r20.json
```

## E-R20 entry after the checkpoint exists

Do not run this block until the operator supplies the exact E checkpoint.  Reuse
the already audited development bundle and release-test receipt from steps 1-4.

```bash
export RWWPO_BABILONG_CELL=E
export RWWPO_BABILONG_STEP=20
export RWWPO_BABILONG_PARTITION=development
export RWWPO_BABILONG_EXPERIMENT_SEED=2026
export RWWPO_BABILONG_CHECKPOINT=<ABSOLUTE_E_GLOBAL_STEP_20>
export RWWPO_BABILONG_PIPELINE_ROOT=<NEW_ONE_USE_ABSOLUTE_E_R20_ROOT>

bash "$RWWPO_REPO_DIR/scripts/h20/run_rwwpo2_babilong_cell.sh"
```

## R50 and R400 safety rule

For R50/R400 the generic entry additionally requires:

```bash
export RWWPO_BABILONG_TRAINING_ATTEMPT_AUDIT=<ABSOLUTE_PASS_ATTEMPT_AUDIT>
export RWWPO_BABILONG_TRAINING_ATTEMPT_AUDIT_SHA256="$(sha256sum \
  "$RWWPO_BABILONG_TRAINING_ATTEMPT_AUDIT" | awk '{print $1}')"
```

The entry rejects a checkpoint whose cell, seed, round, inventory, output root,
or training commit differs from that audit.  R20 is explicitly labelled an
unaudited low-budget diagnostic checkpoint; it cannot be promoted to formal
evidence by renaming its output.

## H20 attempt history

The first `fec051a4...` development entry
`rwwpo2_babilong_fec051a4_bd_r20_r1` stopped before release tests, source
materialization, or GPU work.  The preparation shell treated the normal virtual
environment launcher `.venv/bin/python` as an invalid symlink even though the
authenticated release-test verifier deliberately resolves and fingerprints the
interpreter target.  Its paired E watcher therefore exited without evaluation.
Both roots are consumed `NO_GO` evidence and must not be reused.  The correction
accepts only an executable interpreter whose fully resolved target is a regular
file; immutable manifests, model metadata, training data, and checkpoint files
remain non-symlink inputs.

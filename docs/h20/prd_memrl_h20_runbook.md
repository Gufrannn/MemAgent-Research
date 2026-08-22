# PRD-MemRL H20 fail-closed runbook

The production training cases intentionally remain `RELEASE_NO_GO` until the learned-prior worker,
dual update/checkpoint path, distributed weight synchronization, and independent reviews are complete.
The commands below exercise and bind the surrounding protocol without claiming an H20 run occurred.

Required environment: `WORK_ROOT` (absolute), `EXPECTED_COMMIT` (full reviewed SHA), `RUN_ID`
(unique), and `GPU_PAIR` (two distinct ascending physical GPU indices such as `2,7`). The lock files are
`$WORK_ROOT/locks/memagent_h20_gpu_N.lock`. A conflict or an occupied device returns NO-GO and never
kills a process.

## Certified Original read-only sources

Do not rerun Original and do not derive rows from an aggregate. The curve source is
`/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821/certificates/original_s128_curve_final_report.json`;
artifact paths and SHA-256 values must be read from that report's certified inventory. Bind protocol
equality to `/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json`
(compute its SHA-256 read-only on H20; never guess it). Bind S128 identity to
`/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json`,
whose SHA-256 is `6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411` and
`eval_manifest_hash` is `351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a`.
There is no certified standalone `baseline_bundle.json`: materialize the import certificate from the
curve report's per-interface artifacts, verify every file SHA, and independently recompute normalized
EM/token-F1/format before `bind`. `actual_loss_rank0/1.jsonl` is absent, so actual-loss analyses remain
`PENDING_ACTUAL_LOSS_LEDGER`; aggregate metrics and dense rewards are not loss substitutes.

```bash
export WORK_ROOT=/absolute/h20/work
export EXPECTED_COMMIT=<reviewed-40-character-sha>
export RUN_ID=prd-memrl-seed2026-<unique-suffix>
export GPU_PAIR=2,7
export PRD_PYTHON=$WORK_ROOT/.venv/bin/python

export ORIGINAL_CURVE_REPORT=/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821/certificates/original_s128_curve_final_report.json
export ORIGINAL_TRAINING_REPORT=/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/original_t25_final_report.json
export ORIGINAL_TRAINING_RESOLVED=/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json
export STABLE_S128_RESOLVED=/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json
sha256sum "$ORIGINAL_TRAINING_RESOLVED"
test "$(sha256sum "$STABLE_S128_RESOLVED" | cut -d' ' -f1)" = 6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411

scripts/h20/run_qwen25_7b_prd_memrl.sh e0
E1_ROWS=/absolute/frozen/e1_rows.jsonl scripts/h20/run_qwen25_7b_prd_memrl.sh e1
# Place the independent PRD_PAPER_REVIEW_GO certificate in $WORK_ROOT/logs/prd_memrl/$RUN_ID/certificates/paper_review.json.
scripts/h20/run_qwen25_7b_prd_memrl.sh preflight
export BASELINE_CERT=$WORK_ROOT/logs/prd_memrl/$RUN_ID/certificates/original_import.json
$PRD_PYTHON tools/h20/import_prd_original_baseline.py \
  --curve-report "$ORIGINAL_CURVE_REPORT" \
  --stable-resolved "$STABLE_S128_RESOLVED" \
  --validation-parquet /absolute/path/to/the/frozen/hotpotqa_dev.parquet \
  --original-training-final-report "$ORIGINAL_TRAINING_REPORT" \
  --original-training-resolved "$ORIGINAL_TRAINING_RESOLVED" \
  --output "$BASELINE_CERT"
scripts/h20/run_qwen25_7b_prd_memrl.sh bind

for capacity in 128.0 256.0 512.0; do
  CAPACITY_NATS=$capacity scripts/h20/run_qwen25_7b_prd_memrl.sh prepare-run
  CAPACITY_NATS=$capacity scripts/h20/run_qwen25_7b_prd_memrl.sh train-t25
done
```

Each capacity is one fresh-base process through step 25, saving 5/10/15/20/25. At step 5 the trainer
performs only the cheap numerical/checkpoint/ledger/weight-sync health check and immediately continues;
it does not run S128 or wait for Original analysis. After T25, generate fixed-S128 terminal rows for all
five Method checkpoints and call `evaluate` once per capacity with `EVAL_ANCHORS=5,10,15,20,25`, then
run `final-audit`. `prepare-continuation`/`recover-from-t5` is disaster recovery only and requires that
capacity's authenticated T5 health certificate and exact checkpoint. Failed runs remain under their
unique `RUN_ID`; a scientific revision requires a new variant identity.

# PRD-MemRL H20 fail-closed runbook

The production training cases intentionally remain `RELEASE_NO_GO` until the learned-prior worker,
dual update/checkpoint path, distributed weight synchronization, and independent reviews are complete.
The commands below exercise and bind the surrounding protocol without claiming an H20 run occurred.

Required environment: `WORK_ROOT` (absolute), `EXPECTED_COMMIT` (full reviewed SHA), `RUN_ID`
(unique), and `GPU_PAIR` (two distinct ascending physical GPU indices such as `2,7`). The lock files are
`$WORK_ROOT/locks/memagent_h20_gpu_N.lock`. A conflict or an occupied device returns NO-GO and never
kills a process.

```bash
export WORK_ROOT=/absolute/h20/work
export EXPECTED_COMMIT=<reviewed-40-character-sha>
export RUN_ID=prd-memrl-seed2026-<unique-suffix>
export GPU_PAIR=2,7
export PRD_PYTHON=$WORK_ROOT/.venv/bin/python

scripts/h20/run_qwen25_7b_prd_memrl.sh e0
E1_ROWS=/absolute/frozen/e1_rows.jsonl scripts/h20/run_qwen25_7b_prd_memrl.sh e1
# Place the independent PRD_PAPER_REVIEW_GO certificate in $WORK_ROOT/logs/prd_memrl/$RUN_ID/certificates/paper_review.json.
scripts/h20/run_qwen25_7b_prd_memrl.sh preflight
BASELINE_CERT=/absolute/readonly/original_import.json scripts/h20/run_qwen25_7b_prd_memrl.sh bind

for capacity in 128.0 256.0 512.0; do
  CAPACITY_NATS=$capacity scripts/h20/run_qwen25_7b_prd_memrl.sh prepare-t5
done
```

After a reviewed production trainer creates all three complete step-5 checkpoints and raw fixed-S128
terminal rows, run `evaluate` for each capacity with `EVAL_ANCHORS=5` (the evaluator independently
recomputes EM/token-F1/format), then `t5-gate`. Only a three-capacity PASS
permits `prepare-continuation`; each continuation must name its own exact
`frontier/cN/checkpoints/global_step_5` and reach anchors 10, 15, 20, and 25 without overwriting T5.
After reevaluation of every anchor, run `final-audit`. Failed runs and certificates are retained under
their unique `RUN_ID`; revisions require a new run/variant identity.

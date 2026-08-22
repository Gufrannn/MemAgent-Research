# PRD-MemRL H20 fail-closed runbook

This is a code-release runbook, not evidence that H20 execution passed. `GPU_PAIR` remains a required
dynamic, canonical ascending pair chosen after inspecting the machine. Locks are
`/data/cw/memagent_work/locks/memagent_h20_gpu_N.lock`; contention is NO-GO and never kills a process.

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
export WORK_ROOT=/data/cw/memagent_work
export REPO=/data/cw/memagent_work/code/MemAgent-Research-prd-memrl
export BRANCH=h20/qwen25-7b-prd-memrl-t25-frozen-20260822
export GPU_PAIR=${GPU_PAIR:?set an idle canonical ascending pair such as N,M}
export RUN_ID=prd-memrl-seed2026-$(date -u +%Y%m%dT%H%M%SZ)
export PRD_PYTHON=/data/cw/memagent_work/.venv/bin/python
export PRD_BASE_MODEL=/data/cw/memagent_work/models/Qwen2.5-7B-Instruct
export PRD_PRIOR_MODEL=/data/cw/memagent_work/models/Qwen2.5-0.5B-Instruct
export TRAIN_PARQUET=/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_train_32k.parquet
export VALIDATION_PARQUET=/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_dev.parquet
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p /data/cw/memagent_work/code
if test -e "$REPO/.git"; then git -C "$REPO" fetch origin "$BRANCH"; else git clone --branch "$BRANCH" --single-branch https://github.com/Gufrannn/MemAgent-Research.git "$REPO"; fi
cd "$REPO"; git switch "$BRANCH"; git pull --ff-only origin "$BRANCH"
export EXPECTED_COMMIT=$(git rev-parse "origin/$BRANCH")
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"; test -z "$(git status --porcelain)"

export ORIGINAL_CURVE_REPORT=/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821/certificates/original_s128_curve_final_report.json
export ORIGINAL_TRAINING_REPORT=/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/original_t25_final_report.json
export ORIGINAL_TRAINING_RESOLVED=/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json
export STABLE_S128_RESOLVED=/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json
sha256sum "$ORIGINAL_TRAINING_RESOLVED"
test "$(sha256sum "$STABLE_S128_RESOLVED" | cut -d' ' -f1)" = 6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411
test "$(sha256sum "$VALIDATION_PARQUET" | cut -d' ' -f1)" = 54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6

if test -f "$PRD_PRIOR_MODEL/config.json"; then
  $PRD_PYTHON tools/h20/materialize_prd_prior.py --target "$PRD_PRIOR_MODEL" --certificate "$WORK_ROOT/logs/prd_memrl/$RUN_ID/certificates/prior_materialization.json"
else
  $PRD_PYTHON tools/h20/materialize_prd_prior.py --target "$PRD_PRIOR_MODEL" --certificate "$WORK_ROOT/logs/prd_memrl/$RUN_ID/certificates/prior_materialization.json" --download
fi

scripts/h20/run_qwen25_7b_prd_memrl.sh e0
$PRD_PYTHON tools/h20/issue_prd_paper_review.py --expected-commit "$EXPECTED_COMMIT" \
  --output "$WORK_ROOT/logs/prd_memrl/$RUN_ID/certificates/paper_review.json"
scripts/h20/run_qwen25_7b_prd_memrl.sh preflight
export BASELINE_CERT=$WORK_ROOT/logs/prd_memrl/$RUN_ID/certificates/original_import.json
$PRD_PYTHON tools/h20/import_prd_original_baseline.py \
  --curve-report "$ORIGINAL_CURVE_REPORT" \
  --stable-resolved "$STABLE_S128_RESOLVED" \
  --validation-parquet "$VALIDATION_PARQUET" \
  --original-training-final-report "$ORIGINAL_TRAINING_REPORT" \
  --original-training-resolved "$ORIGINAL_TRAINING_RESOLVED" \
  --output "$BASELINE_CERT"
scripts/h20/run_qwen25_7b_prd_memrl.sh bind

for capacity in 128.0 256.0 512.0; do
  CAPACITY_NATS=$capacity scripts/h20/run_qwen25_7b_prd_memrl.sh prepare-run
  CAPACITY_NATS=$capacity scripts/h20/run_qwen25_7b_prd_memrl.sh train-t25
done

for capacity in 128.0 256.0 512.0; do
  for anchor in 5 10 15 20 25; do
    CAPACITY_NATS=$capacity ANCHOR=$anchor scripts/h20/run_qwen25_7b_prd_memrl.sh produce-s128
  done
  cid=c${capacity%.*}
  CAPACITY_NATS=$capacity EVAL_ANCHORS=5,10,15,20,25 \
    EVAL_INPUT_TEMPLATE="$WORK_ROOT/logs/prd_memrl/$RUN_ID/frontier/$cid/raw_terminal/anchor_{anchor}.jsonl" \
    scripts/h20/run_qwen25_7b_prd_memrl.sh evaluate
done
scripts/h20/run_qwen25_7b_prd_memrl.sh final-audit
```

Each capacity is one fresh-base process through step 25, saving 5/10/15/20/25. At step 5 the trainer
performs only the cheap numerical/checkpoint/ledger/weight-sync health check and immediately continues;
it does not run S128 or wait for Original analysis. After T25, generate fixed-S128 terminal rows for all
five Method checkpoints and call `evaluate` once per capacity. E1 is a posthoc on-policy Method
diagnostic and does not consume nonexistent Original log-probabilities or block training. `prepare-continuation`/`recover-from-t5` is disaster recovery only and requires that
capacity's authenticated T5 health certificate and exact checkpoint. Failed runs remain under their
unique `RUN_ID`; a scientific revision requires a new variant identity.

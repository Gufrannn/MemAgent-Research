# HDR-MemRL H20 runbook (fail closed)

This branch does not claim any H20, E0, E1, or Method result. Original is never
rerun. Training is one fresh-base run through T25 with checkpoints at
T5/T10/T15/T20/T25. Fixed-S128 evaluation and Original comparison happen only
after training completes.

## Frozen authorities

- Original curve report: `/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821/certificates/original_s128_curve_final_report.json`
- Original curve ledger: `/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821/original_s128_curve_execution_ledger.jsonl`
- Accepted Original training resolved manifest: `/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json`, SHA256 `814fc1219530f013d4c82e5edd67bc04be7b62699936da9598303911ab7fc220`
- Stable-S128 resolved manifest: `/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json`, SHA256 `6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411`

All authority inputs are read-only. The importer discovers terminal artifacts
only through the final report inventory, verifies each file SHA, rebuilds all
128 metric rows from terminal text and frozen parquet ground truth, and checks
the six canonical row digests.

## Copyable chain

Replace only `RELEASE_SHA`, `GPU_PAIR`, `HDR_RUN_ID`, and the reviewer-attestation
SHA. `GPU_PAIR` must contain two idle H20 IDs in ascending order.

```bash
set -euo pipefail
cd /data/cw/memagent_work/MemAgent-Research
git fetch origin h20/qwen25-7b-hdr-memrl-t25-frozen-20260822
git checkout h20/qwen25-7b-hdr-memrl-t25-frozen-20260822
git pull --ff-only origin h20/qwen25-7b-hdr-memrl-t25-frozen-20260822

export MEMAGENT_HDR_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_HDR_EXPECTED_COMMIT=<RELEASE_SHA>
export GPU_PAIR=2,5
export HDR_RUN_ID=hdr-seed2026-v1
export MEMAGENT_HDR_REVIEW_SHA256=<REVIEW_ATTESTATION_SHA256>
export HDR_REPO=$PWD
export HDR_ROOT=$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID
export HDR_CERT=$HDR_ROOT/certificates
export HDR_LEDGER=$HDR_ROOT/hdr_execution_ledger.jsonl
export HDR_OUTPUT=$MEMAGENT_HDR_WORK_ROOT/logs/memory_agent/qwen25_7b_hdr_memrl_dro_seed2026_$HDR_RUN_ID
mkdir -p "$HDR_CERT"

test "$(git rev-parse HEAD)" = "$MEMAGENT_HDR_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test "$(sha256sum /data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json | awk '{print $1}')" = 814fc1219530f013d4c82e5edd67bc04be7b62699936da9598303911ab7fc220

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py preflight \
  --manifest manifests/h20/qwen25_7b_hdr_memrl_seed2026.json \
  --accepted-manifest /data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json \
  --output "$HDR_CERT/p0.json" --ledger "$HDR_LEDGER"

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/prepare_hdr_horizon_suite.py \
  --input "$MEMAGENT_HDR_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  --input-sha256 54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6 \
  --model "$MEMAGENT_HDR_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --horizons 8 12 16 32 --count 128 \
  --output-parquet "$HDR_ROOT/e0/horizon_suite.parquet" \
  --receipts "$HDR_ROOT/e0/receipts.json" --roots "$HDR_ROOT/e0/eval_roots.json"

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/prepare_hdr_root_ids.py \
  --input "$MEMAGENT_HDR_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  --input-sha256 798b7a2a9ece4f40884e2a9d02d165d7352df7763d1569ceaf402b45f76896f8 \
  --output "$HDR_ROOT/e0/train_roots.json"

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py e0 \
  --manifest manifests/h20/qwen25_7b_hdr_memrl_seed2026.json \
  --accepted-manifest /data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json \
  --source-parquet "$MEMAGENT_HDR_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  --train-source-parquet "$MEMAGENT_HDR_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  --suite-parquet "$HDR_ROOT/e0/horizon_suite.parquet" \
  --tokenizer-root "$MEMAGENT_HDR_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --receipts "$HDR_ROOT/e0/receipts.json" --train-roots "$HDR_ROOT/e0/train_roots.json" \
  --eval-roots "$HDR_ROOT/e0/eval_roots.json" --output "$HDR_CERT/e0.json" --ledger "$HDR_LEDGER"

CUDA_VISIBLE_DEVICES=$GPU_PAIR $MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/run_hdr_strict_vllm_eval.py \
  --suite "$HDR_ROOT/e0/horizon_suite.parquet" \
  --model "$MEMAGENT_HDR_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --output "$HDR_ROOT/e1/frozen_predictions_scored.json" --seed 2026 --tensor-parallel-size 2

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py e1 \
  --manifest manifests/h20/qwen25_7b_hdr_memrl_seed2026.json \
  --rows "$HDR_ROOT/e1/frozen_predictions_scored.json" --receipts "$HDR_ROOT/e0/receipts.json" \
  --model-path "$MEMAGENT_HDR_WORK_ROOT/models/Qwen2.5-7B-Instruct" --seed 2026 \
  --git-commit "$MEMAGENT_HDR_EXPECTED_COMMIT" --output "$HDR_CERT/e1.json" --ledger "$HDR_LEDGER"

# The released independent paper_review.json must be copied into $HDR_CERT here.
# Its exact bytes must hash to MEMAGENT_HDR_REVIEW_SHA256.

HDR_TARGET_STEP=25 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py train-health \
  --anchor 5 --output-root "$HDR_OUTPUT" --weight-sync-ledger "$HDR_ROOT/hdr_weight_sync_ledger.jsonl" \
  --output "$HDR_CERT/t5_train_health.json" --ledger "$HDR_LEDGER"

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py baseline-import \
  --final-report /data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821/certificates/original_s128_curve_final_report.json \
  --curve-ledger /data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821/original_s128_curve_execution_ledger.jsonl \
  --stable-resolved /data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json \
  --validation-parquet "$MEMAGENT_HDR_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  --materialized-rows "$HDR_ROOT/original_curve/recomputed_metric_rows.json" \
  --output "$HDR_CERT/baseline_import.json" --ledger "$HDR_LEDGER"

for SPEC in 'all:8 10 12 16 24 32' 'nominal:8'; do
  NAME=${SPEC%%:*}; HORIZONS=${SPEC#*:}
  $MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/prepare_hdr_horizon_suite.py \
    --input "$MEMAGENT_HDR_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
    --input-sha256 54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6 \
    --identity-resolved /data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json \
    --identity-resolved-sha256 6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411 \
    --model "$MEMAGENT_HDR_WORK_ROOT/models/Qwen2.5-7B-Instruct" --horizons $HORIZONS --count 128 \
    --output-parquet "$HDR_ROOT/eval/fixed_s128_${NAME}_horizons.parquet" \
    --receipts "$HDR_ROOT/eval/fixed_s128_${NAME}_receipts.json" \
    --roots "$HDR_ROOT/eval/fixed_s128_${NAME}_roots.json"
done
mv "$HDR_ROOT/eval/fixed_s128_nominal_horizons.parquet" "$HDR_ROOT/eval/fixed_s128_nominal_h8.parquet"

for HDR_ANCHOR in 5 10 15 20 25; do
  export HDR_ANCHOR
  bash scripts/h20/audit_qwen25_7b_hdr_anchor.sh
done

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py final-audit \
  --run-root "$HDR_ROOT" --output-root "$HDR_OUTPUT" --report "$HDR_CERT/final_audit.json"
```

Any failed or occupied-GPU condition is a NO-GO. The scripts never kill other
processes and never overwrite an existing run or evaluation artifact.

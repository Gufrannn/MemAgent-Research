# HDR-MemRL H20 runbook (fail closed)

This branch does not claim that H20 training has been run. The method launcher exists now, but refuses GPU access until P0, E0, E1, baseline import, and independent paper-framing review all contain PASS decisions. E1 NO-GO is terminal for training.

## Runtime contract

- Branch: `h20/qwen25-7b-hdr-memrl-t25-frozen-20260822`.
- Use the exact released commit printed by the delivery; never `HEAD` as an implicit identity.
- `GPU_PAIR` is any two idle H20 IDs in canonical ascending order, including non-contiguous pairs.
- Locks are acquired separately at `locks/memagent_h20_gpu_N.lock`; contention or observed GPU processes produce NO-GO and never kill a process.
- `HDR_RUN_ID` is unique and immutable. A failed output is retained; retry with a new run ID/variant commit.
- Method starts from Qwen2.5-7B fresh base at update 1. Step-3 Original warm-start is forbidden.

## Copyable sequential chain

```bash
git fetch origin h20/qwen25-7b-hdr-memrl-t25-frozen-20260822
git checkout h20/qwen25-7b-hdr-memrl-t25-frozen-20260822
git reset --keep <RELEASE_EXACT_SHA>

export MEMAGENT_HDR_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_HDR_EXPECTED_COMMIT=<RELEASE_EXACT_SHA>
export GPU_PAIR=2,5
export HDR_RUN_ID=hdr-seed2026-v1
export HDR_REPO=$PWD
export HDR_CERT=$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/certificates
export HDR_LEDGER=$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/hdr_execution_ledger.jsonl
mkdir -p "$HDR_CERT"

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py preflight \
  --manifest manifests/h20/qwen25_7b_hdr_memrl_seed2026.json \
  --accepted-manifest manifests/h20/qwen25_7b_original_t25_seed2026.json \
  --output "$HDR_CERT/p0.json" --ledger "$HDR_LEDGER"

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/prepare_hdr_horizon_suite.py \
  --input "$MEMAGENT_HDR_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  --input-sha256 54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6 \
  --model "$MEMAGENT_HDR_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --horizons 8 12 16 32 --count 128 \
  --output-parquet "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e0/horizon_suite.parquet" \
  --receipts "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e0/receipts.json" \
  --roots "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e0/eval_roots.json"

# train_roots.json is produced from the frozen train split by the identical
# preparer invocation (a separate output path); only its roots file is used here.
$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py e0 \
  --manifest manifests/h20/qwen25_7b_hdr_memrl_seed2026.json \
  --receipts "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e0/receipts.json" \
  --train-roots "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e0/train_roots.json" \
  --eval-roots "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e0/eval_roots.json" \
  --output "$HDR_CERT/e0.json" --ledger "$HDR_LEDGER"

CUDA_VISIBLE_DEVICES=$GPU_PAIR $MEMAGENT_HDR_WORK_ROOT/.venv/bin/python \
  tools/h20/run_hdr_strict_vllm_eval.py \
  --suite "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e0/horizon_suite.parquet" \
  --model "$MEMAGENT_HDR_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --output "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e1/frozen_predictions_scored.json" \
  --seed 2026 --tensor-parallel-size 2

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py e1 \
  --manifest manifests/h20/qwen25_7b_hdr_memrl_seed2026.json \
  --rows "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e1/frozen_predictions_scored.json" \
  --receipts "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e0/receipts.json" \
  --model-path "$MEMAGENT_HDR_WORK_ROOT/models/Qwen2.5-7B-Instruct" --seed 2026 \
  --git-commit "$MEMAGENT_HDR_EXPECTED_COMMIT" \
  --output "$HDR_CERT/e1.json" --ledger "$HDR_LEDGER"

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py baseline-import \
  --bundle "$MEMAGENT_HDR_BASELINE_BUNDLE" \
  --expected-bundle-sha256 "$MEMAGENT_HDR_BASELINE_BUNDLE_SHA256" \
  --authority-manifest "$MEMAGENT_ORIGINAL_CURVE_AUTHORITY_MANIFEST" \
  --output "$HDR_CERT/baseline_import.json" --ledger "$HDR_LEDGER"

# paper_review.json is supplied only by the independent reviewer and must say
# {"status":"PASS","decision":"PAPER_FRAMING_GO"}. No local bypass exists.

HDR_TARGET_STEP=5 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
# Merge the actor read-only, then run run_hdr_strict_vllm_eval.py on the fixed
# S128 nominal suite and full registered+unseen horizon suite. The same evaluator
# is used for every anchor. It emits raw prediction/gold rows; no aggregate is trusted.
$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python scripts/model_merger.py --backend fsdp \
  --hf_model_path "$MEMAGENT_HDR_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --local_dir "$MEMAGENT_HDR_WORK_ROOT/logs/memory_agent/qwen25_7b_hdr_memrl_seed2026_$HDR_RUN_ID/global_step_5/actor" \
  --target_dir "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/eval/t5_merged"
CUDA_VISIBLE_DEVICES=$GPU_PAIR $MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/run_hdr_strict_vllm_eval.py \
  --suite "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/eval/fixed_s128_horizon_suite.parquet" \
  --model "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/eval/t5_merged" \
  --output "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/eval/t5_horizons.json"
CUDA_VISIBLE_DEVICES=$GPU_PAIR $MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/run_hdr_strict_vllm_eval.py \
  --suite "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/eval/fixed_s128_nominal_h8.parquet" \
  --model "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/eval/t5_merged" \
  --output "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/eval/t5_s128_nominal.json"
$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py health-gate \
  --anchor 5 --baseline-import "$HDR_CERT/baseline_import.json" \
  --method-s128 "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/eval/t5_s128_nominal.json" \
  --method-horizons "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/eval/t5_horizons.json" \
  --original-horizons "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/eval/original_t5_horizons_authority.json" \
  --nominal 8 --unseen 10 24 --output "$HDR_CERT/t5_health.json" --ledger "$HDR_LEDGER"

# The launcher requires t5_health PASS before T10. Repeat merge, strict raw
# S128+horizon evaluation, and health-gate at T10/T15/T20 before each continuation.
HDR_TARGET_STEP=10 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_TARGET_STEP=15 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_TARGET_STEP=20 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_TARGET_STEP=25 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
# T25 health-gate additionally requires --uniform-horizons from the independently
# committed, budget-matched uniform-ERM variant; it enforces the registered +2pp gate.

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py final-audit \
  --run-root "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID" \
  --output-root "$MEMAGENT_HDR_WORK_ROOT/logs/memory_agent/qwen25_7b_hdr_memrl_seed2026_$HDR_RUN_ID" \
  --report "$HDR_CERT/final_audit.json"
```

## Monitoring and recovery

Inspect `train_to_tN.log`, the hash-chained HDR ledger, the independent weight-sync ledger, and `hdr_dro_state.json` at every anchor. Recovery is allowed only from the immediately preceding complete anchor in the same run and exact commit. The launcher rejects missing actor/data/dual state and rejects an existing target. Never delete a failed directory to reuse its identity.

Estimated H20 wall time depends on the accepted Original measurements. Budget approximately the same trajectory generation per update as Original, with extra turns for larger horizons; preflight/E0 are CPU-bound, E1 is a zero-training multi-horizon evaluation, T5 is the first five-update fresh run, and each continuation adds five updates plus fixed S128/horizon evaluation. Record observed duration rather than claiming a local estimate as measured H20 time.

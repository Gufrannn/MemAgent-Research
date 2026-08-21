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
export MEMAGENT_HDR_BASELINE_BUNDLE_SHA256=<CONTROLLER_AUTHORIZED_BASELINE_BUNDLE_SHA256>
export MEMAGENT_HDR_REVIEW_SHA256=<INDEPENDENT_REVIEW_ATTESTATION_SHA256>
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

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/prepare_hdr_root_ids.py \
  --input "$MEMAGENT_HDR_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  --input-sha256 798b7a2a9ece4f40884e2a9d02d165d7352df7763d1569ceaf402b45f76896f8 \
  --output "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e0/train_roots.json"

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py e0 \
  --manifest manifests/h20/qwen25_7b_hdr_memrl_seed2026.json \
  --accepted-manifest manifests/h20/qwen25_7b_original_t25_seed2026.json \
  --source-parquet "$MEMAGENT_HDR_WORK_ROOT/datasets/hotpotqa/hotpotqa_dev.parquet" \
  --train-source-parquet "$MEMAGENT_HDR_WORK_ROOT/datasets/hotpotqa/hotpotqa_train_32k.parquet" \
  --suite-parquet "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID/e0/horizon_suite.parquet" \
  --tokenizer-root "$MEMAGENT_HDR_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
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
HDR_ANCHOR=5 HDR_VARIANT=dro bash scripts/h20/audit_qwen25_7b_hdr_anchor.sh
HDR_TARGET_STEP=10 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_ANCHOR=10 HDR_VARIANT=dro bash scripts/h20/audit_qwen25_7b_hdr_anchor.sh
HDR_TARGET_STEP=15 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_ANCHOR=15 HDR_VARIANT=dro bash scripts/h20/audit_qwen25_7b_hdr_anchor.sh
HDR_TARGET_STEP=20 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_ANCHOR=20 HDR_VARIANT=dro bash scripts/h20/audit_qwen25_7b_hdr_anchor.sh
HDR_TARGET_STEP=25 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_ANCHOR=25 HDR_VARIANT=dro bash scripts/h20/audit_qwen25_7b_hdr_anchor.sh
# T25 health-gate additionally requires --uniform-horizons from the independently
# committed, budget-matched uniform-ERM variant; it enforces the registered +2pp gate.

# Budget-matched uniform ERM is a separate experiment identity/output/ledger.
# Run it from fresh base with its own immutable run ID before the DRO T25 audit:
export HDR_VARIANT=uniform HDR_RUN_ID=hdr-uniform-seed2026-v1
HDR_TARGET_STEP=5 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_ANCHOR=5 bash scripts/h20/audit_qwen25_7b_hdr_anchor.sh
HDR_TARGET_STEP=10 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_ANCHOR=10 bash scripts/h20/audit_qwen25_7b_hdr_anchor.sh
HDR_TARGET_STEP=15 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_ANCHOR=15 bash scripts/h20/audit_qwen25_7b_hdr_anchor.sh
HDR_TARGET_STEP=20 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_ANCHOR=20 bash scripts/h20/audit_qwen25_7b_hdr_anchor.sh
HDR_TARGET_STEP=25 bash scripts/h20/run_qwen25_7b_hdr_memrl.sh
HDR_ANCHOR=25 bash scripts/h20/audit_qwen25_7b_hdr_anchor.sh
# Copy nothing blindly: register the uniform T25 raw evaluation SHA in the DRO
# authority bundle, then expose its audited JSON at the exact path required by
# the DRO anchor auditor. Restore HDR_VARIANT=dro and its original HDR_RUN_ID.

$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python tools/h20/hdr_memrl_control.py final-audit \
  --run-root "$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID" \
  --output-root "$MEMAGENT_HDR_WORK_ROOT/logs/memory_agent/qwen25_7b_hdr_memrl_seed2026_$HDR_RUN_ID" \
  --report "$HDR_CERT/final_audit.json"
```

## Monitoring and recovery

Inspect `train_to_tN.log`, the hash-chained HDR ledger, the independent weight-sync ledger, and `hdr_dro_state.json` at every anchor. Recovery is allowed only from the immediately preceding complete anchor in the same run and exact commit. The launcher rejects missing actor/data/dual state and rejects an existing target. Never delete a failed directory to reuse its identity.

Estimated H20 wall time depends on the accepted Original measurements. Budget approximately the same trajectory generation per update as Original, with extra turns for larger horizons; preflight/E0 are CPU-bound, E1 is a zero-training multi-horizon evaluation, T5 is the first five-update fresh run, and each continuation adds five updates plus fixed S128/horizon evaluation. Record observed duration rather than claiming a local estimate as measured H20 time.

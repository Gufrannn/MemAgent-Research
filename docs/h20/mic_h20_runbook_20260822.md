# MIC H20 execution runbook

Scientific status at publication of this branch: **REFRAME, GPU-locked until
P0 + E0 + E1 + baseline import + independent paper/code review all PASS.** The
presence of launch code is not authorization to bypass a failed gate.

## Evidence inputs

The canonical read-only sources shared by every method branch are:

- Original S128 curve: `/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821`.
  Its certified final report is the only authority for the absolute artifact
  paths and file SHA-256 values for `I`, `Original5`, `Original10`,
  `Original15`, `Original20`, and `Original25`. The published per-interface
  canonical-row digests in the method manifest are not file hashes.
- Original training protocol: `/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821`.
  Protocol equality binds its `certificates/p0_resolved_manifest.json`, not the
  S128 curve resolved manifest. Its exact resolved SHA must be frozen from the
  read-only H20 artifact before release; it must not be guessed.
- Fixed-S128 identity: `/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2`.

There is no certified standalone baseline inventory yet. It must be
materialized from the certified S128 final-report inventory, then MIC must
independently recompute EM, token-F1, and format success from all 128 rows.
The materialized inventory SHA must be certified out-of-band before P0. Do not
infer paths, scores, or hashes from aggregate fields. Original actual-loss
rank ledgers were never collected; their status is
`PENDING_ACTUAL_LOSS_LEDGER` and they must not be synthesized or rerun.

The H20 host must already contain the certified Original evidence. Do not rerun
Original. Create a read-only JSON inventory with `files` (every consumed file's
absolute path and SHA-256) and `prediction_files` for exactly `I`, `Original5`,
`Original10`, `Original15`, `Original20`, and `Original25`. Each prediction item
contains `path`, `sha256`, `interface`, and the certified `expected_aggregate`.
Prediction JSONL rows contain `stable_key`, `source_order_index`, `output`, and
`ground_truth`. MIC recomputes normalized EM, token F1, and format success from
those fields; stored reward/aggregate fields are not trusted.

E1 consumes a read-only JSON bundle with `states` and `outcomes`. Each state is
limited to stable IDs, turn index, question, visible chunks through that turn,
and exact materialized memory. Gold, future chunks, current-row outcome, and
generated answers are rejected by schema. Outcomes live in a separate mapping
and enter only fold-exclusive fitting/evaluation.

## Copy/paste sequence

```bash
export MEMAGENT_MIC_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_MIC_REPO_DIR=/data/cw/memagent_work/code/MemAgent-MIC
export MEMAGENT_MIC_EXPECTED_COMMIT=<RELEASE_SHA_FROM_HANDOFF>
export MEMAGENT_MIC_GPU_PAIR=2,7
export MEMAGENT_MIC_RUN_ID=mic-main-v1-seed2026
export MEMAGENT_MIC_ORIGINAL_RESOLVED_MANIFEST=/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json
export MEMAGENT_MIC_BASELINE_INVENTORY=/data/cw/memagent_work/evidence/original/baseline_inventory.json
export MEMAGENT_MIC_BASELINE_AUTHORITY_SHA256=<CERTIFIED_OUT_OF_BAND_INVENTORY_SHA256>
export MEMAGENT_MIC_E1_BUNDLE=/data/cw/memagent_work/evidence/mic/e1_frozen_original_trajectories.json

cd /data/cw/memagent_work/code
git fetch origin h20/qwen25-7b-mic-t25-frozen-20260822
git clone --no-checkout <REPOSITORY_URL> MemAgent-MIC  # omit when already cloned
cd "$MEMAGENT_MIC_REPO_DIR"
git fetch origin h20/qwen25-7b-mic-t25-frozen-20260822
git switch --detach "$MEMAGENT_MIC_EXPECTED_COMMIT"
git switch -C h20/qwen25-7b-mic-t25-frozen-20260822 "$MEMAGENT_MIC_EXPECTED_COMMIT"

bash scripts/h20/preflight_qwen25_7b_mic.sh
bash scripts/h20/run_qwen25_7b_mic_t5.sh
bash scripts/h20/eval_audit_qwen25_7b_mic.sh 5

bash scripts/h20/continue_qwen25_7b_mic.sh 5 10
bash scripts/h20/eval_audit_qwen25_7b_mic.sh 10
bash scripts/h20/continue_qwen25_7b_mic.sh 10 15
bash scripts/h20/eval_audit_qwen25_7b_mic.sh 15
bash scripts/h20/continue_qwen25_7b_mic.sh 15 20
bash scripts/h20/eval_audit_qwen25_7b_mic.sh 20
bash scripts/h20/continue_qwen25_7b_mic.sh 20 25
bash scripts/h20/eval_audit_qwen25_7b_mic.sh 25
```

The equivalent unattended sequence is
`bash scripts/h20/run_qwen25_7b_mic_full_chain.sh`, but the explicit sequence is
preferred so the T5 health report can be inspected before continuation.

## Runtime guarantees

- `GPU_PAIR` is required, canonical ascending, distinct, and may be
  non-contiguous. CUDA sees it as logical devices `0,1`.
- The runner acquires
  `locks/memagent_h20_gpu_N.lock` independently for both physical GPUs. A lock
  conflict or any compute PID produces `MIC_NO_GO`; no process is killed.
- Fresh T5 uses the untouched Qwen2.5-7B base with MIC active at update 1.
  Resume accepts only `5->10->15->20->25` within the same run. Original step 3
  is neither accepted nor referenced as a source checkpoint.
- Output roots include the explicit run ID. Fresh and evaluation entry points
  refuse existing paths. Failed evidence remains in place; recovery uses the
  exact last complete checkpoint or a new run/variant ID, never overwrite.
- Actor/FSDP/vLLM synchronization remains covered by the repository's existing
  sampled-weight receipts. MIC adds a separate critic checkpoint at every
  update, bound to the exact actor Git commit and OOF fold bundle.
- The execution ledger is append-only and hash chained. Audit reconstructs its
  chain, critic checkpoint bindings, role delivery hashes, and telescoping
  error. A missing or modified entry fails closed.
- Evaluation is strict vLLM, greedy, `n=1`, and actor-only. The result producer
  joins the frozen S128 identity inventory, then independently recomputes the
  metrics. Dense reward and critic diagnostics never substitute for performance.

## T5 revision and recovery

T5 continuation requires both `MIC_T5_AUDIT_PASS` and
`MIC_T5_HEALTH_PASS`. A failed T5 is preserved. At most one new variant may be
created, with a new manifest/run ID/commit/output/ledger, and only for the
preregistered classes `numerical_instability` or `critic_regularization`. A
second T5 failure is `NO-GO_T5`.

Normal wall-clock estimates on two H20s are hardware/queue dependent. Budget
approximately 15--30 minutes for P0/E0/E1 and baseline verification, 2--5 hours
per five-update training segment, and 1--3 hours per fixed-S128 evaluation.
These are planning ranges, not claims from a local H20 run. This branch was not
GPU-executed on the development machine.

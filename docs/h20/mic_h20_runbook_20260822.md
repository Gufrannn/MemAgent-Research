# MIC H20 execution runbook

Scientific framing remains **REFRAME pending fixed-S128 performance**. Training
and evaluation are separately fail-closed: Method training requires P0, E0 and
independent paper/code review; performance claims require all five health audits,
the release-pinned checkpoint authority, and the final fixed-S128 audit.

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

There is no certified standalone baseline inventory. The evaluation entry
deterministically materializes it from the MIC-P0-bound Original curve final
report: terminal artifacts are located by the report's certified file SHA,
joined to the frozen S128 identity/parquet, and accepted only when an
independent recomputation exactly reproduces every certified canonical
metric-row digest. The generated inventory is not a trust root and cannot
self-authenticate. Do not infer paths, scores, or hashes from aggregate fields.
Original actual-loss
rank ledgers were never collected; their status is
`PENDING_ACTUAL_LOSS_LEDGER` and they must not be synthesized or rerun.

The H20 host must already contain the certified Original evidence. Do not rerun
Original or hand-author an inventory. Normalized rows and their provenance are
written append-only under the Method run root at evaluation time.

MIC mechanism diagnostics use Method's own on-policy materialized states and
outcomes. Missing Original-internal trajectories or actual-loss tensors never
block Method training and are not reconstructed from aggregates.

The frozen Original curve resolved manifest is separately bound by the
out-of-band SHA in
`manifests/h20/qwen25_7b_mic_original_curve_authority.json`; the generated
baseline inventory is never allowed to authenticate it. Method checkpoint
authority has two explicit modes. A fresh run records the complete two-shard
inventory atomically at every saved anchor in the MIC hash chain; evaluation
replays that chain and rehashes all ten shards. The already-completed legacy
`runtimefix3` run predates that writer and therefore uses its operator-computed
read-only ten-shard digest pinned in
`manifests/h20/qwen25_7b_mic_checkpoint_authority.json`. Every anchor also
matches loaded actor/vLLM sampled weights to the authenticated target update
and verifies the full actor shards remain byte-identical during generation.

## Copy/paste sequence

```bash
export MEMAGENT_MIC_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_MIC_REPO_DIR=/data/cw/memagent_work/code/MemAgent-MIC
export MEMAGENT_MIC_EXPECTED_COMMIT=<RELEASE_SHA_FROM_HANDOFF>
export MEMAGENT_MIC_GPU_PAIR=2,7
export MEMAGENT_MIC_RUN_ID=mic-main-v1-seed2026
export MEMAGENT_MIC_ORIGINAL_RESOLVED_MANIFEST=/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json
export MEMAGENT_MIC_ORIGINAL_CURVE_REPORT=/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821/certificates/original_s128_curve_final_report.json

cd /data/cw/memagent_work/code
git fetch origin h20/qwen25-7b-mic-t25-frozen-20260822
git clone --no-checkout <REPOSITORY_URL> MemAgent-MIC  # omit when already cloned
cd "$MEMAGENT_MIC_REPO_DIR"
git fetch origin h20/qwen25-7b-mic-t25-frozen-20260822
git switch --detach "$MEMAGENT_MIC_EXPECTED_COMMIT"
git switch -C h20/qwen25-7b-mic-t25-frozen-20260822 "$MEMAGENT_MIC_EXPECTED_COMMIT"

bash scripts/h20/preflight_qwen25_7b_mic.sh
bash scripts/h20/run_qwen25_7b_mic_t5.sh
bash scripts/h20/audit_health_qwen25_7b_mic.sh 5

bash scripts/h20/continue_qwen25_7b_mic.sh 5 10
bash scripts/h20/audit_health_qwen25_7b_mic.sh 10
bash scripts/h20/continue_qwen25_7b_mic.sh 10 15
bash scripts/h20/audit_health_qwen25_7b_mic.sh 15
bash scripts/h20/continue_qwen25_7b_mic.sh 15 20
bash scripts/h20/audit_health_qwen25_7b_mic.sh 20
bash scripts/h20/continue_qwen25_7b_mic.sh 20 25
bash scripts/h20/audit_health_qwen25_7b_mic.sh 25

# Only after Method-T25 and all health audits exist. This materializes the
# baseline once, evaluates all five anchors, and writes final_eval_audit.json.
bash scripts/h20/eval_all_qwen25_7b_mic.sh
```

The equivalent unattended sequence is
`bash scripts/h20/run_qwen25_7b_mic_full_chain.sh`, but the explicit sequence is
preferred so the cheap T5 training-health report can be inspected before
continuation. Fixed-S128 generation is deliberately deferred until T25.

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
  never overwrite artifacts. Training recovery uses the exact last complete
  checkpoint. Evaluation writes append-only
  `eval_tN_attempts/attempt_KKKK` directories: a failed attempt remains intact,
  and rerunning the same all-anchor command allocates the next attempt. A
  completed anchor is skipped only after `prepare-eval --verify-existing` and
  `evaluate --verify-existing` reauthenticate its primary artifacts; the final
  audit repeats those checks for the certificate-bound successful attempt.
- Actor/FSDP/vLLM synchronization remains covered by the repository's existing
  sampled-weight receipts. MIC adds a separate critic checkpoint at every
  update, bound to the exact actor Git commit and OOF fold bundle.
- The execution ledger is append-only and hash chained. Audit reconstructs its
  chain, critic checkpoint bindings, role delivery hashes, and telescoping
  error. A missing or modified entry fails closed.
- Evaluation is strict vLLM, greedy, `n=1`, and actor-only. It loads only the
  two FSDP model shards; optimizer, scheduler, RNG, and dataloader state remain
  untouched. The training evidence writer is disabled. The full actor shards
  must match the release-pinned five-anchor authority and remain byte-identical;
  actor/vLLM sampled weights must also match the historical target-step sync and
  remain identical before/after generation. The strict Original S128 generation
  fields are asserted inside the trainer. The result producer exact-joins frozen
  identities and independently recomputes metrics. Baseline, prepare, metric and
  final audits all re-open their primary evidence instead of trusting a prior
  PASS JSON. Dense reward and critic diagnostics never substitute for performance.

## T5 revision and recovery

T5 continuation requires both `MIC_T5_AUDIT_PASS` and
`MIC_T5_TRAINING_HEALTH_PASS`; this checks finite optimizer/gradient,
checkpoint, ledger, and weight-sync evidence, not S128 performance. A failed T5
is preserved. At most one new variant may be
created, with a new manifest/run ID/commit/output/ledger, and only for the
preregistered classes `numerical_instability` or `critic_regularization`. A
second T5 failure is `NO-GO_T5`.

Normal wall-clock estimates on two H20s are hardware/queue dependent. Budget
approximately 5--15 minutes for P0/E0, 2--5 hours
per five-update training segment, and 1--3 hours per fixed-S128 evaluation.
These are planning ranges, not claims from a local H20 run. This branch was not
GPU-executed on the development machine.

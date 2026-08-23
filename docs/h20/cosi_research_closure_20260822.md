# CORAL H20 runbook: research gates, one fresh early-budget T25 pilot, then five development anchors

Scientific state at commit time is recorded in `docs/papers/coral_paper_draft.md`. No H20 result is claimed by this repository. The committed manifest keeps `training_authorized=false`: the repository cannot authorize itself. The fresh-T25 executable becomes runnable only when externally issued, content-addressed paper-framing, E0, direct single-update E1, read-only Original-baseline, and scientific-scope/data-overlap certificates all pass and their expected hashes are supplied outside the repository. See `docs/h20/COSI_RELEASE_STATUS.md`; older tips are not valid H20 entries.

## 1. Exact checkout and explicit bindings

```bash
export MEMAGENT_COSI_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_COSI_REPO_DIR=/data/cw/memagent_work/code/MemAgent-Research
export MEMAGENT_COSI_EXPECTED_COMMIT=<EXACT_REMOTE_SHA>
export MEMAGENT_COSI_GPU_PAIR=2,7
export MEMAGENT_COSI_RUN_ID=coral_seed2026_primary_v1
export MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST=/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json
export MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST_SHA256=<READ_ONLY_FROZEN_ORIGINAL_T25_RESOLVED_64HEX>
export MEMAGENT_COSI_S128_RESOLVED_MANIFEST=/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json
export MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256=6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411

cd "$MEMAGENT_COSI_REPO_DIR"
git fetch origin h20/qwen25-7b-cosi-t25-frozen-20260822
git switch -C h20/qwen25-7b-cosi-t25-frozen-20260822 "$MEMAGENT_COSI_EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$MEMAGENT_COSI_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
```

Before exporting the Original resolved SHA placeholder above, run `sha256sum /data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json` read-only and freeze the printed 64-hex value in the trusted experiment record. The runner verifies that value but never modifies or reruns Original.

`MEMAGENT_COSI_GPU_PAIR` is any two distinct physical H20 indices in canonical ascending order; it may be non-contiguous. Every GPU entry acquires `locks/memagent_h20_gpu_N.lock` in ascending order, checks `nvidia-smi`, and exits without changing another process if either card is busy.

There is no standalone baseline bundle. The research entry authenticates the completed Original curve at `/data/cw/memagent_work/logs/s128_original_all_anchor_frozen_20260821` through its final report, P0, resolved manifest, append-only ledger, and the six externally frozen canonical metric-row digests in the Method manifest. Artifact paths and SHA-256 values are read only from the authenticated final inventory; paths are never guessed. The importer binds the frozen S128 identity order to the authenticated validation parquet and independently recomputes normalized exact match, token-F1, and format. It never reruns Original and never treats dense reward as performance. Original `actual_loss_rank0/1.jsonl` does not exist and is recorded as `PENDING_ACTUAL_LOSS_LEDGER`; it is not required by the CORAL training or performance comparison.

The accepted Original training protocol is the resolved manifest under `/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821`, not the S128 curve resolved manifest. Before the first run, compute its SHA-256 read-only, record it through the trusted experiment channel, and place that frozen value in `MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST_SHA256`. Do not derive the value from a copied or modified file.

## 2. Research gates (no long training)

```bash
bash scripts/h20/run_qwen25_7b_cosi_research.sh
## The GPU diagnostic entry is deliberately locked at this source tip.
## It exits before GPU locking until capacity and an exact replacement identity
## are frozen in a later independently reviewed commit.
bash scripts/h20/run_qwen25_7b_coral_e1_producer.sh  # expected fail-closed exit 81
```

E0 is an exact CPU enumeration. E1 is produced inside the real trainer, not imported from external JSON: odd writer updates 1,3,...,15 each contribute four new b4 roots with two trajectory replicas. After the actual writer update and audited vLLM synchronization, the same roots/seeds are regenerated. Source-policy final answers are discarded. Cached-old-memory and refreshed-memory terminal prompts are each sampled anew at the proposal weights with identical terminal request seeds, and reward/GRPO advantage is recomputed per branch. The actor then performs read-only actual-loss backward passes for cached answer, duplicate cached answer, refreshed answer, and cached writer. Four frozen, nonlinearly basis-separated 256-bucket CountSketch maps, an optimizer-step non-mutation check, the Gate-A ledger, and full actor inventories are sealed. The 32 roots are descriptive units nested four per writer proposal; uncertainty and the LCB are computed from the eight proposal means. This diagnostic output is never a Method warm-start.

The E1 entry is unconditionally locked before GPU locking at this source tip. Versions v3--v11 are retired evidence identities. In particular, v11 produced proposal receipts 1/3/5 and checkpoints through step 6 before `/data` reached 100% capacity; it is incomplete and cannot be resumed or reused. Supplying an arbitrary post-v11 run ID does not unlock the entry. A later source commit may remove the lock only after storage is provisioned for the complete append-only inventory and an independent review freezes one exact fresh replacement identity. Every accepted replacement must rerun both oracles and all updates from the fresh base and import no earlier artifact. Before the distributed sketch oracle or trainer starts, the same `$MEMAGENT_COSI_WORK_ROOT/.venv/bin/python` executes a zero-tensor-leaf `DataProto` clone oracle. Its complete runtime report and SHA are embedded by the sealer and revalidated by the final E1 auditor; bypassing that entry gate cannot yield a valid E1 certificate. The two-rank sketch oracle additionally calibrates 64 distinct parameter ordinals against the pure fixed-map reference before trainer entry.

Expected H20 time for E1 is approximately 2--5 hours because it includes eight same-root regenerated rollouts and 128 root-level backward measurements. This is the required mechanism diagnostic, not the fresh Method curve; the fresh-T25 command remains locked unless it and the independent framing review pass.

An independent reviewer must write an authenticated `paper_framing_review.json` with decision `CORAL_PAPER_FRAMING_GO` into `logs/cosi_preflight/certificates`. No script synthesizes that scientific judgment.

Before binding the five gate hashes below, run the data-only scope command in Section 7 for the exact current commit and require its PASS output. This command is separate so an already authenticated reusable E0/baseline certificate is never overwritten merely to produce the commit-specific scope certificate.

After the reviewer has inspected the real E1 report and issued that certificate,
bind all five gate files from outside the repository before T5:

```bash
CERT="$MEMAGENT_COSI_WORK_ROOT/logs/cosi_preflight/certificates"
export MEMAGENT_COSI_PAPER_REVIEW_SHA256=<EXTERNALLY_ISSUED_PAPER_REVIEW_64HEX>
export MEMAGENT_COSI_E0_REPORT_SHA256=<EXTERNALLY_ISSUED_E0_64HEX>
export MEMAGENT_COSI_E1_REPORT_SHA256=<EXTERNALLY_ISSUED_E1_64HEX>
export MEMAGENT_COSI_BASELINE_REPORT_SHA256=<EXTERNALLY_ISSUED_BASELINE_64HEX>
export MEMAGENT_COSI_SCOPE_REPORT_SHA256=<EXTERNALLY_ISSUED_SCOPE_REPORT_64HEX>
test "$(shasum -a 256 "$CERT/paper_framing_review.json" | awk '{print $1}')" = "$MEMAGENT_COSI_PAPER_REVIEW_SHA256"
test "$(shasum -a 256 "$CERT/coral_e0.json" | awk '{print $1}')" = "$MEMAGENT_COSI_E0_REPORT_SHA256"
test "$(shasum -a 256 "$CERT/coral_e1_final_report.json" | awk '{print $1}')" = "$MEMAGENT_COSI_E1_REPORT_SHA256"
test "$(shasum -a 256 "$CERT/baseline_import.json" | awk '{print $1}')" = "$MEMAGENT_COSI_BASELINE_REPORT_SHA256"
test "$(shasum -a 256 "$CERT/coral_scope_data_${MEMAGENT_COSI_EXPECTED_COMMIT}.json" | awk '{print $1}')" = "$MEMAGENT_COSI_SCOPE_REPORT_SHA256"
python tools/h20/audit_qwen25_7b_cosi.py \
  --run-root "$MEMAGENT_COSI_WORK_ROOT/logs/cosi_preflight" \
  --stage research \
  --output "$CERT/research_audit.json"
```

The expected hashes must arrive through a trusted channel independent of the
files being checked; never derive them from the local certificates. They are
runtime inputs, not values stored in the manifest. A missing or substituted
gate therefore fails before trainer entry.

## 3. One fresh-base Method early-budget pilot through T25

```bash
bash scripts/h20/run_qwen25_7b_cosi_t25.sh
```

This creates a new append-only output and runs updates 1--25 from the Qwen2.5-7B base in one trainer invocation. CORAL is active at update 1. It never loads Original step 3 or any Original checkpoint. The run uses seed 2026, b4, n2, mini4, unchanged reward/tokenizer/data order, strict vLLM, two GPUs, audited post-update weight synchronization, and saves T5/T10/T15/T20/T25. Its static budget is 25 actor optimizer updates, 100 prompt groups, 200 sampled trajectories, one PPO epoch and one prompt minibatch per update, with 13 writer-active and 12 terminal-answer-active updates. GRPO has no learned critic. Exact recurrent writer turns and token counts are runtime quantities and must come from the scope certificate described below. This is an early-budget single-seed pilot, not sufficient training or convergence evidence. Non-finite loss/gradient or a weight-sync failure aborts immediately. After training, the entry authenticates the T5 checkpoint and the first five role/gradient/sync ledger records without running evaluation or consulting Original.

Expected H20 time: approximately 3--6 hours for T25, depending on current vLLM/FSDP throughput. This is an estimate, not a measured result.

## 4. Five fixed-S128 development evaluations after T25

```bash
for step in 5 10 15 20 25; do
  bash scripts/h20/run_qwen25_7b_coral_s128.sh "$step"
done
export MEMAGENT_COSI_T5_P0_SHA256=<EXTERNALLY_FROZEN_METHOD_T5_P0_FILE_64HEX>
test "$(sha256sum "$MEMAGENT_COSI_WORK_ROOT/logs/cosi_preflight/certificates/p0_t5.json" | awk '{print $1}')" = "$MEMAGENT_COSI_T5_P0_SHA256"
bash scripts/h20/audit_qwen25_7b_coral_final.sh
```

Only after T25 completes, the evaluator loads each actor checkpoint, performs zero optimizer steps and zero checkpoint writes, uses the same fixed 128 stable identities and strict-vLLM recurrent protocol, independently rejoins ground truth, and rehashes the checkpoint after evaluation. The final audit compares all five Method anchors directly with the authenticated read-only Original T5/T10/T15/T20/T25 curve and reports descriptive paired differences. No T5 performance threshold interrupts the main training run. These S128 results are development screening: the Original curve and Capture32 S128 facts were visible during method framing, so this set is not a blind final test.

Expected H20 time: approximately 40–90 minutes for one fixed-S128 recurrent checkpoint evaluation.

## 5. Recovery only

```bash
bash scripts/h20/resume_qwen25_7b_cosi_t5_to_t25.sh
```

This path is not part of the default run. It is deliberately restricted to the same run ID stopped at the exact T5 boundary: the CORAL ledger must contain exactly steps 1--5 and no step 6+, and `global_step_5` must be complete. Before continue preflight, the recovery entry creates the cheap T5 health certificate from that source ledger/checkpoint. It then loads model, optimizer, scheduler, all four RNG states, and dataloader state on both ranks. Do not change `MEMAGENT_COSI_RUN_ID` for this recovery. An interruption after any step 6+ is preserved as evidence and rejected by this release rather than replayed from T5. The final audit accepts either zero resume events for the default uninterrupted run or exactly one rank-complete authenticated T5 recovery; it rejects ambiguous/multiple resumes. Existing run/evaluation paths are never overwritten.

Expected H20 time for a T5 recovery through update 25 is approximately 2.5–5 hours. Retain the append-only run root on any failure; recovery starts only from an authenticated checkpoint and a new experiment variant/run ID. Run the five fixed-S128 evaluations only after recovery reaches T25.

## 6. Monitoring and recovery

```bash
tail -f "$MEMAGENT_COSI_WORK_ROOT/logs/coral/$MEMAGENT_COSI_RUN_ID/gate_a_execution_ledger.jsonl"
tail -f "$MEMAGENT_COSI_WORK_ROOT/logs/coral/$MEMAGENT_COSI_RUN_ID/coral_execution_ledger.jsonl"
nvidia-smi -i "$MEMAGENT_COSI_GPU_PAIR"
```

Do not kill unknown GPU processes. A lock conflict, busy device, dirty checkout, wrong SHA, manifest drift, missing gate, ledger discontinuity, checkpoint mutation, resume mismatch, or actor/vLLM digest mismatch is a NO-GO. Use a new run ID for any authorized recovery; never delete or overwrite failed evidence.

## 7. Read-only scope/leakage certificate

Before interpreting any curve, run the content-addressed audit in the H20 venv. It authenticates both parquet files and the frozen S128 manifest, then computes exact question+context identity intersections; it does not train or evaluate a model.

```bash
SCOPE_CERT="$MEMAGENT_COSI_WORK_ROOT/logs/cosi_preflight/certificates/coral_scope_data_${MEMAGENT_COSI_EXPECTED_COMMIT}.json"
"$MEMAGENT_COSI_WORK_ROOT/.venv/bin/python" tools/h20/audit_coral_scientific_scope.py \
  --manifest manifests/h20/qwen25_7b_cosi_seed2026.json \
  --stable-resolved "$MEMAGENT_COSI_S128_RESOLVED_MANIFEST" \
  --stable-resolved-sha256 "$MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256" \
  --work-root "$MEMAGENT_COSI_WORK_ROOT" \
  --repo-dir "$MEMAGENT_COSI_REPO_DIR" \
  --expected-commit "$MEMAGENT_COSI_EXPECTED_COMMIT" \
  --output "$SCOPE_CERT"
```

The scope entry is data-only and rejects postrun-budget arguments. After a complete T25 pilot, the sole promotion path is `scripts/h20/audit_qwen25_7b_coral_final.sh` with the externally frozen Method-T5 P0 SHA shown in Section 4. That final entry derives writer turns/tokens and trajectories, imports the 100-example semantic cursor only from authenticated Original authority, requires two rank-complete optimizer-state acknowledgements plus full 199-parameter `dtensor` load coverage at every update, hashes all five rank-complete anchor checkpoints, and independently recomputes model/tokenizer inventory, exhaustive non-whitelist resolved-config equality, authority projections, gates, and GPU identity. A missing server artifact remains `PENDING`; it is never guessed from a filename.

## 8. Required attribution controls after the primary pilot

The primary 25-update pilot is sampled-trajectory-budget matched, but CORAL has only 13 writer-active and 12 terminal-answer-active updates. A paper claim therefore additionally requires Original-25, a random/same-mask half-exposure control, a terminal-first parity reversal, PCGrad or equivalent same-batch conflict control, and a role-exposure-matched CORAL run. Report generated tokens, reward calls, forward/backward FLOPs, weight swaps, active-role tokens, and wall time separately. These variants require independent manifests, output roots, ledgers, and commits; they may not overwrite the primary run. At least three fresh counterbalanced seeds and an untouched, independently frozen confirmation set are required before any generalization claim.

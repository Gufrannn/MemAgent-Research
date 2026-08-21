# CORAL H20 runbook: research gates, fresh T5, and T25 continuation

Scientific state at commit time is recorded in `docs/papers/coral_paper_draft.md`. No H20 result is claimed by this repository. The committed manifest keeps `training_authorized=false`: the repository cannot authorize itself. The T5 executable becomes runnable only when externally issued, content-addressed paper-framing, E0, direct single-update E1, and read-only Original-baseline certificates all pass and their expected hashes are supplied outside the repository.

## 1. Exact checkout and explicit bindings

```bash
export MEMAGENT_COSI_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_COSI_REPO_DIR=/data/cw/memagent_work/code/MemAgent-Research
export MEMAGENT_COSI_EXPECTED_COMMIT=<EXACT_REMOTE_SHA>
export MEMAGENT_COSI_GPU_PAIR=2,7
export MEMAGENT_COSI_RUN_ID=coral_seed2026_primary_v1
export MEMAGENT_COSI_BASELINE_INDEX=/data/cw/read_only_original_bundle/bundle_index.json
export MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST=/data/cw/read_only_original_training/p0_resolved_manifest.json
export MEMAGENT_COSI_S128_RESOLVED_MANIFEST=/data/cw/read_only_original_s128/p0_resolved_manifest.json
export MEMAGENT_COSI_E1_RUN_ID=coral_e1_seed2026_v3

cd "$MEMAGENT_COSI_REPO_DIR"
git fetch origin h20/qwen25-7b-cosi-t25-frozen-20260822
git switch -C h20/qwen25-7b-cosi-t25-frozen-20260822 "$MEMAGENT_COSI_EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$MEMAGENT_COSI_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
```

`MEMAGENT_COSI_GPU_PAIR` is any two distinct physical H20 indices in canonical ascending order; it may be non-contiguous. Every GPU entry acquires `locks/memagent_h20_gpu_N.lock` in ascending order, checks `nvidia-smi`, and exits without changing another process if either card is busy.

The Original bundle must be mounted read-only. Its index authenticates every source file and the I/Original5/10/15/20/25 prediction JSONL. Import recomputes normalized exact match, token-F1, and format from terminal text and parquet ground truth; stored aggregates and training dense reward are never accepted as performance.

## 2. Research gates (no long training)

```bash
bash scripts/h20/run_qwen25_7b_cosi_research.sh
## GPU mechanism diagnostic; it cannot be reused as Method warm-start.
bash scripts/h20/run_qwen25_7b_coral_e1_producer.sh
```

E0 is an exact CPU enumeration. E1 is produced inside the real trainer, not imported from external JSON: odd writer updates 1,3,...,15 each contribute four new b4 roots with two trajectory replicas. After the actual writer update and audited vLLM synchronization, the same roots/seeds are regenerated. Source-policy final answers are discarded. Cached-old-memory and refreshed-memory terminal prompts are each sampled anew at the proposal weights with identical terminal request seeds, and reward/GRPO advantage is recomputed per branch. The actor then performs read-only actual-loss backward passes for cached answer, duplicate cached answer, refreshed answer, and cached writer. Four frozen, nonlinearly basis-separated 256-bucket CountSketch maps, an optimizer-step non-mutation check, the Gate-A ledger, and full actor inventories are sealed. The 32 roots are descriptive units nested four per writer proposal; uncertainty and the LCB are computed from the eight proposal means. This diagnostic output is never a Method warm-start.

Expected H20 time for E1 is approximately 2--5 hours because it includes eight same-root regenerated rollouts and 128 root-level backward measurements. This is the required mechanism diagnostic, not the fresh Method curve; no T5 command becomes authorized unless it and the independent framing review pass.

An independent reviewer must write an authenticated `paper_framing_review.json` with decision `CORAL_PAPER_FRAMING_GO` into `logs/cosi_preflight/certificates`. No script synthesizes that scientific judgment.

After the reviewer has inspected the real E1 report and issued that certificate,
bind all four gate files from outside the repository before T5:

```bash
CERT="$MEMAGENT_COSI_WORK_ROOT/logs/cosi_preflight/certificates"
export MEMAGENT_COSI_PAPER_REVIEW_SHA256=<EXTERNALLY_ISSUED_PAPER_REVIEW_64HEX>
export MEMAGENT_COSI_E0_REPORT_SHA256=<EXTERNALLY_ISSUED_E0_64HEX>
export MEMAGENT_COSI_E1_REPORT_SHA256=<EXTERNALLY_ISSUED_E1_64HEX>
export MEMAGENT_COSI_BASELINE_REPORT_SHA256=<EXTERNALLY_ISSUED_BASELINE_64HEX>
test "$(shasum -a 256 "$CERT/paper_framing_review.json" | awk '{print $1}')" = "$MEMAGENT_COSI_PAPER_REVIEW_SHA256"
test "$(shasum -a 256 "$CERT/coral_e0.json" | awk '{print $1}')" = "$MEMAGENT_COSI_E0_REPORT_SHA256"
test "$(shasum -a 256 "$CERT/coral_e1_final_report.json" | awk '{print $1}')" = "$MEMAGENT_COSI_E1_REPORT_SHA256"
test "$(shasum -a 256 "$CERT/baseline_import.json" | awk '{print $1}')" = "$MEMAGENT_COSI_BASELINE_REPORT_SHA256"
python tools/h20/audit_qwen25_7b_cosi.py \
  --run-root "$MEMAGENT_COSI_WORK_ROOT/logs/coral_e1/$MEMAGENT_COSI_E1_RUN_ID" \
  --stage research \
  --output "$CERT/research_audit.json"
```

The expected hashes must arrive through a trusted channel independent of the
files being checked; never derive them from the local certificates. They are
runtime inputs, not values stored in the manifest. A missing or substituted
gate therefore fails before trainer entry.

## 3. Fresh-base Method-T5

```bash
bash scripts/h20/run_qwen25_7b_cosi_t5.sh
```

This creates a new append-only output and starts from the Qwen2.5-7B base. CORAL is active at update 1. It never loads Original step 3 or any Original checkpoint. The run uses seed 2026, b4, n2, mini4, unchanged reward/tokenizer/data order, strict vLLM, two GPUs, audited post-update weight synchronization, and saves every step for recovery.

Expected H20 time: approximately 35–70 minutes for T5, depending on current vLLM/FSDP throughput. This is an estimate, not a measured result.

## 4. T5 fixed-S128, Original-T5 comparison, and health gate

```bash
bash scripts/h20/run_qwen25_7b_coral_s128.sh 5
bash scripts/h20/audit_qwen25_7b_coral_t5.sh
```

The evaluator loads actor weights only, performs zero optimizer steps and zero checkpoint writes, uses the same fixed 128 stable identities and deterministic recurrent protocol as the authenticated Original evaluation, independently rejoins ground truth, and rehashes the checkpoint after evaluation. The health gate requires the exact five role phases, nonzero tokens and gradients in both roles, five matching actor/vLLM digests, valid append-only ledgers, and token-F1 within two points of authenticated Original-T5. Failure preserves all evidence and blocks continuation. Only one separately committed preregistered failure-class revision is permitted.

Expected H20 time: approximately 40–90 minutes for one fixed-S128 recurrent checkpoint evaluation.

## 5. Exact resume to T25 and all anchors

```bash
bash scripts/h20/resume_qwen25_7b_cosi_t5_to_t25.sh
for step in 10 15 20 25; do
  bash scripts/h20/run_qwen25_7b_coral_s128.sh "$step"
done
bash scripts/h20/audit_qwen25_7b_coral_final.sh
```

Resume must load `global_step_5` model, optimizer, scheduler, RNG, and dataloader state. The final audit requires exactly one authenticated resume event, one role update per step 1–25, the frozen phase at every step, matching post-update vLLM digests, and checkpoint-bound S128 reports at 5/10/15/20/25. Existing run/evaluation paths are never overwritten.

Expected H20 time: approximately 2.5–5 hours for updates 6–25 plus 3–6 hours for four fixed-S128 evaluations. Retain the append-only run root on any failure; recovery starts only from an authenticated checkpoint and a new experiment variant/run ID.

## 6. Monitoring and recovery

```bash
tail -f "$MEMAGENT_COSI_WORK_ROOT/logs/coral/$MEMAGENT_COSI_RUN_ID/gate_a_execution_ledger.jsonl"
tail -f "$MEMAGENT_COSI_WORK_ROOT/logs/coral/$MEMAGENT_COSI_RUN_ID/coral_execution_ledger.jsonl"
nvidia-smi -i "$MEMAGENT_COSI_GPU_PAIR"
```

Do not kill unknown GPU processes. A lock conflict, busy device, dirty checkout, wrong SHA, manifest drift, missing gate, ledger discontinuity, checkpoint mutation, resume mismatch, or actor/vLLM digest mismatch is a NO-GO. Use a new run ID for any authorized recovery; never delete or overwrite failed evidence.

## 7. Required attribution controls after the primary run

The primary 25-rollout curve is trajectory-budget matched, but CORAL has only 13 writer-active and 12 terminal-answer-active updates. A paper claim therefore additionally requires Original-25, a random/same-mask half-exposure control, a terminal-first parity reversal, PCGrad or equivalent same-batch conflict control, and a role-exposure-matched CORAL run. Report generated tokens, reward calls, forward/backward FLOPs, weight swaps, active-role tokens, and wall time separately. These variants require independent manifests, output roots, ledgers, and commits; they may not overwrite the primary run.

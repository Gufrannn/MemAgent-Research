# Qwen2.5-7B evidence-gated idea suite

`run_7b_idea.sh` is the only training launcher. `IDEA_ARM=qa_only_original` is independently runnable. Every other arm reads one append-only evidence ledger and fails closed. Code availability is not training authorization.

NCR is the only conditional method candidate. Its ledger must certify candidate-free Shape A T0 (`P2_raw^T0` versus `+D_pre^audit`), exact linked same-write keys, real-GRPO exact-tie coverage, parameter-independent frozen readout, writer-only routing, bitwise unchanged non-tie advantage, gradient safety, and wins over QA-only plus all registered information/compute-matched baselines. T1 candidate gates may not leak into T0. “Temporal” is reserved for update-local claims.

CERC is a native-credit control and permanently rejects cross-variant grouping. Typed Boundary is a five-arm prompt diagnostic only because of the MemTX collision. Target-aligned repair is a concept placeholder with no launcher and may never stack with NCR.

Fresh runs stop at two steps. Resume is explicit to step 3. Longer anchors are only 25/50/100/200 and require `PHASE=extended CONFIRM_EXTENDED_RUN=1 EXTENDED_STEPS=...`; checkpoint and terminal rules must be frozen before metric unblinding, and step 400 is never automatic. Full RULER is not launched here: it is allowed only after the unique repair has at least three confirmed training seeds, using authoring 8 then 64 examples/task/length and a 16K→32K→64K 20% exact floor staircase.

## H20 examples

Use the repository's existing HotpotQA parquet files and a unique `EXP` for every fresh run.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRAIN=/data/MemAgent/data/hotpotqa_train.parquet
export VAL=/data/MemAgent/data/hotpotqa_dev.parquet
export MODEL=/data/models/Qwen2.5-7B-Instruct

IDEA_ARM=qa_only_original PHASE=fresh2 EXP=qa_original_s2026_fresh2 \
  bash experiments/7b_ideas/run_7b_idea.sh 2>&1 | tee qa_original_s2026_fresh2.log
IDEA_ARM=qa_only_original PHASE=resume3 EXP=qa_original_s2026_fresh2 \
  RESUME_FROM="$PWD/runs/qa_original_s2026_fresh2/global_step_2" \
  bash experiments/7b_ideas/run_7b_idea.sh 2>&1 | tee qa_original_s2026_resume3.log
python experiments/7b_gate_a/verify_gate_a.py --run-dir "$PWD/runs/qa_original_s2026_fresh2" \
  --fresh-log qa_original_s2026_fresh2.log --resume-log qa_original_s2026_resume3.log
```

NCR and its registered baselines require one adjudicated evidence ledger and one frozen per-trajectory score manifest:

```bash
export IDEA_EVIDENCE_LEDGER=/audit/frozen_evidence.jsonl
export IDEA_REWARD_MANIFEST=/audit/frozen_trajectory_scores.jsonl
export IDEA_MANIFEST_HASH=<64-hex-sha256>
export NCR_FROZEN_READOUT_HASH=<64-hex-sha256>
export IDEA_LAMBDA=0.1

IDEA_ARM=ncr_certified_routing PHASE=fresh2 EXP=ncr_s2026_fresh2 bash experiments/7b_ideas/run_7b_idea.sh
IDEA_ARM=generic_qa_aux PHASE=fresh2 EXP=generic_qa_s2026_fresh2 bash experiments/7b_ideas/run_7b_idea.sh
IDEA_ARM=generic_frozen_judge_tournament PHASE=fresh2 EXP=generic_judge_s2026_fresh2 bash experiments/7b_ideas/run_7b_idea.sh
IDEA_ARM=information_matched_raw_judge PHASE=fresh2 EXP=raw_judge_s2026_fresh2 bash experiments/7b_ideas/run_7b_idea.sh
IDEA_ARM=uniform_tie_rescue PHASE=fresh2 EXP=uniform_tie_s2026_fresh2 bash experiments/7b_ideas/run_7b_idea.sh
```

`uniform_tie_rescue` is AVSPO/DAPO-style, not an exact reproduction. CERC and Typed Boundary intentionally have no training command: CERC is a native-credit control and Typed Boundary is diagnostic-only.

OOTT is also analysis-only. Before any OOTT stratum run, validate its frozen manifest with `analysis/validate_oott_stratum_timing_20260819.py`. P strata may screen preregistered heterogeneity; O strata are descriptive support/coverage maps that must retain the unstratified total contrast; Y strata go only to RED calibration/error analysis. None can unlock training.

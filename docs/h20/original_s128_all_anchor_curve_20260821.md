# Qwen2.5-7B corrected Original fixed-S128 all-anchor curve

This Git closure evaluates one already frozen cohort and does not train a model. The six points are recurrent base `I` and corrected Original-style checkpoints at updates 5, 10, 15, 20, and 25. Every point uses the same 128 existing HotpotQA validation rows, stable evaluation identities, recurrent prompt/chunk/template protocol, and deterministic strict-vLLM decoding (`n=1`, greedy).

The primary measurements are normalized exact match and token F1, independently recomputed from terminal text and parquet ground truth with a fixed denominator of 128. Format success and historical substring exact match are diagnostics. Training dense reward is not a performance metric. The curve and paired differences describe this curated S128 challenge only; they are neither causal estimates nor unbiased HotpotQA population estimates. This is not a reproduction of the Original paper's 7B execution protocol. Published `R` remains a historical reference and is not rerun.

## Evidence and checkpoint rules

- P0 reconstructs the exact S128 identity hash `351d7e58...d12d84a` and consumes the passed stable-I r2 certificate.
- It authenticates the completed corrected Original training report and hash-chained ledger from commit `b7bf64937b5825513df86ab963816b73604f102c`.
- Checkpoint directories must be exactly `global_step_5`, `global_step_10`, `global_step_15`, `global_step_20`, and `global_step_25`; each must contain the complete seven-file world-size-2 inventory frozen by training.
- For every anchor, training actor-master, actor-effective, and both vLLM worker digests must close. Evaluation then loads exactly the two P0-frozen actor shards, restores no optimizer/scheduler/RNG/data state, and proves the same digests before the first rollout and after validation.
- P0 requires the completed prior I/T25 S128 evidence and imports only a hash-authenticated read-only reference to its I and T25 artifacts. It neither copies nor overwrites them. Missing, partial, or altered prior evidence is a hard failure; this task never silently expands into a rerun of I or Original25.
- New evidence is append-only under `$MEMAGENT_ORIGINAL_CURVE_WORK_ROOT/logs/s128_original_all_anchor_frozen_20260821`.

## H20 execution

Use the exact published commit for this branch in `MEMAGENT_ORIGINAL_CURVE_EXPECTED_COMMIT`. GPU execution is not authorized by the manifest itself. Set `MEMAGENT_ORIGINAL_CURVE_GPU_PAIR` to the exact allocated ascending pair (for example `4,7`). P0 records their indices, UUIDs, names, and visibility order, and revalidates the same physical identities throughout the run. The runner acquires the dynamic-pair protocol's per-GPU locks in ascending order, so overlapping instances of this runner, the migrated commit-retain runner, and future runners using the same contract cannot race through the idle check. Legacy fixed-pair launchers are not migrated and must not run concurrently. Non-contiguous physical indices become logical CUDA devices `0,1`; world size and tensor-parallel semantics remain unchanged.

```bash
export MEMAGENT_ORIGINAL_CURVE_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_ORIGINAL_CURVE_REPO_DIR=/data/cw/memagent_work/code/MemAgent-Research
export MEMAGENT_ORIGINAL_CURVE_EXPECTED_COMMIT=<PUBLISHED_COMMIT_SHA>
export MEMAGENT_ORIGINAL_CURVE_GPU_PAIR=4,7

cd "$MEMAGENT_ORIGINAL_CURVE_REPO_DIR"
git fetch origin \
  refs/heads/h20/qwen25-7b-original-all-anchor-s128-frozen-20260821:refs/remotes/origin/h20/qwen25-7b-original-all-anchor-s128-frozen-20260821
git switch --detach "$MEMAGENT_ORIGINAL_CURVE_EXPECTED_COMMIT"
git switch -C h20/qwen25-7b-original-all-anchor-s128-frozen-20260821 "$MEMAGENT_ORIGINAL_CURVE_EXPECTED_COMMIT"

PYTHONDONTWRITEBYTECODE=1 "$MEMAGENT_ORIGINAL_CURVE_WORK_ROOT/.venv/bin/python" \
  tools/h20/preflight_qwen25_7b_original_s128_curve.py \
  --manifest manifests/h20/qwen25_7b_original_s128_curve_seed2026.json \
  --check-runtime --write-certificate
```

Stop after P0 and inspect that its status and decision are `PASS` and `ORIGINAL_S128_CURVE_P0_PASS`. Then run the curve inside `screen`:

```bash
screen -S xinman-original-s128-curve
cd "$MEMAGENT_ORIGINAL_CURVE_REPO_DIR"
bash scripts/h20/run_qwen25_7b_original_s128_curve.sh
```

Detach with `Ctrl-A`, then `D`. The runner holds both physical-GPU locks through the idle check and the complete run, processes the six interfaces in preregistered order, and writes the final report only after all 768 terminal outcomes pass independent audit.

Read-only re-audit:

```bash
cd "$MEMAGENT_ORIGINAL_CURVE_REPO_DIR"
PYTHONDONTWRITEBYTECODE=1 "$MEMAGENT_ORIGINAL_CURVE_WORK_ROOT/.venv/bin/python" \
  tools/h20/audit_qwen25_7b_original_s128_curve.py \
  --manifest manifests/h20/qwen25_7b_original_s128_curve_seed2026.json
```

The terminal decision must be `ORIGINAL_S128_CURVE_PASS`. No result should be interpreted from a partial ledger or a failed audit.

# CORAL scientific-scope audit

**Source status:** scope-audit implementation complete locally; H20 content-intersection certificate **PENDING**.  **Scientific classification:** the T5--T25 experiment is a single-seed early-budget pilot, not a convergence or sufficient-training study.  No Method fixed-S128 result has been run in this branch.

## Training budget

The frozen primary configuration has 25 actor optimizer updates. Every update consumes four prompt groups and samples two trajectories per group: 100 prompt groups and 200 sampled training trajectories in total. `ppo_epochs=1` and the four-prompt PPO minibatch equals the four-prompt training batch, so there is one prompt minibatch per update. Dynamic token packing may create a runtime-dependent number of microbatch backward sections, but those sections are accumulated before the single optimizer step and must not be counted as additional optimizer updates.

CORAL activates memory-writer tokens on 13 odd updates and terminal-answer tokens on 12 even updates. The role mask retains Original's full-response token denominator. Exact materialized writer turns, writer tokens, and terminal-answer tokens are not statically known: the read-only audit derives turns from `rollout_seed_audit.jsonl` and token partitions from the hash-chained `coral_execution_ledger.jsonl` after a complete T25 run.

The advantage estimator is GRPO, so `ray_trainer.py` sets `use_critic=False`: critic fit updates are exactly zero. The reference policy is frozen and receives zero fit updates. The dense reward is a deterministic function rather than a learned reward model. There is no learned auxiliary model. `trainer.test_freq=-1`; the T5 health gate checks only finite loss/gradient, checkpoint closure, ledgers, and weight synchronization, so early-stopping benchmark evaluations are exactly zero.

Consequently, T25 can support only an early-budget mechanism/performance screen. It cannot support “after sufficient training,” convergence, asymptotic failure, or definitive superiority over Original. A full claim requires the frozen exposure/parity controls, at least three fresh counterbalanced seeds, an exposure-matched budget, and an untouched confirmation set.

## Data roles and leakage estimands

| Role | Frozen source | Current evidence |
|---|---|---|
| Actor training | `/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_train_32k.parquet`; SHA `798b7a2a9ece4f40884e2a9d02d165d7352df7763d1569ceaf402b45f76896f8`; method manifest `manifests/h20/qwen25_7b_cosi_seed2026.json`, SHA `c247d40f94c269ee8ce80344b2a01f766133c51ce82ebb00e5e332dad573afd7` | Exact parquet/manifest SHA frozen; per-row content inventory and intersection PENDING H20 audit |
| Critic fit | none | GRPO creates no critic; 0 rows / 0 updates |
| Prior/reference fit | none | Qwen base/reference is frozen; 0 rows / 0 updates |
| Auxiliary fit | none | no learned auxiliary component; 0 rows / 0 updates |
| Hyperparameter/method selection | synthetic E0; E1 roots from actor-training parquet; previously observed Original/Capture32 S128 results | S128 is adaptively used at benchmark level |
| Early stopping | none | `test_freq=-1`; T5 health has no S128/Original access |
| Fixed S128 | `/data/cw/memagent_work/datasets/hotpotqa/hotpotqa_dev.parquet`; SHA `54c71348875c8d535d1eebd3bb0ebdb7264297d01b3ec5d225cf8be0e9e77ff6`; resolved manifest `/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json`; file SHA `6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411`; identity SHA `351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a` | Authenticated 128-row development screen |

There is no accepted E1 selection-root inventory yet: v11 is incomplete and excluded, so its partial receipts are not promoted. The selection source remains the authenticated actor-training parquet above; the exact step-independent E1 root inventory and its receipt SHA are **PENDING a complete accepted E1 certificate**.

Selection provenance is explicit: E0 uses no HotpotQA rows and is defined by `tools/h20/coral_e0.py` (source SHA `9543ec1b797db69c0cc6e95ff90c563d5a7c3d0dc8c88b21ca14581faf300d84`); E1 draws only from the actor-training parquet at SHA `798b7a2a9ece4f40884e2a9d02d165d7352df7763d1569ceaf402b45f76896f8`, with its accepted root inventory still PENDING; the adaptive benchmark component is the stable S128 resolved manifest and identity SHA listed above. Critic, reference, auxiliary, and early-stopping sources are empty, so they have no data root or manifest SHA rather than an inferred zero-overlap file.

The overlap key is not a filename or split-local integer index. For every row, the audit hashes the exact user-question text and exact recurrent context text, then hashes that pair into a canonical content identity. It separately reports question-only and context-only intersections to expose partial reuse. The required counts are currently:

- `actor_train ∩ S128`: **PENDING H20 content audit**;
- `critic_fit ∩ S128`: **0** by absence of a critic/data source;
- `selection ∩ S128`: **128 by construction at benchmark level**, because authenticated Original/Capture32 S128 results were already available during CORAL framing. This is adaptive reuse, not direct actor-training leakage.

No statement that train and S128 are disjoint is authorized until the content certificate reports zero question+context pairs, zero question-only matches, and zero context-only matches. Any nonzero exact or partial match makes the direct-leakage gate fail.

## Adaptive-use decision and untouched confirmation

Fixed S128 is downgraded to **development screening**. Although the Method's own T5/T25 S128 outputs have not been observed, the benchmark's Original curve and Capture32 facts were used while selecting and framing the route and its success gates. It therefore cannot be called a blind final test.

The confirmation protocol must be created by an authority independent of method development after the algorithm, seeds, controls, hyperparameters, and analysis code are frozen. It must draw examples from a source not loaded by any MemAgent branch, publish only a content-addressed identity manifest and size before execution, keep prompts/labels inaccessible to implementers, run once across all frozen methods and seeds, and reveal labels only to the sealed evaluator. Its question+context inventory must have zero intersection with actor training, E1/selection data, and S128. One additional recurrent text-memory task is still required for external validity.

## Reproducible H20 certificates

Data-only scope certificate (read-only; no model load):

```bash
export MEMAGENT_COSI_EXPECTED_COMMIT=<EXACT_AUDIT_COMMIT>
export MEMAGENT_COSI_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_COSI_S128_RESOLVED_MANIFEST=/data/cw/memagent_work/logs/stable_i4x2_frozen_20260821r2/certificates/p0_resolved_manifest.json
export MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256=6c17c818fb372cf3c024504b3fa70576a6a3792203f69bf6aaf3690fdffb3411
cd /data/cw/memagent_work/code/MemAgent-Research
test "$(git rev-parse HEAD)" = "$MEMAGENT_COSI_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
SCOPE=/data/cw/memagent_work/logs/cosi_preflight/certificates/coral_scope_data_${MEMAGENT_COSI_EXPECTED_COMMIT}.json
"$MEMAGENT_COSI_WORK_ROOT/.venv/bin/python" tools/h20/audit_coral_scientific_scope.py \
  --manifest manifests/h20/qwen25_7b_cosi_seed2026.json \
  --stable-resolved "$MEMAGENT_COSI_S128_RESOLVED_MANIFEST" \
  --stable-resolved-sha256 "$MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256" \
  --work-root "$MEMAGENT_COSI_WORK_ROOT" \
  --repo-dir /data/cw/memagent_work/code/MemAgent-Research \
  --expected-commit "$MEMAGENT_COSI_EXPECTED_COMMIT" \
  --output "$SCOPE"
python -m json.tool "$SCOPE"
```

The scope entry is deliberately data-only and cannot promote a postrun budget. After a complete Method T25 run, the sole promotion path is `scripts/h20/audit_qwen25_7b_coral_final.sh` with an externally frozen `MEMAGENT_COSI_T5_P0_SHA256`. That final entry authenticates Original/S128 through the manifest authority, independently recomputes the model/tokenizer inventory and complete non-whitelist resolved-config equality, imports the 100-group semantic cursor only from the authenticated Original P0, checks all five current gate projections and the canonical GPU pair, requires full 199-parameter `dtensor` synchronization coverage on both ranks for every update, and verifies all five world-size-two checkpoint inventories. An incomplete run remains PENDING and is never promoted.

## Five-column handoff

| Scientific conclusion | Direct leakage | Adaptive benchmark risk | Paper wording correction | Remaining blockers |
|---|---|---|---|---|
| T25 is a single-seed 25-update/100-group/200-trajectory early-budget pilot; no convergence or sufficient-training claim. | Actor-train/S128 exact content overlap PENDING H20; critic/prior/aux overlap is zero because none is fit. | Confirmed: S128 is a development screen because Original/Capture32 S128 facts were observed during method framing. | Abstract, Introduction, experiment table, training paragraph, and runbook now say early-budget pilot, report role exposure, and prohibit blind-final/convergence language. | H20 overlap certificate; complete E1 after storage remediation and new reviewed run ID; Method T25; actual budget certificate; controls, ≥3 seeds, untouched confirmation set, external task. |

## Current operational evidence

E1 v11 at source commit `e4f897a2ee9e9e2cfd50916eaca181d628661893` is **NOT RELEASE**. It produced proposal receipts 1/3/5 and checkpoints through step 6, then stopped with `/data` at 100% capacity. It cannot be resumed or reused. This failure does not provide a T25 budget or performance result. A replacement E1 remains blocked until storage capacity for the full evidence inventory is measured and provisioned and a new run identity/source tip is independently reviewed.

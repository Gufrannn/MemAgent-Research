# Cross-Occupancy Update Certificates for Recurrent Text-Memory Policies

**Route status (2026-08-22): MERGE / STANDALONE NO-GO.** This document preserves the preregistered frame and negative design result; it is not a report of positive COSI experiments. Independent review exposed an incompatibility between a meaningful simultaneous certificate and the frozen compute/acceptance contract. The crossed replay remains a candidate diagnostic for a broader recurrent-policy-optimization paper, conditional on E1, but this branch does not authorize T5 training.

## Abstract (204 words)

Reinforcement learning can train a language model to repeatedly overwrite a compact textual memory while reading a long stream and then answer from the resulting state. Such a model is both the writer of future observations and the policy that consumes them. Consequently, an update that improves actions sampled under the old memory occupancy need not improve the updated policy in its self-induced closed loop. We study whether this mismatch is measurable and actionable in recurrent text-memory policies. We propose **Cross-Occupancy Update Certificates (COSI)**, an outer loop around a provisional GRPO update. On a preregistered held-out transport batch, COSI caches writer candidates from the old and provisional checkpoints once, then evaluates each candidate with old and provisional continuations under common future random seeds. The resulting 2×2 design exactly decomposes the measured checkpoint-level return change into a writer substitution term, a continuation substitution term, and their interaction. A root-clustered lower confidence bound accepts the provisional state, triggers preregistered backtracking, or restores a bitwise-authenticated model, optimizer, scheduler, scaler, and RNG snapshot. COSI does not claim a distribution-free global safety guarantee: its certificate is conditional on the frozen transport distribution and replay contract. We first test exact decomposition and rollback in a recurrent toy process, then test whether real adjacent Original checkpoints exhibit transport residuals beyond numerical noise and ordinary KL. Only if these gates pass will we compare COSI with compute-matched GRPO, KL backtracking, and lagged-policy baselines from a fresh common base. All empirical improvements remain placeholders pending authorized experiments.

## 1. Introduction

Long-context language agents increasingly replace an ever-growing interaction history with a compact state that is repeatedly read and rewritten. MemAgent is a particularly clean instance: a language model reads a document in chunks, overwrites a bounded free-text memory after each chunk, and finally answers from the materialized memory. Its multi-conversation reinforcement-learning procedure makes the entire recurrent workflow trainable without changing the transformer architecture. This design is attractive precisely because the memory is ordinary text. It also creates a policy-optimization problem that is easy to miss when the training trajectory is flattened into tokens.

At writer turn (t), the policy emits a text (M_{t+1}). That text is not merely an action whose consequence is supplied by an external stationary simulator; it becomes part of the next prompt. The same parameter vector then consumes the state it just wrote, emits another state, and ultimately answers. Updating the shared model therefore changes two linked mechanisms at once: which memory texts are produced and how any fixed memory text is interpreted downstream. A local GRPO or PPO surrogate computed on trajectories from the old checkpoint can be positive even when the provisional model performs worse after it rolls into the memory occupancy that it induces for itself.

This observation alone is not new safe policy improvement. Classical conservative and trust-region policy iteration derive performance lower bounds by controlling policy divergence. TRPO optimizes an old-state surrogate under a trust region; PPO approximates this strategy with clipped objectives. High-confidence policy improvement accepts policies using statistical lower bounds, and SPIBB constrains deviations in poorly supported regions. More recent sequence-level objectives such as GSPO change the likelihood-ratio unit for language-model training. These methods are essential baselines. They do not, however, directly expose the two roles played by one recurrent text model at an update boundary: *writer policy* and *continuation policy*. Nor do they produce a replay receipt showing whether a negative update arose from changed memories, changed consumption of fixed memories, or their interaction.

We therefore frame COSI as an **update certificate**, not as a universal safety theorem. For each provisional update from checkpoint (O) to (N), we freeze a small outcome-blind transport set of stable roots. We generate old-writer and new-writer candidate texts exactly once and authenticate their token-level hashes. For every cached candidate, we replay the remaining recurrent computation with either old or new weights. Common future seeds are derived from the stable root, candidate identity, turn, and replica; no cell may resample the writer candidate. This yields four potential outcomes,

\[
Y_{OO},\;Y_{NO},\;Y_{ON},\;Y_{NN},
\]

where the first index selects the writer checkpoint and the second selects the continuation checkpoint. Their factorial contrast separates writer substitution under the old continuation, continuation substitution for old-writer states, and a writer-by-continuation interaction. The algebra is exact for the cached replay population. It is deliberately a checkpoint-level transport diagnostic, not a learned predictor of whether an individual candidate should be committed. In particular, it does not revive the rejected candidate-level Paired Effect route.

The distinction matters scientifically. If adjacent Original checkpoints show that (Y_{NN}-Y_{OO}) is numerically indistinguishable from the old-occupancy writer contrast, or if a standard KL statistic fully explains every residual, COSI has no independent object and the route stops. If, instead, continuation and interaction terms are stable across roots and materially contribute to sign reversals, the same certificate provides a natural optimizer outer loop. A provisional GRPO step is held outside the durable checkpoint namespace. COSI computes a root-clustered lower confidence bound for the closed-loop contrast on held-out transport roots. It accepts when the preregistered criterion passes, tries only the preregistered damped steps when it does not, and otherwise restores the exact pre-update state. Acceptance never queries the frozen S128 performance evaluation.

The engineering contract is part of the estimand. A four-cell aggregate without candidate hashes does not identify a crossed comparison because candidate resampling confounds writer substitution with Monte Carlo variation. Different future seeds inflate cell differences with continuation noise. Weight swapping without full checkpoint hashes can silently mix policies. Restoring model weights without optimizer moments, scheduler, gradient scaler, dataloader cursor, and random generators is not rollback. COSI therefore treats all of these as authenticated evidence and fails closed when any receipt is missing.

The proposed paper story has four layers. First, we formalize a shared-parameter recurrent policy and give a finite counterexample where an old-occupancy writer improvement coexists with a negative closed-loop update. Second, we provide an exact 2×2 decomposition on a cached replay population and state the assumptions under which a clustered lower bound covers its target mean. Third, we implement a reversible propose–cross–accept optimizer with append-only receipts. Fourth, conditional on the mechanism gates, we evaluate fresh training from the same Qwen2.5-7B base against the authenticated corrected Original curve and compute-matched alternatives at updates 5, 10, 15, 20, and 25.

Our intended contributions are:

1. **Problem and counterexample.** We isolate role coupling in recurrent text-memory RL: one parameter update changes both the distribution of materialized future states and the continuation applied to those states. We construct a minimal reversal that survives a positive old-continuation writer contrast.
2. **Identification.** We introduce an authenticated cached-writer × crossed-continuation design. It exactly decomposes the measured checkpoint transition into writer, continuation, and interaction contrasts and localizes occupancy reversal without fitting a candidate-selection model.
3. **Algorithm and systems contract.** We wrap an otherwise unchanged provisional GRPO update with a held-out root-clustered certificate, preregistered backtracking, and bitwise rollback of all training state. Every accepted or rejected transition is connected to weight, candidate, seed, and checkpoint hashes in an append-only ledger.
4. **Conditional empirical study.** If E1 validates the mechanism, we will measure a reversal atlas over training, compare against KL/backtracking and lagged-policy controls at matched total rollout compute, and test whether fewer certified reversals translate into better multi-anchor and variable-tracking robustness. No positive result is asserted before these experiments.

The framing also specifies its failure boundary. COSI is not warranted if real transport residuals are absent, if ordinary KL predicts them completely, if acceptance degenerates to always accept or always reject, if the extra cost exceeds three times Original without mechanism signal, or if improvements disappear under compute matching. In any of those cases, the appropriate scientific output is a negative diagnostic or a merge into a broader optimizer study, not a renamed method paper.

## 2. Problem formulation

### 2.1 Two distinct estimands (reframed after independent review)

The paper must not conflate a local boundary intervention with a full policy transition. We therefore register two separate quantities.

The **full-update contrast** is \(\Delta^{full}(x)=R(\tau_N(x))-R(\tau_O(x))\), where \(\tau_O\) is generated from a fresh root entirely with checkpoint O and \(\tau_N\) entirely with checkpoint N. This is the only quantity eligible for an update acceptance certificate. Old and new full trajectories use a preregistered common-random-number coupling, but they need not share materialized memories.

The four-cell quantity below is a **boundary-local crossed replay contrast**. A root for this estimand is not a raw question: it is an authenticated materialized prefix \(z_t=(x,M_t,t)\) produced under a separately declared prefix policy (for E1, old-checkpoint O). The writer at boundary \(t\) is switched, its candidate is cached, and the remaining suffix is replayed under O or N. It diagnoses how an update acts at old-prefix support. It is never called the full closed-loop return. Results are stratified by boundary turn; aggregating boundaries does not turn them into \(\Delta^{full}\).

Let a root (x) contain a question and an ordered evidence stream. At a designated writer boundary, checkpoint (w\in\{O,N\}) samples a candidate memory (m_w=g_w(x;u_w)). A continuation checkpoint (c\in\{O,N\}) consumes the *fixed cached candidate* and the remaining evidence under future randomness (v), producing bounded terminal score

\[
Y_{wc}(x,u_w,v)=R\!\left(h_c(x,m_w,v)\right)\in[a,b].
\]

The transport estimand is defined only over a preregistered materialized-prefix distribution, writer replicas, and common-future-seed coupling. It is not the S128 screening metric and does not reuse training-batch outcomes for acceptance.

For each authenticated replay unit define

\[
W_O=Y_{NO}-Y_{OO},\qquad
C_O=Y_{ON}-Y_{OO},\qquad
I_{WC}=Y_{NN}-Y_{NO}-Y_{ON}+Y_{OO},
\]

and \(\Delta_{cl}=Y_{NN}-Y_{OO}\). Roots, rather than cells or replicas, are the independent uncertainty clusters. Root-level contrasts are computed before confidence bounds.

### Proposition 1 (cached factorial identity)

For every replay unit whose four cells share the authenticated root contract and whose two cells for each writer checkpoint use the identical cached candidate, the identity

\[
\Delta_{cl}=W_O+C_O+I_{WC}
\]

holds exactly over real arithmetic. A machine implementation must close within a preregistered floating-point tolerance and must reject missing cells, duplicate cells, candidate-hash disagreement, weight-hash disagreement, or future-seed disagreement.

This is an algebraic identification statement, not a causal generalization claim. Its value is that each term is measured on the same finite replay support rather than inferred from unrelated aggregate evaluations.

### Proposition 2 (a permitted provisional update can reverse)

There exists a two-stage recurrent process with a shared two-parameter stochastic policy and bounded reward for which one explicit clipped-PPO/GRPO gradient step has positive sampled surrogate improvement, obeys a preregistered likelihood-ratio clip/KL aperture, and nevertheless satisfies \(\mathbb E[W_O]>0\) and \(\mathbb E[\Delta^{full}]<0\). E0 must enumerate the finite state/action probabilities, sampled group advantages, gradient and step size with rational or interval arithmetic. An arbitrary O/N pair is insufficient. The construction will also show which approximation used by practical PPO breaks the premises of TRPO's theoretical monotonic-improvement result; it will not claim a contradiction with exact TRPO.

This proposition is intentionally narrower than the general performance-difference lemma: it demonstrates why the writer-only crossed contrast is insufficient in the shared text-memory setting. It does not claim that TRPO's exact theoretical procedure would accept the bad update.

### Proposition 3 (finite-family conditional full-update certificate)

Before training, the protocol partitions an outcome-blind root inventory into 25 disjoint update blocks, with two disjoint backtracking sub-blocks inside each update. A root is queried for an acceptance outcome at most once. Assume roots are independent draws from the frozen training-transport distribution, root-level full-update contrasts are bounded in \([a-b,b-a]\), and all randomness follows the committed seed coupling. A simultaneous one-sided Hoeffding bound with total family-wise error \(\alpha_{FW}=0.05\) allocates \(\alpha_{u,k}=0.05/(25\times2)\) to update \(u\), attempt \(k\). COSI accepts only if

\[
\operatorname{LCB}_{1-\alpha_{u,k}}(\mathbb E[\Delta^{full}])\ge 0.
\]

No root reuse, IID-cell inference, post-hoc bootstrap, or adaptive change to the proposal family is permitted. This conservative protocol avoids claiming that ordinary alpha spending repairs reusable-holdout overfitting. Coverage does not extend to a changed root distribution, sampler, reward, horizon, or model family. A tolerance \(\delta>0\), if studied as an ablation, is called a *non-inferiority tolerance* and must report the worst cumulative budget \(\sum_u\delta_u\); it is not the primary improvement certificate.

### Proposition 4 (captured-state rollback contract)

If serialization captures every mutable training component and no external worker retains unversioned weights, hashing the canonical byte representation before proposal and after rollback proves equality of the *captured state*. The required state includes model parameters and buffers, optimizer tensors and param-group metadata, scheduler, mixed-precision scaler, framework and device RNG states, dataloader cursor, global step, and versioned rollout-worker namespace. A model-only restore is failure. Equality of subsequent distributed execution is a separate empirical determinism test and may not be claimed bitwise unless NCCL, CUDA, vLLM kernels, hardware and all determinism flags are themselves certified.

## 3. Closest-work claim matrix and novelty boundary

| Primary source | What it solves | What it does not establish for this paper | Consequence for COSI |
|---|---|---|---|
| [MemAgent (Yu et al., 2025/ICLR 2026)](https://arxiv.org/abs/2507.02259) and [official project/code](https://memagent-sialab.github.io/) | End-to-end RL for repeated free-text overwrite memory and long-context extrapolation. | Does not report checkpoint-crossed writer/continuation outcomes or reversible accept/reject updates. | Supplies the recurrent shared-policy setting and Original implementation, not novelty for RL memory itself. |
| [MEM1](https://mit-mi.github.io/mem1-site/) | Constant-memory long-horizon agents with end-to-end RL and dynamic consolidation. | Does not isolate update-boundary role coupling with a cached 2×2 checkpoint design. | External memory-RL neighbor; COSI must generalize beyond one Hotpot protocol if it survives. |
| [Unified Memory Agent](https://arxiv.org/abs/2602.18493) | Joint memory operations and answering under one policy across several memory tasks. | No reported high-confidence checkpoint acceptance based on crossed materialized states. | Strongest recent shared-policy memory neighbor; motivates an external Ledger-QA-style test. |
| [MemoPilot](https://openreview.net/forum?id=gNWNtstp3r) | Trains a memory updater against later performance while freezing the task agent. | Separates updater and consumer, so it does not study a single update changing both roles. | Useful negative control: role coupling should weaken when continuation is frozen. |
| [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347), [GRPO/DeepSeekMath](https://arxiv.org/abs/2402.03300), and [DAPO](https://arxiv.org/abs/2503.14476) | Practical on-policy surrogate optimization; GRPO removes the critic; DAPO stabilizes scalable LLM RL. | A positive minibatch surrogate is not a direct measurement of the provisional policy under its self-induced future text states. | COSI leaves the inner Original update unchanged and audits its closed loop. |
| [GSPO](https://arxiv.org/abs/2507.18071) | Sequence-level importance ratios/clipping better aligned with sequence rewards. | Changes action granularity but does not cross writer and continuation checkpoints on fixed materialized states. | Mandatory baseline/analysis neighbor; if sequence KL/ratio explains residuals, COSI is redundant. |
| [TRPO](https://proceedings.mlr.press/v37/schulman15.html) | Monotonic-improvement bound from an old-state surrogate plus policy-divergence control. | Practical bounds may be loose; it does not provide the proposed role-specific forensic decomposition. | COSI must not claim generic monotonic improvement; KL/backtracking is a required matched baseline. |
| [Safe Policy Iteration](https://proceedings.mlr.press/v28/pirotta13.html) and [HCPI](https://proceedings.mlr.press/v37/thomas15.html) | Lower-bound policy selection and high-confidence acceptance. | Do not specialize the evaluation design to cached free-text writer states and shared continuations. | LCB acceptance is inherited machinery, not a novelty claim. Novelty can only lie in the estimand/replay design and evidence. |
| [SPIBB](https://proceedings.mlr.press/v97/laroche19a.html) and [Decision-Point RL](https://proceedings.mlr.press/v258/sharma25a.html) | Restrict changes in uncertain state-action regions for batch safe improvement. | Operate from fixed offline support and do not authenticate checkpoint-crossed generative replay. | Establish a high bar for the word “safe”; COSI uses “certificate” and states conditional coverage. |
| [Taylor Expansion Policy Optimization](https://proceedings.mlr.press/v119/tang20d/tang20d.pdf) | Higher-order performance expansion exposes interactions across time steps. | Does not provide the concrete cached writer × continuation factorial audit for text memory. | Theory section must compare the 2×2 interaction to higher-order policy terms and avoid claiming interaction itself is new. |
| [Doubly Robust OPE](https://proceedings.mlr.press/v48/jiang16.html) and [multiple-policy high-confidence evaluation](https://proceedings.mlr.press/v206/dann23a.html) | Off-policy value estimation and simultaneous confidence reasoning. | COSI proposes direct controlled replay, not importance-weighted reuse of unrelated trajectories. | OPE baselines are relevant if direct replay becomes too costly; LCB multiplicity must account for repeated proposals. |

**Novelty boundary.** The defensible residual is the conjunction of (i) one free-text policy generating materialized future prompts and consuming them after the same parameter update, (ii) cached candidate × crossed checkpoint replay that exactly attributes a measured update to writer, continuation, and interaction terms, and (iii) a fully reversible optimizer that uses this authenticated diagnostic online. Factorial algebra, common random numbers, lower confidence bounds, rollback, safe policy improvement, and recurrent memory RL are individually prior art. If E1 cannot demonstrate that their conjunction reveals behavior missed by KL/sequence-ratio diagnostics, this route is not a standalone top-conference paper.

## 4. Method and ablation map

1. **Propose.** Run the exact Original GRPO update into a provisional namespace. Snapshot all mutable state before the update.
2. **Cache writers.** On frozen held-out roots, generate one old and one provisional candidate per registered replica with independent writer seeds. Candidates are immutable content-addressed token arrays.
3. **Cross continuations.** Evaluate each cached writer candidate under both continuation checkpoints. The corresponding old/new continuation cells receive an identical common future seed schedule.
4. **Validate.** Require four-cell support, exact root/candidate/weight/seed hashes, bounded independently recomputed reward, and decomposition closure.
5. **Cluster and decide.** Use a fresh disjoint root block to estimate the full O-versus-N trajectory contrast. Accept only at LCB\(\ge0\), try the next preregistered dampening factor on its disjoint sub-block, or rollback. The crossed boundary decomposition controls a preregistered *precision allocation*: boundaries with large continuation/interaction variance receive more of a fixed diagnostic replay budget, but it cannot override a failing full-update bound. A direct full-contrast-only variant tests whether decomposition adds value.
6. **Commit evidence.** Append a hash-chained receipt before making the accepted namespace durable. Rejection records both pre/post restoration hashes.

Every optimization maps to an ablation: caching versus illegal resampling; common versus independent future seeds; interaction-aware versus additive-only acceptance; root clustering versus invalid cell-IID uncertainty; full-state versus model-only rollback; exact four-cell versus aggregate-only score differences; transport-batch size and sequential bounds; cache/weight-swap engineering with identical estimand.

## 5. Experimental plan

### E0: exact toy and systems closure

| Artifact | Required result | Failure decision |
|---|---|---|
| Rational recurrent counterexample | \(W_O>0\), \(\Delta_{cl}<0\), exact four-term closure | NO-GO theory if no non-degenerate shared-policy construction |
| Hash/support adversarial suite | Reject missing/duplicate cell, wrong candidate, seed, weight, reward, or root | BLOCKER |
| Root-clustered bound tests | Coverage simulation and rejection of cell-IID pseudo-replication | BLOCKER |
| Full-state rollback | Byte-identical state hashes plus identical next update/output | BLOCKER |
| Original-off regression | Disabled COSI is behaviorally identical to Original path | BLOCKER |

### E1: frozen adjacent-checkpoint transport residual

Use authenticated adjacent corrected Original checkpoints (5→10, 10→15, 15→20, 20→25; optionally 0→5 only if the exact base interface closes). Before any outcome, freeze 32 stable roots per transition, two writer replicas, the middle eligible writer boundary, and all seeds. Duplicate deterministic replays on eight roots define a numerical/replay noise ceiling \(\epsilon_{noise}\); any hash divergence is infrastructure failure. Define root-level reversal as \(W_O>0\) and \(\Delta_{local}<0\). E1 mechanism PASS requires all of: (i) the one-sided 95% root-clustered lower bound on reversal proportion exceeds 0.10 for at least two adjacent transitions; (ii) either continuation or interaction has absolute standardized root effect at least 0.30 and exceeds \(5\epsilon_{noise}\); and (iii) after fixed ridge regression removes token KL, sequence KL, candidate length and turn effects from the continuation or interaction term itself, the leave-one-transition-out standardized residual effect remains at least 0.30 with multiplicity-adjusted \(p\le.05\). C/I are never used to predict their algebraic sum \(W+C+I\). Failure yields E1 NO-GO/MERGE. Five-update anchor gaps are mechanism screening only; an online route would additionally require E1b single-update provisional snapshots. No long training occurs before this result.

### Fresh-training tables, conditional on E0/E1 GO

| Table | Rows | Columns / primary comparisons |
|---|---|---|
| Main multi-anchor curve | Original, COSI | T5/10/15/20/25 normalized EM, token-F1, format; paired fixed-S128 deltas |
| Compute-matched methods | Original+unused audit rollouts, KL/backtracking, target-network lag, COSI | total generated tokens, wall time, accepted updates, reversal rate, F1 |
| Identification ablations | no interaction, aggregate-only, resampled candidates, independent future seeds | bias/variance, decomposition failures, decisions changed |
| Decision ablations | random accept-rate matched, additive-only LCB, different preregistered precision levels | acceptance rate, failed updates, performance |
| Mechanism atlas | checkpoint transition × writer turn/horizon | \(W_O,C_O,I_{WC},\Delta_{cl}\), KL, reversal sign |
| External validity | HotpotQA and fixed RULER Variable Tracking mini-suite | state retention, closed-loop reversal, final performance |
| Negative/boundary | frozen continuation (MemoPilot-like), short horizon, low-KL transitions | when interaction disappears and COSI should add no value |

Primary T5 health requirements remain: acceptance strictly between 20% and 80%, rollback/resume closure, total compute matching, and fixed-S128 token-F1 no worse than authenticated Original-T5 by more than 2 points. T25 success requires the frozen program's effect gate; multi-seed and untouched confirmation require separate authorization.

## 6. Failure criteria and current decision

The route is `NO-GO` as a standalone paper if any of the following holds: E0 cannot produce a non-degenerate role-coupled reversal; rollback is not bitwise; E1 residuals are numerical noise; KL/sequence diagnostics explain the residual completely; direct replay exceeds the 3× cost ceiling without mechanism signal; T5 accepts nearly everything or nothing after the single preregistered precision revision; compute-matched performance fails; or the external task shows no role-coupled failure axis. A diagnostic-only E1 may be merged into a broader recurrent-policy optimization paper, but it must not be marketed as COSI safe improvement.

Current decision: **MERGE / STANDALONE NO-GO**. With 25 updates, two attempts, family-wise error 0.05 and bounded return differences in [-1,1], Bonferroni gives per-attempt alpha 0.001. A Hoeffding certificate with 32 independent roots has half-width

\[
2\sqrt{\log(1000)/(2\cdot32)}\approx0.657.
\]

Reducing this to 0.05 requires at least \(\lceil 4\log(1000)/(2\cdot0.05^2)\rceil=5527\) independent roots per attempt. This violates the 3x compute ceiling and makes the required 20--80% T5 acceptance aperture implausible. Replacing the bound with an unvalidated heuristic would remove the central certificate claim. Therefore T5 is permanently unauthorized on this standalone branch. E0 and E1 tooling is retained as a fail-closed diagnostic research closure; a future merged route would need a new estimand, power analysis, and authorization.

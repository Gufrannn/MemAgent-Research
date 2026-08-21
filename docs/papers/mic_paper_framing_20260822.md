# Paper II framing: Materialized Memory Innovation Credit

Status: **REFRAME; conditional KEEP only if E0 and E1 pass.** This document is a
preregistration and paper skeleton, not a statement of positive MIC results.

## Working title

**Credit at the State Boundary: Leakage-Audited Innovation Critics for
Recurrent Text Memory**

## Abstract (205 words)

Recurrent language agents repeatedly overwrite a free-text memory before
answering a question. Existing outcome-based training commonly broadcasts one
terminal advantage to every writer and answer token, even though different
memory writes can preserve, delete, or introduce evidence. A natural response
is to predict terminal return after each write and use successive prediction
differences as turn-level credit. That idea alone is not novel: it is closely
related to return decomposition and actor--critic methods. We instead study a
narrower, falsifiable question: can credit be estimated at the exact boundary
where a text memory becomes the next recurrent state, without allowing future
text or outcomes to leak into the estimator? We formulate materialized memory
states as a filtration and define writer innovations as differences between
cross-fitted conditional-value predictions; final-answer tokens receive only
the terminal residual. The estimator is accompanied by stable-example folds,
out-of-fold calibration, a feature firewall, role-specific gradient receipts,
independent critic checkpoints, and trajectory-wise telescoping checks at
machine precision. We state conditions under which the oracle increments are
valid control variates and characterize approximation-error terms; we do not
claim that Doob decomposition itself is new. Before training, an oracle toy MDP
and frozen Original trajectories must show exact closure, absence of leakage,
and predictive value beyond turn/length baselines. If these gates pass, we will
compare fresh-from-base training under an identical budget at five checkpoints;
otherwise the method is rejected. Results are intentionally left pending.

## 1. Introduction

Long-context memory agents convert an ever-growing document into a sequence of
bounded text states. In MemAgent, a writer reads the question, the previous
memory, and the next chunk, then emits a complete replacement memory. After the
last rewrite, the same model produces a final answer. This makes every writer
completion both a language action and the materialized state presented to the
next call. The training signal, however, is typically much coarser: one terminal
outcome is normalized across sampled trajectories and broadcast across all
writer and final-answer tokens.

Broadcast is unbiased as a trajectory-level score-function estimator under the
usual assumptions, but it does not distinguish which rewrite changed the
reachable terminal return. An early write that preserves a bridge entity, a
late write that deletes it, and an answer that compensates by guessing may all
receive the same scalar. The practical concern is therefore not that terminal
policy gradients are intrinsically invalid. It is that their writer component
can have high variance and can couple early memory updates to stochasticity in
later writing and answering. Whether this is a material problem in the frozen
MemAgent setting is an empirical question, not a premise.

A tempting solution is to learn a return predictor on prefixes and assign the
difference between consecutive predictions to consecutive decisions. This is
not, by itself, a sufficient research contribution. RUDDER already formalized
return redistribution and operationalized differences in return predictions;
standard actor--critic methods use conditional values as baselines; hindsight
and counterfactual credit methods condition on future events; recent multi-turn
LLM-agent systems introduce turn-level advantages or hierarchical transition
interfaces. The mathematical identity behind a Doob martingale is classical.
Calling a prefix-prediction difference a new credit theorem would overclaim.

The residual problem is specific to recurrent free-text state construction.
First, the state boundary is often ambiguous in token-concatenated training:
the relevant object is the exact post-write text that is subsequently consumed,
not a hidden activation, pre-write prompt, or retrospective summary. Second,
return prediction is unusually vulnerable to target leakage. A training row may
quietly contain the gold answer, a future chunk, the current row's outcome, or
the generated final answer. Non-cross-fitted fitting can also memorize stable
examples and manufacture apparently sharp turn credit. Third, writer and answer
tokens share model parameters, so a claimed separation of credit is meaningful
only if the actual optimizer tensors and gradients demonstrate role-specific
delivery. Fourth, an approximate decomposition can silently change the total
trajectory signal unless the residual is constructed algebraically and checked
per trajectory.

We therefore investigate **Materialized Memory Innovation Credit (MIC)** as a
leakage-audited estimator and training protocol. Let the admissible information
after writer turn `t` include only the question, chunks through `t`, the exact
materialized memory, and turn metadata. A critic predicts terminal return from
that information. Stable example IDs are assigned to deterministic folds; a
trajectory receives predictions only from a critic whose training set excludes
its stable example. Writer turn `t` receives the stopped-gradient difference
between the post-write value and its admissible pre-write value. Final-answer
tokens receive the terminal residual. The residual is defined from the same
predictions so that the trajectory ledger closes to the observed terminal
return to floating-point tolerance.

This framing makes two important concessions. MIC does not add a dense task
reward, and its increments are not claimed to be causal effects of individual
words. They mix the value of newly observed evidence and the action-induced
memory transition under the chosen filtration. Nor does telescoping guarantee a
good critic: a useless constant critic can telescope perfectly. Predictive
feasibility, conditional-mean diagnostics, calibration, variance, and leakage
tests are separate gates. Cross-fitting limits in-sample memorization but does
not make a misspecified value estimator unbiased.

The paper is designed around a sequence of falsifiable tests. E0 uses a finite
toy MDP with an analytic conditional value to recover exact increments, checks
shuffle and fold exclusions, and requires machine-precision closure. E1 uses
frozen Original trajectories and outcomes, never retraining Original. The OOF
critic must outperform a critic restricted to turn and length, remain calibrated
by fold and turn, pass martingale conditional-mean diagnostics, and prove that
no held-out stable ID contributed its outcome to training. Only then can a fresh
Method-T5 run be formed from the same Qwen2.5-7B base and the accepted Original
manifest, with only MIC fields differing.

Conditional on those gates, the empirical study will compare terminal broadcast,
turn/length baselines, a conventional scalar value critic, non-cross-fitted
diagnostic fitting, shuffled states, MIC without answer-residual separation, and
the full method. All methods use the same data order, samples, reward, tokenizer,
rollout budget, and checkpoints. Performance is independently recomputed on the
frozen S128 predictions using normalized exact match, token F1, and format
validity; critic losses and dense diagnostics are never reported as task
performance. External validity is tested on a small frozen variable-tracking
suite that stresses early deletion and late recovery.

Our intended contributions are therefore conditional rather than asserted:

1. A precise state-boundary formulation for recurrent free-text writers and a
   proposition separating exact algebraic closure from value-estimation error.
2. A cross-fitted, leakage-audited role-specific estimator whose predictions,
   checkpoints, folds, and optimizer delivery can be independently reconstructed.
3. A credit stress suite and frozen-trajectory feasibility protocol that can
   reject the method before expensive training.
4. If and only if the gates pass, a same-budget five-anchor study of whether
   materialization-aligned credit improves performance or writer-gradient
   variance. Numerical results remain `[PENDING E0/E1/T5/T25]`.

## 2. Problem formulation and propositions

For trajectory `i`, let `Q_i` be the question, `C_i,1:t` the chunks revealed by
writer turn `t`, `M_i,t` the exact bytes/token sequence emitted by that turn and
subsequently consumed, `A_i` the generated final answer, and `R_i` the frozen
terminal reward. Define the admissible sigma-field

```text
F_i,t = sigma(Q_i, C_i,1:t, M_i,0:t, public turn metadata, rollout randomness
              already realized through t).
```

The implementation retains all `C_i,1:t` and `M_i,0:t`; it never substitutes
only the latest overwritten memory. `F_i,t` explicitly excludes
`C_i,t+1:T`, `A_i`, gold answers, and `R_i`. Let
`V_i,t = E[R_i | F_i,t]`. The initial value `V_i,0` is evaluated before the first
writer action using only its admissible pre-write state. Define

```text
D_i,t = V_i,t - V_i,t-1
E_i = R_i - V_i,T.
```

**Proposition 1 (oracle closure and innovation).** For integrable `R`,
`V_t = E[R|F_t]` is a martingale with respect to the increasing admissible
filtration. Hence `E[D_t|F_t-1]=0` and
`R = V_0 + sum_t D_t + E` almost surely. This is a direct application of a
classical conditional-expectation identity and is not claimed as novel.

**Proposition 2 (post-decision control-variate interpretation).** If the chunk
arrival at turn `t` is exogenous to the writer policy and the materialized memory
is the complete policy-controlled transition variable consumed downstream, then
`V_t - E[V_t|F_t-1]` is a zero-mean control variate at the pre-write information
set. Additional assumptions are required to interpret it as the causal effect
of the writer action; MIC makes no unconditional causal claim.

**Proposition 3 (approximation ledger).** For OOF predictions
`Vhat_t = V_t + e_t`, defining `Dhat_t=Vhat_t-Vhat_t-1` and
`Ehat=R-Vhat_T` preserves exact telescoping algebraically. The local writer
credit error is `e_t-e_t-1`, while the final residual error is `-e_T`.
Therefore closure alone cannot validate the critic; E1 must separately bound
OOF prediction error, calibration error, and conditional innovation means.

**Proposition 4 (group exclusion).** Under deterministic stable-ID folds, if
each fold model is fit only on other folds and immutable training receipts bind
the exact stable IDs and outcomes, no trajectory's own outcome contributes to
its OOF prediction. This guards direct self-outcome fitting, not distributional
dependence between related examples; stable-root grouping is required wherever
multiple rows share a root.

**Proposition 5 (finite-sample gradient discrepancy).** Let `S_t` be the
score-function sum over writer turn `t`, assume `E[||S_t||^2] <= G_t^2`, and
let the fold-exclusive value error satisfy
`E[(Vhat_t-V_t)^2] <= eps_t^2` on held-out roots. The discrepancy between the
oracle writer innovation gradient and MIC's held-out gradient obeys

```text
|| E[S_t (Dhat_t-D_t)] ||
<= G_t sqrt(E[(e_t-e_t-1)^2])
<= G_t (eps_t + eps_t-1).
```

The first inequality is Cauchy--Schwarz and the second is Minkowski. For final
answer score `S_A` with second-moment bound `G_A`, the corresponding discrepancy
is at most `G_A eps_T`. With disjoint writer/answer masks, summing these bounds
gives `sum_t G_t(eps_t+eps_t-1)+G_A eps_T`. Cross-fitting makes each evaluated
root independent of its own fitted outcome conditional on the training folds;
it does not make `eps_t` zero or remove shared-root dependence, hence folds are
root-grouped. Token normalization multiplies each term by its declared role
normalizer and is recorded in the gradient ledger. This proposition is a bound,
not a claim that an approximate critic is unbiased.

**Counterexample 1 (why current overwrite alone is not a filtration).** If
`M_1` contains a bit and `M_2` overwrites it with a constant, then
`sigma(M_1)` is not contained in `sigma(M_2)`. MIC therefore stores the complete
arrived-chunk history and every materialized memory `M_1:t` in the critic state;
using only the current overwrite is an invalid ablation and cannot support the
martingale claim.

## 3. Closest-work claim matrix and novelty boundary

| Work | What it solves | What it does not establish for this setting | MIC residual, if E1 passes |
|---|---|---|---|
| MemAgent (Yu et al., 2025; official paper/code) | Recurrent overwrite memory trained end-to-end with multi-conversation RL | Does not identify per-materialized-write credit or an OOF value firewall | Exact post-write state boundary and role-delivery audit |
| RUDDER (Arjona-Medina et al., NeurIPS 2019; official code) | Return-equivalent reward redistribution; differences of return predictions | Does not address free-text state materialization, stable-ID leakage, or shared writer/answer delivery | Audit protocol and application-specific filtration, not return decomposition itself |
| Hindsight Credit Assignment (Harutyunyan et al., NeurIPS 2019) and Counterfactual Credit Assignment (Mesnard et al., ICML 2021) | Credits past actions using future-event-conditioned likelihood/value constructions | Future conditioning is deliberately inadmissible to the MIC critic; different estimator and target | Outcome-blind alternative with explicit firewall; may trade power for validity |
| Quantile Credit Assignment (Mesnard et al., ICML 2023) | Luck-conditioned baselines with unbiased-gradient and variance analysis | Does not bind credit to recurrent text memory states | Important variance baseline/theory comparator |
| Sequence Compression / Chunked-TD (Ramesh et al., ICML 2024) | Uses predicted transition boundaries to accelerate temporal credit | Not a free-text writer/final-answer cross-fitted protocol | Motivates state-boundary stress tests; no broad novelty claim |
| Turn-level Credit Assignment for LLM Agents (Zeng et al., 2025) | Turn-level advantage estimation for multi-turn tool agents | Different state/action semantics; no exact materialized-memory filtration or stable-ID OOF certificate reported | Direct algorithmic baseline; sharply reduces MIC novelty |
| Agent Lightning / LightningRL (Luo et al., 2025; official code) | Hierarchical transition interface and credit module for arbitrary agent traces | General framework does not itself prove outcome-blind, cross-fitted memory-state innovations | MIC must win on estimator validity/mechanism, not transition decomposition |
| Step-level Q-value models (Zhai et al., AAAI 2025) | MCTS-annotated step values guide agent action selection | Uses additional search/preference supervision and inference-time selection | Same-budget, no-extra-supervision contrast |
| GRU-Mem (Sheng et al., 2026) | Dense update/exit rewards and gates for recurrent memory | Changes control structure and adds intermediate reward signals | MIC keeps memory workflow/reward fixed |
| Context-lite multi-turn RL / dual-discounting GAE (2025) | Separates step- and token-level credit in multi-turn training | Does not target exact overwrite-state innovation with OOF firewall | Must be included if implementation/details are comparable |
| Information bottleneck / rate-distortion RL | Constrains representation information or capacity | Addresses state sufficiency/complexity, not attribution of a fixed terminal return | Adjacent representation route, not a MIC contribution |
| Safe policy improvement / DRO and robust RL | Controls update risk or distribution shift | Does not estimate which materialized write changed conditional value | Out of scope except as alternative explanations/boundaries |

Primary sources: [MemAgent](https://arxiv.org/abs/2507.02259),
[official MemAgent code](https://github.com/BytedTsinghua-SIA/MemAgent),
[RUDDER](https://papers.neurips.cc/paper_files/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html),
[official RUDDER code](https://github.com/ml-jku/rudder),
[HCA](https://arxiv.org/abs/1912.02503),
[Counterfactual Credit Assignment](https://proceedings.mlr.press/v139/mesnard21a.html),
[Quantile Credit Assignment](https://proceedings.mlr.press/v202/mesnard23a.html),
[Sequence Compression / Chunked-TD](https://proceedings.mlr.press/v235/ramesh24b.html),
[turn-level LLM-agent credit](https://arxiv.org/abs/2505.11821),
[Agent Lightning](https://arxiv.org/abs/2508.03680), and
[step-level Q-value models](https://ojs.aaai.org/index.php/AAAI/article/view/34924).

**Novelty boundary.** We will not claim novelty for Doob decomposition,
successive return-prediction differences, actor--critic baselines, cross-fitting
in general, or turn-level credit in general. A publishable residual exists only
if the combination of materialized-state semantics and fail-closed evidence
changes what can be measured or trained, and if E1/T5 demonstrate a mechanism
not reproduced by simpler turn/length or conventional value baselines.

## 4. Experimental plan

### E0: oracle and adversarial unit suite

| Test | Required evidence | PASS criterion |
|---|---|---|
| Analytic branching toy MDP | Oracle `V_t`, `D_t`, residual per trajectory | Values/increments recovered to `<=1e-12` float64 |
| Telescoping | Per-trajectory components | Max absolute closure error `<=1e-12` float64 |
| Stable folds | Sorted stable/root IDs and fold receipt | Deterministic, balanced; no group crosses train/holdout |
| OOF exclusion | Per-fold train/predict ID hashes | Held-out ID/root absent from fit outcomes |
| Firewall | Explicit feature schemas and taint attempts | gold/future/current outcome/generated answer all rejected |
| Shuffle | Shuffled state-to-outcome association | Predictive advantage disappears within preregistered tolerance |
| Role routing | Token masks, scalar credits, gradient ledger | writer gets `D_t`; final gets only `R-V_T`; no overlap |
| Checkpoint | Critic-only state and actor hash | Independent critic restore; actor unchanged |

### E1: frozen Original trajectories

The input bundle must be read-only and hash-authenticated. E1 is feasibility,
not performance training. Fold assignment is stable-root grouped. Models and
hyperparameters are selected without fixed-S128 task performance.

| Table/figure | Rows | Columns / curves |
|---|---|---|
| OOF prediction | turn/length baseline; scalar critic; MIC critic | MSE, MAE, Brier or calibration slope/intercept, fold CIs |
| Innovation diagnostics | each writer turn and fold | mean `Dhat`, SE, conditional-bin means, coverage |
| Variance | terminal broadcast; scalar baseline; MIC | writer advantage and gradient proxy variance |
| Signal allocation | writer turns; final answer | absolute/signed credit share, zero fraction, residual share |
| Leakage audit | every forbidden feature/taint | rejection status and receipt hash |
| Closure | all trajectories | maximum/quantiles of float64 error |

E1 PASS requires all of: at least **5%** OOF MSE improvement over the frozen
turn/length baseline; calibration slope in `[0.2, 1.8]`; every fold × turn ×
pre-value-sign conditional innovation group with at least eight samples has an
absolute Studentized mean no greater than `3.5`; no fold
or root leakage; conditional innovation means compatible with zero after
multiple-testing correction; non-degenerate writer credit; answer residual does
not absorb all signal; machine-precision closure. Failure of prediction power is
`NO-GO_E1`, not permission to add gold or future inputs.

### Fresh training tables, only after dual GO

| Method | T5 EM/F1/format | T10 | T15 | T20 | T25 | writer-grad variance | compute |
|---|---:|---:|---:|---:|---:|---:|---:|
| Certified Original import | `[frozen evidence]` | | | | | | |
| Terminal broadcast | `[same as Original definition]` | | | | | | |
| Turn/length critic | `[pending]` | | | | | | |
| Standard scalar value critic | `[pending]` | | | | | | |
| MIC | `[pending]` | | | | | | |

Required ablations: non-cross-fitted diagnostic (never a legal method), shuffled
materialized states, pre-write rather than post-write state, MIC without answer
residual separation, fold count, critic capacity/regularization, and role-mask
normalization. Each algorithmic or engineering optimization receives a distinct
commit, manifest, experiment name, output root, and ledger; no result overwrite.

### Failure criteria and revision rule

- E0 failure, leakage, or closure failure: stop until corrected; no T5 command.
- E1 no predictive value beyond turn/length: `NO-GO_E1`.
- E1 predictive but conditional-mean/calibration failure: one preregistered
  simplification/regularization revision, then `NO-GO_E1` if repeated.
- T5 mechanism inactive or calibration failure: one matching failure-class
  revision only.
- T5 S128 token F1 worse than certified Original-T5 by more than 2 points:
  `NO-GO_T5` even if critic loss improves.
- T25 success remains the frozen program criterion: +2 F1, or F1 within 1 point
  with at least 30% writer-gradient variance reduction and +2 worst-anchor F1;
  multi-seed confirmation remains separately authorized.

## 5. Complete paper story and decision points

1. **Phenomenon:** measure whether terminal broadcast produces high-variance or
   misallocated writer signal on frozen trajectories. `[PENDING E1]`
2. **Failure mechanism:** show with an oracle deletion/recovery MDP why equal
   trajectory credit cannot identify state-changing writes.
3. **Insight:** the defensible boundary is the exact text state consumed by the
   next call; validity depends as much on information exclusion as prediction.
4. **Method:** stable-root cross-fitted value differences, algebraic terminal
   residual, role-specific delivery, independent critic lifecycle.
5. **Theory:** oracle martingale identity plus explicit approximation and
   delivery error accounting; no novelty claim for the classical identity.
6. **Identification:** feature taint firewall, OOF receipts, calibration,
   conditional means, exact closure, and gradient ledger.
7. **Empirics:** same-base, same-budget T5--T25 anchors only after E0/E1 and
   independent framing review GO.
8. **External validity:** frozen variable-tracking stress suite; boundary cases
   include uninformative states, endogenous chunk selection, related-ID leakage,
   and answer-side compensation.

Current decision: **REFRAME**, not yet GO. The next admissible transitions are
`REFRAME -> NO-GO_E1` or `REFRAME -> KEEP_AFTER_E1`; paper numbering does not
override the evidence.

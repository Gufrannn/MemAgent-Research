# Paper I working draft — Prefix-Divergence Trust Regions for State-Writing Policies

> **Post-diagnostic revision.** The binary hard-rollback implementation has now
> produced a formally NO-GO, diagnostic-only T5–T25 run: seven nonzero commits,
> no accepted anchor proposal, and mean fixed-S128 F1 delta -0.01029 versus the
> certified Original curve.  These observations do not establish causality.
> The active algorithm proposal is the transactional feasible-step controller
> specified in `tf_rwwpo_revision_20260822.md`; no TF-RWWPO result exists yet.

**Scientific status (2026-08-22): PAPER-FRAMING GO; empirical claims remain conditional on Method evidence.** Sequence-level and
sub-sequence policy optimization are already established by GSPO, FSPO, and
GSsPO. “Use a joint ratio for a writer response” is therefore not a sufficient
novelty claim. The viable residual is narrower and harder: a free-text writer
action becomes the next observation, so the relevant change of measure is the
*trajectory prefix of materialized state rewrites*, while the terminal answer is
not a state transition and should retain the accepted Original token objective.
The mechanism claim is retained only if Method actual-loss evidence shows a
recurrent-prefix phenomenon beyond single-response and token diagnostics.

## Tentative title

**Prefix-Divergence Trust Regions for State-Writing Policies**

## Abstract (203 words; no unobserved result is asserted)

Language-memory agents repeatedly generate free-form text that is materialized
as the state observed at the next turn. Standard PPO-style training nevertheless
treats every generated token as a separately clipped action. Recent sequence-
and sub-sequence-level objectives correct related granularity mismatches for
single responses and multi-turn reasoning, but they do not distinguish a text
segment that *creates future occupancy* from a terminal segment that only
reports an answer. We study this distinction in recurrent memory writing. We
show algebraically that tokenwise clipping is factorization-dependent and does
not bound the likelihood ratio of a trajectory prefix of materialized writes.
We then formulate Recurrent Whole-Write Policy Optimization (RWWPO), which uses
the exact joint likelihood ratio of the writer prefix while leaving terminal
answer tokens under the accepted tokenwise PPO objective. Its normalization is
chosen so that, at the behavior policy, the writer gradient is exactly equal—not
merely proportional—to Original PPO; only higher-order update geometry changes.
RWWPO pairs this surrogate with per-turn prefix ESS, an absolute prefix-log-ratio
cap, and chi-square certificates
computed from the actual loss tensors. The empirical claim is deliberately
conditional. We train once from the same fresh Qwen2.5-7B base through T25,
retain five checkpoints, and use the Method's actual-loss ledger to test whether
prefix behavior adds information beyond token KL, clipping, and length. After
training, all five checkpoints are evaluated under one frozen protocol and
compared with the certified read-only Original curve and matched granularity
ablations. A missing mechanism signal or negative training result is a
falsification outcome rather than hidden tuning.

## 1. Introduction

An autoregressive language model is usually optimized token by token. This is a
natural computational factorization, but it need not be the decision
factorization of the environment. The distinction becomes concrete in a
recurrent memory agent. At writer turn (t), the model emits an entire text
(M_{t+1}); the environment then materializes that text as the memory state
available to later writer turns and to the final reader. A locally plausible
token edit can therefore change the support and likelihood of every downstream
state. The final answer is different: it terminates the trajectory and does not
become a future observation. Treating both roles as an undifferentiated matrix
of response tokens hides this asymmetry.

This problem is not solved merely by invoking “sequence-level RL.” GSPO already
argues that token importance ratios are invalid for a sequence-rewarded language
model and uses a length-normalized response likelihood ratio. FSPO further shows
that fixed sequence clipping can create length-dependent acceptance and proposes
length-aware bands. Workflow-R1's GSsPO aligns ratios with Think–Action
sub-sequences in multi-turn agents. These results substantially narrow our
novelty boundary. RWWPO cannot claim sequence ratios, semantic segmentation, or
generic stability as new. Its potential contribution must instead arise from a
different probability object: the cumulative likelihood of the sequence of
*state-producing writes* through a recurrent turn. A per-write ratio controls
the action just sampled; a prefix ratio measures the change of measure for the
actual chain of states on which the next shared policy acts. After the full Adam
proposal, a frozen descending grid of genuinely realized parameter
displacements is re-forwarded on the actual-loss minibatch. The largest tested
feasible point is committed transactionally; if none is feasible, model,
optimizer, scheduler, and RNG state are restored.

Why can this matter? Let a writer output contain (K) tokens and let every
token ratio lie inside a PPO clipping interval. The joint ratio is their
product, so its logarithm is their sum. Even mild same-sign changes can move the
joint action far outside any comparable trust region, while alternating changes
can conceal substantial factor-level movement. Across recurrent turns, the
effect compounds again. Token KL is an average local diagnostic and does not,
without additional assumptions, bound the empirical change-of-measure variance
of materialized writer prefixes. TRPO bounds policy improvement using policy
divergence under an occupancy distribution, and occupancy-divergence methods
directly constrain visitation shifts. Neither is directly estimable for the
combinatorial free-text state space in this workload. Prefix importance weights
offer a falsifiable, on-batch certificate: normalized ESS reports concentration,
and chi-square reports the corresponding second-moment inflation.

The algorithmic design follows from this role distinction. For writer tokens,
RWWPO sums current-minus-old log probabilities over each whole write and then
over all materialized writes up to turn (t). The writer surrogate uses the
final available writer-prefix ratio for a trajectory. For final-answer tokens,
it preserves Original tokenwise clipped PPO. Both components share Original's
active-token denominator. This denominator is not cosmetic. At the behavior
point every importance ratio equals one, and differentiating the joint
log-likelihood yields the sum of the same token score functions as Original.
With the shared denominator, the writer gradients are exactly equal. Thus a
comparison isolates higher-order geometry instead of silently changing writer
loss scale.

The second component is an identification protocol rather than another reward.
For every optimizer epoch and minibatch, we record the old and current
log-probabilities actually consumed by the loss, writer/final/turn masks,
advantages, active denominator, stable trajectory identity, and per-turn prefix
ESS/chi-square. Statistics reconstructed from generation-engine probabilities
or aggregate training logs are inadmissible. Missing tensors fail closed. This
ledger enables three distinct questions: whether token clipping can coexist with
prefix collapse; whether collapse is explained only by length; and whether a
constraint leaves nonzero update aperture rather than freezing the writer.

The current evidence does not answer those questions. The accepted Original
run establishes a corrected fresh-base T0-to-T25 curve on a fixed S128 evaluation
set, and existing Capture32 evidence concerns a different same-candidate
mechanism. Neither contains actual-loss current log-probabilities across PPO
epochs. We therefore do not claim that RWWPO improves performance, or even that
the target phenomenon occurs. E0 tests exact gradients, mask closure, and the
disabled-path equivalence. Original actual-loss tensors do not exist and are not
a main-experiment prerequisite. Prefix diagnostics are computed from the
Method's own append-only on-policy actual-loss ledger. An independently
authorized Original collection-only run may be reported as a finite diagnostic,
but never as the certified Original baseline.

The empirical study starts Method from the same fresh Qwen2.5-7B base used by
Original, enables RWWPO at update 1, and holds data order, rollout seeds,
reward, tokenizer, effective batch, trajectory budget, and evaluation protocol
fixed. We compare five anchors (T5/10/15/20/25), test joint-only and matched
trust-region alternatives, and evaluate a small RULER Variable Tracking suite as
an external state-transition stress test. A T5 degradation larger than two F1
points, a permanently saturated constraint, or a writer with zero update
aperture falsifies the method under the preregistered contract.

Subject to those gates, the intended contributions are:

1. **Theory:** factorization counterexamples and a change-of-measure statement
   connecting writer-prefix chi-square/ESS to error in downstream bounded-loss
   expectations; plus an exact behavior-point gradient identity.
2. **Algorithm:** a role-aware hybrid objective—joint recurrent writer prefix,
   Original final-answer PPO—with a prefix trust-region controller.
3. **Identification:** an actual-loss, per-epoch/minibatch ledger that makes the
   proposed recurrent trust-region mechanism independently reconstructible.
4. **Evidence:** conditional same-budget multi-anchor tests and an external
   variable-tracking stress test, including negative results and failure bounds.

## 2. Problem formulation and proposed propositions

For trajectory (i), writer turn (t\in\{1,\ldots,T_i\}) emits token sequence
(W_{it}), which is deterministically materialized into the next memory state.
The terminal answer (Y_i) is emitted after the last write. Define

\[
 \ell_{it}(\theta)=\sum_{k\in W_{it}}
 [\log\pi_\theta(w_{itk}\mid h_{itk})-\log\pi_{old}(w_{itk}\mid h_{itk})],
 \quad L_{it}=\sum_{s\le t}\ell_{is},\quad \rho_{it}=e^{L_{it}}.
\]

The writer numerator is \(-\sum_i \rho_{iT_i}A_i\). The answer numerator is
the unchanged Original tokenwise clipped PPO numerator. Their sum is divided by
the count (D) of all active response tokens used by Original.

**Proposition 1 (factorization non-invariance; to prove formally).** There exist
two autoregressive factorizations representing the same joint old and new
distribution over a whole write for which tokenwise clipped PPO produces
different clipped surrogates. Hence the token-clipped objective is not a
function of the whole-write policy alone.

**Proposition 2 (exact first-order matching).** Assume writer tokens of a
trajectory share the same scalar advantage, masks close exactly, and RWWPO uses
Original's denominator (D). At \(\theta=\theta_{old}\), the gradient of the
RWWPO writer term equals the gradient of Original's unclipped behavior-point
writer term exactly. Since PPO clipping is inactive in a neighborhood of ratio
one, it also equals Original clipped PPO at that point.

**Proposition 3 (prefix change-of-measure certificate).** For any bounded
downstream statistic \(|f|\le C\) measurable after writer turn (t), under
absolute continuity,
\[
 |E_{\pi_\theta}f-E_{\pi_{old}}f|
 \le C\sqrt{\chi^2(P_{\theta,1:t}\Vert P_{old,1:t})}.
\]
The empirical normalized-weight quantity satisfies
\(\widehat{ESS}_t/B_t=1/(1+\widehat\chi_t^2)\) under the ledger's convention.
This is a diagnostic bound, not a monotonic-improvement guarantee.

**Counterexample target.** For any token clip width \(\epsilon>0\) and desired
joint ratio (R), choose a sufficiently long write with each token ratio inside
\([1-\epsilon,1+\epsilon]\) yet product at least (R). A recurrent two-turn
construction then maps that write to disjoint downstream states, demonstrating
that token acceptance alone does not control prefix occupancy.

## 3. Closest-work claim matrix and novelty boundary

| Primary source | Solves | Does not solve for this paper | Consequence |
|---|---|---|---|
| Schulman et al., TRPO (ICML 2015) | Trust-region policy improvement via policy divergence | Tractable certificate for free-text recurrent state writes; mixed writer/answer roles | Use as theory foundation; no monotonicity overclaim |
| Schulman et al., PPO (2017) | Practical token/action ratio clipping | Factorization-invariant control of compound text actions or writer prefixes | Original baseline |
| Touati et al., occupancy-divergence regularization (UAI 2020) | Directly regularizes discounted visitation divergence | Does not give the actual-loss prefix certificate or language-memory role split | Compare conceptual target; acknowledge stronger general occupancy framing |
| Zheng et al., GSPO (2025) | Sequence likelihood ratio, sequence clipping/rewarding/optimization | Does not distinguish state-producing writes from terminal answer; uses per-response length-normalized ratio, not recurrent materialized-write prefix ESS | Eliminates “sequence ratio” as novelty; mandatory baseline |
| Mao et al., FSPO (2025) | Diagnoses length unfairness of fixed sequence clipping; length-aware band | Recurrent state role and cross-turn prefix occupancy | Mandatory length-controlled baseline/audit |
| Kong et al., Workflow-R1 / GSsPO (2026) | Structure-aware sub-sequence ratio for Think–Action cycles in multi-turn workflows | Materialized free-text memory prefixes, role-asymmetric final answer, actual-loss prefix ESS certificate | Strongest overlap; paper survives only with real prefix phenomenon |
| Memory-R1 (2025), Mem-alpha (2025), Mem-T (2026), UMA (2026) | Learn memory operations/writers with RL; some add structured operations or dense credit | Whole-prefix trust geometry with exact Original first-order matching | Establishes task importance, not optimizer novelty |
| MMPO (2026) | Belief-entropy intermediate supervision for recursive memory | Reward-invariant higher-order geometry | Distinct; do not add its dense proxy |
| Group DRO / robust RL | Worst-group/environment robustness | Within-update behavior-policy change of measure | Not an RWWPO contribution; relevant only as external robustness context |
| Information bottleneck / policy capacity | Compression/information constraints on policies or states | Prefix importance concentration at fixed reward/memory semantics | Explicitly outside RWWPO; do not merge without new authorization |

**Novelty boundary.** We may claim only the combination of (i) action identity as
materialized free-text state rewrite, (ii) cumulative recurrent writer-prefix
ratio rather than response/sub-sequence ratio, (iii) asymmetric preservation of
Original answer-token PPO, (iv) exact—not scaled—behavior-point writer gradient,
and (v) actual-loss prefix ESS/chi-square identification. If Method ledgers show no
incremental signal beyond GSPO/GSsPO, token KL, and length, status becomes NO-GO
or MERGE into a broader structure-aware policy-optimization study.

## 4. Experiments, ablations, and failure criteria

### Tables/figures planned

| Artifact | Rows / curves | Columns / axes | Claim tested |
|---|---|---|---|
| E0 table | Original, RWWPO, RWWPO-off | loss, gradient max error, cosine, mask closure, finite difference | exact first order and off equivalence |
| Method mechanism table | turns 1..T; length bins | ESS, chi-square, token KL, clipfrac, joint aperture | phenomenon exists beyond length/local diagnostics |
| Method predictive table | held-out update groups | collapse prediction / partial R² | prefix statistic adds information |
| Main curve | Original, RWWPO | S128 EM/F1/format at T0/5/10/15/20/25 | same-budget performance |
| Mechanism curve | methods × anchors | ESS quantiles, saturation, writer grad, aperture | method active, not frozen |
| Matched baselines | Original, GSPO-style, GSsPO/per-write, joint-only, token-KL matched | T5/T25 and mechanism | isolate recurrent prefix constraint |
| Ablations | end-prefix-only, all-token-prefix, answer-only sham, fixed penalty, controller | same metrics | component attribution |
| External validity | Original/RWWPO on frozen RULER Variable Tracking | accuracy by horizon/rewrite count | state-transition relevance |

All performance aggregates are recomputed from prediction rows of the certified
read-only baseline bundle after per-file SHA verification. Dense reward is never
reported as performance. Formal confirmation beyond screening requires at least
three fresh seeds and an untouched confirmation run under separate authorization.

### Preregistered gates

- **E0 PASS:** max absolute writer-gradient error at behavior point within the
  dtype-specific tolerance; closure of response = writer disjoint-union answer;
  RWWPO disabled follows byte-for-byte Original control flow and passes relevant
  regressions.
- **T5 health PASS:** finite loss/gradients, complete checkpoint, valid
  append-only Method ledger and weight sync; no S128 pause is required.
- **T25 success:** T25 F1 +0.02 and five-anchor mean +0.01, with no anchor worse
  than Original by more than 0.02.
- **Immediate NO-GO:** Method prefix diagnostics add no signal beyond local
  diagnostics; method is stable only at effectively zero writer step; missing
  tensors are presented as evidence; or a numeric/contract failure occurs.

## References / primary links reviewed

- [GSPO](https://arxiv.org/abs/2507.18071)
- [FSPO](https://arxiv.org/abs/2509.09177)
- [Workflow-R1 / GSsPO](https://arxiv.org/abs/2602.01202)
- [TRPO](https://proceedings.mlr.press/v37/schulman15.html)
- [Stable Policy Optimization via Occupancy Divergence](https://proceedings.mlr.press/v124/touati20a.html)
- [Memory-R1](https://arxiv.org/abs/2508.19828)
- [Mem-alpha](https://arxiv.org/abs/2509.25911)
- [Mem-T](https://arxiv.org/abs/2601.23014)
- [Group Distributionally Robust RL](https://proceedings.mlr.press/v206/xu23d.html)
- [Policy Information Capacity](https://proceedings.mlr.press/v139/furuta21a.html)

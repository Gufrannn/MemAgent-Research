# Paper I working draft — Controlled Off-Behavior Geometry for State-Writing Policies

> **RWWPO-2 revision (2026-08-23).** The frozen long-study contract is now
> `rwwpo2_r400_k2_preregistration_20260823.md`. The binary hard-rollback implementation has
> produced a formally NO-GO, diagnostic-only T5–T25 run: seven nonzero commits,
> no accepted anchor proposal, and mean fixed-S128 F1 delta -0.01029 versus the
> certified Original curve.  These observations do not establish causality.
> The old K1 run identifies controller behavior only: under the frozen
> single-pass assumptions, tokenwise, per-write, and whole-path objectives are
> algorithmically degenerate. The active proposal is R400-K2 with tokenwise,
> per-write, whole-path, controller, and no-controller cells. H20 RWWPO-2
> attempts have exposed implementation defects and are all NO-GO; no valid R50
> endpoint or RWWPO-2 scientific result exists yet.
> Because those five S128 anchor results were inspected before the controller
> revision, S128 is now a development benchmark for this method family, not a
> blind final test.

**Scientific status (2026-08-27): PAPER-FRAMING GO; no RWWPO-2 R50 endpoint has
yet passed the formal attempt audit and every empirical claim remains
conditional. An E resume segment completed its R50 transactions and checkpoint
but exited in post-save housekeeping; it remains a candidate endpoint pending
the narrow read-only audit, not a scientific PASS.** Sequence-level and
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

## Abstract (no unobserved result is asserted)

Language-memory agents repeatedly generate free-form text that is materialized
as the state observed at the next turn. Standard PPO-style training nevertheless
treats every generated token as a separately clipped action. Recent sequence-
and sub-sequence-level objectives correct related granularity mismatches for
single responses and multi-turn reasoning, but they do not distinguish a text
segment that *creates future occupancy* from a terminal segment that only
reports an answer. We study this distinction in recurrent memory writing. We
first prove a single-pass degeneracy: at the behavior policy, tokenwise,
per-write, and whole-writer-path surrogates have the same exact parameter
gradient and therefore induce the same deterministic transition under a common
complete state. This converts an apparent method result into a sharper design
requirement—objective geometry is identifiable only after a controlled
off-behavior transaction. RWWPO-2 reuses each behavior batch for two audited
full-batch actor transactions, combines an Original tokenwise answer objective
with alternative writer geometries, and realizes a descending grid of actual
parameter proposals subject to per-turn prefix ESS, prompt-root ESS, and
absolute log-ratio feasibility. We preregister controller-only, per-write,
whole-path, no-controller, and hard-rollback cells; same-host shadow gradients;
attempt-independent proposal clocks; and authenticated checkpoint-DAG replay.
R50 is a performance-free mechanism gate. Conditional on that gate, R400 is an
eight-seed paired medium-budget study—not a convergence claim—with macro
token-F1 co-primary contrasts B-D and B-E on a separately sealed, disjoint
confirmation set. No RWWPO-2 performance result is reported here; insufficient
exposure, unstable root support, or either failed co-primary test falsifies the
full claim.

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
over all materialized writes up to turn (t). For final-answer tokens it
preserves Original tokenwise clipped PPO, and both components share Original's
active-token denominator. At the behavior point every importance ratio is one,
so tokenwise, per-write, and whole-path writer gradients are exactly equal.
This is not only a scale-matching property: with one deterministic transaction
per behavior batch, identical complete state and an objective-blind controller,
the three variants are algorithmically degenerate. RWWPO-2 therefore performs
exactly two full-batch transactions on the same frozen behavior batch. Inner 1
tests the equality; inner 2 exposes the higher-order geometry only if inner 1
made a numerically real move.

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
run establishes a corrected fresh-base T0-to-T25 curve on S128, and the old K1
RWWPO run provides only a controller diagnostic. Its inspected five-anchor S128
curve informed the K2 pivot; S128 is consequently development-only. We do not
claim that RWWPO-2 improves performance or that the target off-behavior mechanism
occurs. E0 tests behavior-point full parameter gradients, transition-kernel
closure, proposal clocks, and replay seeds. Prefix diagnostics come only from
the Method's append-only tensor ledger; Original internal tensors are not a
main-experiment prerequisite.

The new study starts every cell from the same fresh Qwen2.5-7B base and enables
its assignment at update 1 while holding data order, logical rollout seeds,
reward, tokenizer, batch, and proposal budget fixed. Each rollout round receives
two full-batch actor transactions, yielding at most 800 proposals through R400.
R50 uses three paired mechanism seeds across D/C/E/B/A and reads no performance
benchmark. Only after finite numerics, common-host gradient equality,
off-behavior exposure, root-support stability, controller aperture, distributed
sync, and recovery replay pass do B/D/E continue to R400 for eight paired seeds.
The final set contains at least 512 unseen content/root-disjoint examples and is
opened once. This is a medium-budget confirmatory design, not evidence of
optimization convergence.

Subject to those gates, the intended contributions are:

1. **Theory:** factorization counterexamples and a change-of-measure statement
   connecting writer-prefix chi-square/ESS to error in downstream bounded-loss
   expectations; plus an exact behavior-point gradient identity.
2. **Algorithm:** a role-aware hybrid objective—joint recurrent writer prefix,
   Original final-answer PPO—with a prefix trust-region controller.
3. **Identification:** an actual-loss, per-epoch/minibatch ledger that makes the
   proposed recurrent trust-region mechanism independently reconstructible.
4. **Evidence:** a performance-free R50 mechanism gate and conditional
   eight-seed R400 confirmation with explicit negative-result boundaries.

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
| E0 table | common-host C/E/B and off-behavior probe | full logprob/parameter-gradient error, optimizer-state induction, clock/seed replay | K1 degeneracy and K2 identifiability |
| R50 mechanism matrix | D/C/E/B/A × three seeds | exposure, conditional shadow separation, root ESS/LOO, accepted alpha, salvage/reject | mechanism exists with nonzero aperture |
| Same-host geometry | both inner transactions of every R50 round, separately by host cell | coefficient difference, cosine, deterministic gradient projections | local whole-path geometry beyond per-write |
| Recovery table | interrupted/replayed attempts | logical seed, proposal clock, accepted clock, checkpoint/tensor-ledger prefix | attempt-invariant authenticated DAG replay |
| R400 learning curves | B/D/E (plus explanatory C) × frozen anchors | training health only; no adaptive performance selection | same-budget dynamics without convergence claim |
| One-time confirmation | B/D/E × eight paired seeds | macro token-F1, strict EM, precision/recall, format | B-D utility and B-E whole-path-package assignment |

Dense reward is never reported as performance. S128 has already informed the
hard-rollback-to-K2 pivot, so it is not an untouched final test and is forbidden
during R50/R400 training. Formal confirmation uses eight preregistered paired
training seeds and a separately sealed, at-least-512-root confirmation set that
is content/root-disjoint from actor training, S128, Capture32, and prior
inspected inventories. It is accessed only after code, thresholds, mechanism
analyses, and checkpoint rules are frozen.

### Preregistered gates

- **E0 PASS:** behavior-point C/E/B full logprob and full parameter gradients
  agree within the predeclared CPU tolerance; exact common-gradient transition
  induction, stateless proposal clock, and attempt-independent seeds close.
- **R50 mechanism PASS:** all five cells and all three mechanism seeds have at
  least 40 eligible rounds, 10 exposed rounds, exposure rate 0.20, five
  activated exposed rounds, activation rate 0.50, root-LOO flip fraction at
  most 0.25, and at least one B proposal salvaged relative to hard rollback.
  No S128 or confirmatory performance is read.
- **R400 confirm eligibility:** R50 PASS; unseen confirmation seal PASS; exact
  B/D/E eight-seed coverage; all attempt-DAG, tensor-ledger, checkpoint,
  accepted-clock, and weight-sync audits PASS.
- **Full claim PASS:** both one-sided paired-seed macro-F1 contrasts B-D and B-E
  pass margin-centered exact sign-flip tests with Holm family-wise correction
  at 0.05, using margins +0.02/+0.01 respectively. Either failure rejects the
  full claim. The longitudinal B-E result is an assignment contrast; local
  same-host shadows, not performance alone, identify geometry activation.

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

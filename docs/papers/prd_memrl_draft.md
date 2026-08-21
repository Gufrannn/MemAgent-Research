# Paper IV Draft — Conditional Innovation Rate for Recurrent Text Memory

**Scientific status (2026-08-22): REFRAME, pre-E1.** This document states a
falsifiable paper program, not a result claim. Bracketed entries are experiment
placeholders. A failure of the Original rate-not-length audit, prior coding-gain
gate, or three-point frontier gate changes the status to NO-GO.

## Working title

**How Many New Nats Should a Memory Rewrite Carry? Conditional Innovation-Rate
Frontiers for Recurrent Text Agents**

## Abstract (203 words)

Recurrent language agents repeatedly overwrite a text memory while reading a
stream. Existing systems choose a maximum token count or add heuristic length
and semantic rewards, but neither quantity measures how much *new* information
a rewrite transmits beyond the previous memory. We formulate each rewrite as a
conditional communication channel from the newly available history to the next
memory. A learned coding prior observes only the previous memory and turn index,
never the arriving evidence, answer, reward, or future, and yields a variational
upper bound on conditional innovation rate through the actor-to-prior log-density
ratio. This produces a constrained recurrent-RL problem whose distortion is
future task return, not reconstruction or embedding similarity. We propose a
three-timescale actor–prior–dual algorithm and evaluate a preregistered family of
capacities rather than selecting one penalty after observing performance. The
paper is designed around falsification: source-taint tests certify the prior's
information set; sampled actor/prior token log-probabilities distinguish rate
from length; successive-rewrite ledgers prevent accounting only the final state;
and future predictive probes test whether low-rate memories remain task
sufficient. At present, no method result is claimed. The certified Original
MemAgent curve improves normalized exact match by 0.046875 from initialization
to update 25 on one fixed S128 evaluation, but does not identify an information
bottleneck. We will report whether conditional rate is a distinct failure axis,
whether a non-degenerate rate–return frontier exists, and where compression
causes irreversible forgetting.

## 1. Introduction

Long-context language agents face a simple physical problem: observations keep
arriving while the state that can be carried forward remains bounded. MemAgent
turns this problem into a recurrent language-policy problem. It reads a chunk,
writes a free-text memory, then exposes that generated memory to the next turn.
End-to-end reinforcement learning can discover useful overwrite behavior and
can extrapolate beyond its training context. Yet the capacity of the recurrent
state is still described almost entirely in tokens. A 200-token string can copy
rare identifiers nearly verbatim or contain generic boilerplate; two strings of
the same length can therefore carry radically different information about what
was newly observed. Conversely, semantic similarity to the incoming chunk does
not establish usefulness for a future question.

This mismatch matters because the scientific object is not concise prose. It is
a task-sufficient state transition. A writer should transmit information that
changes the distribution of future successful behavior, preserve information
already resident in memory without paying for it again, and discard variation
that future decisions do not need. A length penalty cannot separate these
cases. A KL penalty to a fixed pretrained model measures policy drift, not the
new information contributed at a particular rewrite. An intrinsic similarity or
conditional-mutual-information reward can encourage copying salient material,
but does not impose a communication budget and may reward information unrelated
to future utility.

We study a narrower question: **at a fixed previous memory, how much new
information may the next free-text state carry about the history newly visible
to the writer?** Let the actor be
`pi_theta(M_{t+1}|H_t,M_t,t)`. We introduce a coding distribution
`q_psi(M_{t+1}|M_t,t)` whose input is restricted by construction. Because the
prior cannot observe the new chunk, the expected log-density ratio from actor
to prior upper-bounds `I(H_t;M_{t+1}|M_t,t)`. The ratio is measured on the exact
generated writer tokens and summed across rewrites. It is therefore an
operational conditional code length, not a token count and not an embedding
score.

This view leads directly to an algorithm. We maximize the unchanged terminal
task return subject to an expected episode innovation-rate capacity. The actor
receives the ordinary policy-gradient objective plus a Lagrange penalty; the
history-blind prior performs maximum likelihood on stop-gradient actor samples;
and a projected dual variable tracks constraint violation. These components
operate at separated timescales and are checkpointed together. Rather than
tuning one coefficient and reporting its best checkpoint, we freeze at least
three capacities before training and estimate a rate–return frontier at every
T5/T10/T15/T20/T25 anchor.

The distinction from recent memory-RL work is substantive but must be tested.
MemAgent establishes that outcome-driven RL can train successive full-text
rewrites. Memory-R1 learns structured external-memory operations with QA-driven
PPO/GRPO. CMI-Mem, the closest contemporary paper, adds an action-conditioned
CMI-inspired dense reward computed from residual embedding projections. Our
proposal neither adds a CMI reward nor claims that all information from the new
chunk is valuable. It constrains a likelihood-based upper bound on transmitted
innovation while leaving future task return as distortion. Thus CMI-Mem asks
which operation looks informative; this work asks for the best future return
obtainable at each auditable communication rate. If this distinction does not
produce a non-degenerate frontier or predictive-sufficiency behavior beyond
length matching, the paper hypothesis fails.

The empirical strategy begins before method training. On frozen Original
trajectories we fit only the legal history-blind coding prior and test three
facts: rate must not be perfectly ordered by token length; the learned prior
must code better than unigram and turn-only controls without reconstructing the
current evidence; and successive rewrite rates must expose variation associated
with later retention failures. These tests guard against calling length an
information rate or introducing an expensive prior that adds no explanatory
power. A synthetic channel separately checks the variational bound and dual
dynamics, while static and dynamic taint tests mutate forbidden evidence fields
and require invariant prior inputs and outputs.

If these gates pass, fresh Qwen2.5-7B training starts from the same base as the
certified Original at update 1. Data order, effective batch, rollout count,
reward, tokenizer, evaluator, update budget, and GPU count remain identical.
The first comparison is Method-T5 against certified Original-T5; continuation
to later anchors requires mechanism health, capacity attainability, and no more
than two token-F1 points of degradation. The primary T25 test asks whether the
frontier dominates Original at equal or lower measured rate/token compute, or
achieves comparable F1 with at least 20% lower conditional rate. All claims are
conditional on multiple capacity points; no post-hoc single point may stand in
for a frontier.

The intended contributions are fourfold, pending evidence:

1. A conditional-channel formulation of successive free-text memory rewrites,
   with a variational innovation-rate upper bound that does not charge the next
   memory for information predictable from the previous memory.
2. An actor–history-blind-prior–dual training algorithm for constrained recurrent
   language-policy optimization, including complete restartable state.
3. A fail-closed identification protocol separating information rate from token
   length, prior leakage, final-state-only accounting, and apparent compression
   that destroys future predictive sufficiency.
4. A preregistered multi-capacity, multi-anchor frontier study on MemAgent and a
   minimal query-blind long-memory setting, with explicit collapse and NO-GO
   boundaries.

## 2. Problem formulation and main propositions

At turn `t`, let `M_t` be the materialized text state, `E_t` the newly arriving
evidence, and `H_t` the legal actor history including `E_t`. The writer samples

`M_{t+1} ~ pi_theta(. | H_t, M_t, t)`.

The legal coding prior is

`q_psi(M_{t+1} | M_t, t)`.

Its computation graph and serialized example contain exactly `M_t` and `t`.
Gold answers, rewards, current/future evidence, generated final answers, and
full histories are forbidden.

Define the per-turn variational innovation rate

`R_t(theta,psi) = E_pi[log pi_theta(M_{t+1}|H_t,M_t,t) - log q_psi(M_{t+1}|M_t,t)]`,

and episode rate `R=sum_t R_t`. The constrained objective is

`max_theta E[G]  subject to E[R] <= C`,

where `G` is the unchanged terminal task return. The minimization form used in
code is `L_actor = L_GRPO + lambda (R_hat-C)`; `q_psi` minimizes negative
log-likelihood on detached actor text; and
`lambda <- max(0,lambda+eta_lambda(R_hat-C))`.

### Proposition 1 — conditional innovation upper bound

For any normalized `q(M_{t+1}|M_t,t)` with support wherever the actor-induced
conditional marginal has support,

`I(H_t;M_{t+1}|M_t,t) <= E KL(pi_theta(.|H_t,M_t,t) || q(.|M_t,t))`.

**Proof sketch.** Insert the actor-induced marginal
`p_theta(M_{t+1}|M_t,t)`. The expected KL to `q` decomposes into the conditional
mutual information plus
`E KL(p_theta(.|M_t,t)||q(.|M_t,t))`, which is non-negative. Equality holds for
the exact conditional marginal. The synthetic E0 test evaluates this identity
on a finite channel, not merely the inequality.

### Proposition 2 — why length is not rate

There exist equal-token-length rewrite policies with different conditional
innovation rates, and different-length rewrite policies with identical rate.
The first follows from a policy that copies an evidence-dependent symbol versus
one emitting a history-independent symbol at the same length. The second follows
by padding a message with symbols deterministic given `(M_t,t)`. Therefore no
token-only penalty identifies the stated constraint without additional, generally
false, coding assumptions.

### Proposition 3 — successive accounting is necessary

There exist two recurrent channels with identical distribution over final memory
`M_T` and return but different `sum_t I(H_t;M_{t+1}|M_t,t)`: one transmits and
later deletes a bit; the other never transmits it. A final-state bottleneck cannot
distinguish their communication cost. Our ledger must therefore retain every
turn's actor/prior log-probability and mask.

### Optimization statement (conditional, not yet a theorem)

Under bounded stochastic gradients, feasible capacity, adequate prior tracking,
and standard step-size separation, projected stochastic primal-dual updates are
expected to approach a stationary KKT point of the non-convex surrogate. We do
not claim global convergence. E0 tests dual direction and recovery on a convex
finite-channel instance; H20 ledgers test constraint error and oscillation.

## 3. Closest-work claim matrix and novelty boundary

| Primary source | Solves | Does not solve relative to this paper | Boundary / required comparison |
|---|---|---|---|
| [MemAgent, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/4264ee4376776907c0b87ed70b959585-Paper-Conference.pdf) | End-to-end RL for chunked, overwrite-style free-text memory; long-context extrapolation. | No likelihood-based conditional rate, learned history-blind code, dual capacity, or frontier. | Exact host system and Original baseline; no claim that recurrent memory RL itself is new. |
| [Memory-R1 (2025)](https://arxiv.org/abs/2508.19828) | PPO/GRPO trains structured ADD/UPDATE/DELETE/NOOP memory management and answer use. | External structured operations, not successive full-text rewrite rate; QA utility does not measure communication. | Compare conceptually and, where interface permits, include structured-operation baseline only as external validity. |
| [CMI-Mem (2026)](https://arxiv.org/abs/2607.20553) | Adds action-conditioned, embedding residual-projection CMI-inspired reward to QA reward for memory management. | Does not estimate actor/prior code length, enforce a capacity, use future return as the sole distortion, or learn a rate frontier; its CMI path rewards information preservation. | Strongest novelty threat. Implement its legal intrinsic-reward analogue or cite code results as non-comparable; include “CMI reward + Original” ablation if budget permits. Do not claim first CMI memory RL. |
| [Dynamic Long Context Reasoning via compressed memory (2026)](https://arxiv.org/abs/2602.08382) | Learned compression, recall gating, and RL yield accuracy/compute tradeoffs. | Architectural block compression and selection, not conditional information-rate control of full-text state rewrites. | Compare token/compute frontier; avoid claiming first accuracy-efficiency memory frontier. |
| [RepDIB, AISTATS 2023](https://proceedings.mlr.press/v206/islam23a.html) | Variational/discrete bottlenecks learn task-relevant RL representations under distractors. | Latent observation representation, not auditable autoregressive text code or recurrent rewrite accounting. | Establishes that IB-in-RL is not new; use discrete/variational bottleneck as conceptual baseline. |
| [InfoBot, ICLR 2019](https://research.google/pubs/infobot-structured-exploration-in-reinforcementlearning-using-information-bottleneck/) | Goal-conditioned information bottleneck identifies decision states for exploration. | Bottlenecks goal information in action policy, not evidence innovation in memory state. | No novelty claim for learned-prior KL in RL. |
| [Capacity-Limited Actor-Critic (2019)](https://openreview.net/forum?id=ByxAOoR5K7) | Rate–distortion view and actor-critic for capacity-limited policies. | Policy/action channel in gridworld, not natural-language memory state or successive predictive sufficiency. | Establishes primal-dual/rate–utility RL precedent; contribution must reside in channel choice and certificates. |
| [Variational Predictive Information Bottleneck (2020)](https://proceedings.mlr.press/v118/alemi20a.html) | Variational predictive-IB objective and learned variational distributions. | Not a recurrent text-memory RL algorithm or empirical rate frontier. | Theory foundation, not claimed novelty. |
| [Recurrent model-free RL for POMDPs, ICML 2022](https://proceedings.mlr.press/v162/ni22a.html) | Strong recurrent baselines and careful POMDP evaluation. | Does not constrain explicit text-state communication. | Motivates recurrent and memoryless controls in the query-blind test. |
| [InstructGPT (2022)](https://arxiv.org/abs/2203.02155) and PPO | Per-token KL to a fixed reference controls policy drift in language RL. | Reference sees the same prompt and encodes model change, not evidence information injected into memory. | Fixed-base KL is a mandatory ablation at matched realized KL/rate. |
| Token-length Lagrangian | Directly limits storage/compute. | Cannot distinguish predictable padding from evidence-dependent symbols. | Mandatory matched-token baseline; rate claim fails if it explains all effects. |

**Novelty boundary.** The defensible residual is the combination of (i) the
conditional *innovation* channel induced by recurrent full-text overwrite, (ii)
a source-firewalled learned autoregressive coding prior, (iii) future return as
distortion under an explicit constraint, and (iv) successive rate/frontier and
predictive-sufficiency certificates. Learned priors, IB, KL control, primal-dual
RL, memory RL, compression, and Pareto curves are individually prior art. If the
paper reduces empirically to a length penalty or a CMI-shaped reward, REFRAME
becomes NO-GO or MERGE with the relevant neighboring line.

## 4. Experiments, tables, and falsification plan

### E0 — no long training

| Test | Required evidence | PASS | Failure meaning |
|---|---|---|---|
| Finite synthetic channel | exact MI, variational KL decomposition, support checks | upper bound and decomposition within tolerance | objective/estimator invalid |
| Dual toy | capacity, rate, lambda trajectory, primal response | approaches capacity without permanent collapse | timescale/numerics invalid |
| Static taint | AST/data-schema dependency graph | prior path contains only previous memory and turn | leakage; blocker |
| Dynamic taint | mutate evidence/gold/future with fixed legal context | prior input hash and logits bitwise/tolerance invariant | leakage; blocker |
| Checkpoint round trip | actor, prior, both optimizers/schedulers, dual, RNG, step, frontier ID | exact restart receipt | experiments non-recoverable |

### E1 — frozen Original trajectories

**Table 1: Rate is not length.** Per-turn and per-trajectory Kendall/Spearman;
discordant-pair fraction; regression `rate ~ tokens + turn`; residual variance;
same-length rate spread. PASS requires nonzero preregistered discordance and
material residual variation, not just p-values.

**Table 2: Coding prior validity.** NLL/nats per writer token and total nats for
unigram, turn-only, previous-memory-only learned prior, and prohibited
evidence-aware oracle. Legal prior must improve over legal simple baselines but
remain worse than the leaking oracle. Evidence reconstruction/probe from legal
prior context must be at chance conditional on previous memory.

**Table 3: Predictive association (descriptive E1).** Conditional-rate residual
versus later answer correctness, same-candidate retention harm, and memory
survival, controlling token count/turn/root. This cannot select capacity or be
used as reward.

E1 failures: perfect length ordering; no legal coding gain; source leakage;
missing exact actor tokens/log-probs; or insufficient comparable turns. Any is
`NO-GO_E1`, not permission to train.

### Fresh Method frontier

Capacities are frozen after E1 and before any S128 Method result, with at least
three distinct ascending values spanning below/equal/above the Original E1
realized rate. Every capacity is an independent experiment, manifest, ledger,
checkpoint namespace, and commit if algorithmically changed.

**Table 4: T5 health and frontier.** Capacity, realized rate (nats/trajectory),
writer tokens, dual mean/range, constraint error, prior NLL, entropy, collapse
rate, normalized EM, token-F1, format validity. Compare every point to certified
Original-T5. Continue only if the constraint is reachable, the frontier is
non-degenerate, and at least one matched point is within 2pp F1.

**Table 5: T5/10/15/20/25 rate–return curves.** All capacity points at every
anchor; hypervolume and monotonicity reported with uncertainty. No “best point”
table without the full frontier.

**Table 6: T25 matched comparisons.** Original, token-length dual, fixed KL to
base, unconditional/turn-only prior, legal conditional prior, evidence-leaking
oracle (diagnostic only), fixed-beta/no-dual, and single-turn accounting.

**Table 7: Predictive sufficiency.** Freeze memories, train/evaluate identical
future-answer and future-evidence probes; report correctness versus rate and
length. Include counterfactual deletion and irrelevant-evidence injection.

**Table 8: External validity.** Minimal query-blind LongMemEval subset or an
equivalent frozen variable-tracking stream. The memory writer cannot see the
terminal query during ingestion. Report frontier shift, not only accuracy.

### Failure criteria and boundaries

- `NO-GO_E1`: rate is a deterministic ordering of length, legal prior has no
  coding gain, or taint/inventory evidence is incomplete.
- `NO-GO_T5`: after at most one preregistered prior-capacity or dual-timescale
  correction, frontier collapses, all points violate capacity, or all legal
  matched points lose more than 2pp token-F1.
- Main T25 success: at equal/lower conditional rate and token cost, F1 improves
  by at least 2pp; or at F1 within 1pp, conditional rate falls at least 20%, with
  stable five-anchor ordering.
- Scope boundary: a gain explainable by shorter strings alone supports a length
  regularization paper, not this paper. A gain from intrinsic CMI reward supports
  merger with CMI-style memory valuation, not PRD. A frontier visible only when
  the prior reads new evidence is invalid.

## 5. Implementation-to-insight ablation registry

Every optimization must map to the conditional-channel insight:

| Optimization | Why needed | Required ablation |
|---|---|---|
| Low-rank/shared prior backbone | Make legal conditional coding affordable | full small prior vs low-rank at matched NLL/compute |
| Stop-gradient actor samples for prior | Prevent prior loss changing actor representation directly | joint-gradient diagnostic, excluded from valid method |
| Dual EMA / slower update | Reduce sampled-rate oscillation | raw projected dual at same capacities |
| Per-turn capacity allocation | Address heterogeneous rewrite opportunities | global episode capacity at same total C |
| Log-ratio centering/clipping for numerics | Stabilize tails, not redefine rate | exact unclipped ledger and clipped-gradient-only variant |

## Primary-source audit trail

Primary paper pages and official repositories were inspected on 2026-08-22.
The nearest-paper search explicitly included recurrent/agent memory RL,
sequence-level language-policy optimization, conditional information
bottlenecks/rate distortion, POMDP representation learning, and constrained RL.
The claim matrix will be revised if E1 or reviewer search finds a closer source.

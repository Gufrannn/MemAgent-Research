# MIC-v2 scientific contract and preregistration

Date frozen for review: **2026-08-25**

Status: **SCIENTIFIC CONTRACT DRAFT — NO IMPLEMENTATION OR H20 RUN AUTHORIZED**

Route: Paper II, Memory Innovation Credit (MIC)
Historical MIC-v1 evidence: retained as an early-budget negative/mechanism pilot;
it is not silently relabelled as MIC-v2 evidence.

This document freezes the scientific object, estimands, comparison cells,
failure interpretations, and evidence boundaries for MIC-v2. It supersedes the
future experimental design in `mic_paper_framing_20260822.md`; it does not alter
that document's historical claims, artifacts, or adaptive-use disclosure. After
independent scientific review, any substantive change to this contract requires
a new version, explicit rationale, and a new experiment family. Results may not
be used to edit this version in place.

## 1. Research question and central claim

In a recurrent text-memory agent, a new chunk arrives exogenously and the policy
then emits a free-text writer action that is materialized as the memory used by
future calls. Terminal broadcast credit conflates these two events. MIC-v2 asks:

> Can a history-privileged, time-safe critic estimate the policy-relative value
> of the writer action at the post-observation/pre-write to post-write boundary,
> and can that estimate improve finite-sample writer learning after recurrent
> trajectories have diverged?

The core insight is not Doob decomposition, actor--critic, a centralized critic,
cross-fitting, leave-one-out credit, or a reliability gate. The proposed
residual is their combination at a specific recurrent text-state boundary:

> Once recurrent writer trajectories have diverged, sibling centering may remain
> an action-independent control variate, but it no longer estimates the policy
> mean at each trajectory's own pre-write state. Rather than rerolling multiple
> writer actions from every intermediate memory snapshot, MIC amortizes this
> missing state-conditional baseline with a history-privileged pre-write value
> and pairs it with the value of the materialized afterstate.

Memory-R2/LoGo-GRPO already identifies unfair cross-rollout comparison after
memory states diverge and restores same-state comparison with local rerollouts.
Input-dependent policy-gradient baselines already establish that exogenous
input information can enter an action-independent baseline without bias. MIC
does not claim either discovery. Its proposed residual uses no local rerollout
inside online actor-training blocks: it is a time-safe amortized alternative that
simultaneously distinguishes arrived input from the policy-controlled
materialized afterstate. The one sealed formal branching oracle in Section 9 is
diagnostic licensing work, not an online credit producer. The direct experiment must
therefore include a protocol-compatible same-state local-rerollout competitor;
SparseBranch alone is not an adequate proxy for LoGo-GRPO.

MIC-v2 is falsified if writer actions have negligible same-state value
heterogeneity, if the cross-fitted critic cannot approximate branching-oracle
writer advantage, or if matched actor experiments show no boundary-specific
benefit over a stale post-to-post critic.

## 2. Formal decision boundary

For root-replica trajectory `i` and scheduled writer turn `t`, define:

- `H_it`: all exogenous information arrived through turn `t`, including the
  question, raw chunks through `t`, public metadata, and immutable identities;
- `M_i,t-1`: the previously materialized memory;
- `E_it^-`: any time-safe environment state realized before the current write;
- `X_it^- = (H_it, M_i,t-1, E_it^-)`: the privileged pre-write state;
- `O_it = psi(X_it^-)`: the actor-visible writer observation;
- `a_it = (w_it1, ..., w_itL) ~ pi_theta(. | O_it)`: the complete raw writer
  token action;
- `X_it^+ = T(X_it^-, a_it)`: the actual post-write environment afterstate;
- `R_i in [0,1]`: the frozen raw terminal task reward.

The initial post-stage state is explicitly

```text
X_i,0^+ = (question, empty arrived-chunk history, initial memory M_i,0,
           predeclared initial environment metadata).
```

It is assigned the root's outer fold, is predicted OOF, and is included as a
`V^+` training/evaluation row under the same representation and regularization
as later post-write states. Thus `D_i,1^PostPost = V_i,1^+-V_i,0^+` is defined;
there is no synthetic turn-zero writer action.

`X_it^+` must contain an audit-preserved nested history sufficient to recover
the pre-write information and the exact materialized memory. An overwritten
current-memory string alone is not a filtration. No raw writer surface form may
affect future execution through hidden caches, undeclared prompt fields, parser
side channels, or generation state after conditioning on `X_it^+`. If two raw
actions materialize the same declared afterstate but yield different future
environment dynamics, afterstate sufficiency fails and MIC is `NO-GO` until the
missing state variable is declared.

Let the increasing time-safe information sequence be

```text
F_i,t-1^+  subset  G_i,t^-  subset  F_i,t^+,
```

where `G_i,t^- = sigma(X_i,t^-)` and `F_i,t^+ = sigma(X_i,t^+)`. Under frozen
behavior policy `pi`, define

```text
V_i,t-1^+ = E_pi[R_i | F_i,t-1^+]
V_i,t^-   = E_pi[R_i | G_i,t^-]
V_i,t^+   = E_pi[R_i | F_i,t^+].
```

The three ledger components are

```text
D_i,t^chunk  = V_i,t^- - V_i,t-1^+
D_i,t^write  = V_i,t^+ - V_i,t^-
D_i^answer   = R_i - V_i,T^+.
```

They satisfy the interleaved decomposition

```text
R_i = V_i,0^+
      + sum_t D_i,t^chunk
      + sum_t D_i,t^write
      + D_i^answer.
```

Closure is an accounting identity, not evidence that the learned critic is
useful.

## 3. Oracle propositions and claim limits

### Proposition 1: writer action-gradient identity

Let

```text
S_i,t = grad_theta log pi_theta(a_i,t | O_i,t)
      = sum_m grad_theta log pi_theta(w_i,t,m | w_i,t,<m, O_i,t).
```

If the afterstate is sufficient for all downstream effects of the writer
action and the conditional values are exact, then

```text
E[S_i,t D_i,t^write] = E[S_i,t R_i].
```

Equivalently, the pre-write value is an action-independent baseline and the
post-write value is an afterstate action-value surrogate. MIC is not described
as an action-dependent baseline.

### Proposition 2: chunk score orthogonality

Because `D_i,t^chunk` is pre-action measurable,

```text
E[S_i,t D_i,t^chunk] = 0.
```

Consequently, an oracle post-to-post signal and oracle MIC have the same
expected score-function gradient. MIC does not correct an intrinsically biased
PostPost estimator; it removes non-action-local credit and may improve
finite-sample estimation and optimization.

### Proposition 3: raw-credit variance decomposition

Consecutive martingale increments are orthogonal in `L2`. Therefore, under the
oracle filtration,

```text
E[(D_i,t^PostPost)^2]
= E[(D_i,t^write)^2] + E[(D_i,t^chunk)^2],

D_i,t^PostPost = V_i,t^+ - V_i,t-1^+.
```

This is a raw-credit second-moment result. It does not imply unconditional
reduction of every coordinate of the score-weighted policy-gradient variance;
that stronger conclusion requires additional score-norm conditions and is not
claimed without evidence.

### Proposition 4: approximation error

For fold-exclusive predictions

```text
Vhat_i,t^- = V_i,t^- + e_i,t^-
Vhat_i,t^+ = V_i,t^+ + e_i,t^+,
```

the learned writer-credit error is

```text
Dhat_i,t^write - D_i,t^write = e_i,t^+ - e_i,t^-.
```

If `E[||S_i,t||^2] <= G_t^2`, then

```text
||E[S_i,t(Dhat_i,t^write-D_i,t^write)]||
<= G_t sqrt(E[(e_i,t^+-e_i,t^-)^2]).
```

Cross-fitting prevents a root's own outcome from fitting its predictions; it
does not make the critic oracle, eliminate related-root dependence, or guarantee
an unbiased learned policy gradient.

### Proposition 5: local PPO implementation boundary

MIC's theorem concerns the complete sequence score. The implemented PPO loss is
token-factorized. At the behavior point its derivative is proportional to the
sequence score when every sampled token receives the same detached sequence
credit and the denominator is fixed before sampling. Away from the behavior
point, tokenwise clipping is not whole-sequence clipping:

```text
sum_m phi_epsilon(r_m, A) != phi_epsilon(product_m r_m, A).
```

The method therefore claims behavior-point sequence-score consistency and a
local token-factorized PPO delivery, not exact sequence PPO.

## 4. Critic information, representation, and fitting

### 4.1 Information boundary

The actor is memory-limited; the critic is history-privileged and time-safe.
Both `X^-` and `X^+` may contain all raw chunks that have arrived, all prior
materialized memories needed for nested audit, turn, and public metadata. Stable
root/example/content identities are carried only as join, fold, deduplication,
and audit keys. They are excluded from every text serializer, hash feature,
actor-hidden probe, value-head covariate, and gate statistic. The critic inputs
exclude:

- future chunks or future environment state;
- gold/reference answers;
- terminal or dense reward fields;
- current-row outcomes or scores;
- generated final answers;
- future writer text;
- raw generation caches or undeclared writer-completion side channels.

### 4.2 Shared representation and symmetric heads

The primary preregistered representation is a no-gradient probe of the current
frozen behavior-policy checkpoint. It encodes question, arrived-chunk history,
and memory components separately, uses a fixed question-conditioned history
aggregation, fixed history--memory interactions, and a fixed random projection.
The representation extractor and projection are shared between stages.

Stage is not injected as a natural-language marker into the frozen encoder.
Two symmetric low-capacity heads are fit:

```text
Vhat^-(z(X^-), t)
Vhat^+(z(X^+), t).
```

They have identical functional class, feature dimension, regularization grid,
target, fitting weights, and stopping rule. With continuous `R in [0,1]`, the
finite E1-dev candidate set is exactly:

1. turn, arrived-chunk count, prior active-turn count, arrived-context token
   count, current materialized-memory token count, and intercept only;
2. signed text hashing with dimension `4096` and seed `20260825`;
3. frozen actor-hidden component probes followed by a fixed Rademacher
   projection of dimension `128`;
4. the same actor-hidden probe with projection dimension `256`.

For candidates 2--4 the head is either L2 fractional logistic or bounded ridge,
with `lambda in {1e-4,1e-3,1e-2,1e-1,1,10}`. The turn/length model uses both
head families and the same grid. One candidate specification consists of one
representation, one head family, and one lambda shared by both `V^-` and `V^+`.
Its unique E1-dev selection score is computed from exactly four stable
content-root folds, `H(content_root,"e1-selection") mod 4`, and is

```text
0.5 * [root-weighted four-fold OOF MSE(V^-) +
       root-weighted four-fold OOF MSE(V^+)].
```

Root weighting follows the declared root -> trajectory -> turn-pair rule. Ties
within `1e-6` choose, in order: smaller feature dimension, larger lambda,
fractional logistic before bounded ridge, then the candidate order 1--4 above.
Projection entries and the exact hash implementation are sealed before E1-dev.
No independent learned encoder, feature, dimension, head family, or regularizer
value may be added after E1-dev begins.

Head semantics are fixed. Fit-fold numeric features are centered and divided by
their fit-fold standard deviations; a zero standard deviation maps to scale one.
The transformation is applied unchanged to heldout rows. The intercept is never
penalized, fitting weights sum to one, and `lambda` multiplies the squared L2 norm
of non-intercept coefficients. Fractional logistic minimizes Bernoulli
cross-entropy with the continuous target `R in [0,1]` and predicts with the
logistic link. Bounded ridge minimizes squared error in closed form and clips
only its final predictions to `[0,1]`. Optimization tolerance, maximum
iterations, linear-algebra backend and version are manifest-frozen before
E1-dev; non-convergence or a non-finite solution fails that candidate rather
than invoking an alternative solver.

No candidate sees the current raw writer completion, its sampled-token count,
parse route, generation stop reason, or any statistic computed from those fields.
For `V^-`, “current materialized memory” is `M_t-1`; for `V^+`, it is `M_t`.
This is the only phase-varying memory component, and both stages use the same
serializer and feature extractor.

Post-write probes are reconstructed from the declared materialized state. They
may not reuse generation hidden states or KV caches. The behavior-policy
checkpoint, serializer, tokenizer, probe token, component order, pooling,
projection seed and matrix hash, head specification, and feature hashes are
bound in every block receipt.

### 4.3 Root-cross-fitted block estimator

Each actor block freezes one behavior policy and collects:

```text
64 roots x 4 independent trajectory replicas = 256 trajectories.
```

The default is four stable outer folds and three inner folds. Fold assignment is
a global stable hash of canonical root/content-cluster identity, never a
set-relative round robin. All aliases, duplicates, and rows sharing a content
root remain in the same fold. The target is raw terminal return, not GRPO group
advantage, dense reward, S128 performance, or a post-hoc label.

Each outer-heldout root receives only predictions from heads fit without that
root or content cluster. After E1-dev, every actor block uses the single frozen
E1-selected representation, head family and lambda. Inner folds refit only that
specification and determine the reliability gate; they never perform model or
hyperparameter selection. Critic loss uses root -> trajectory -> turn-pair equal
weighting so roots with more turns do not dominate fitting.

The sequence for every block is strictly

```text
freeze behavior policy
-> collect complete block
-> seal raw trajectory/outcome bundle
-> nested root cross-fit
-> seal OOF values, credits, gates and critic checkpoints
-> actor PPO epoch using sealed credits
-> next behavior-policy checkpoint.
```

No actor minibatch may update before the entire block's OOF artifacts are
sealed.

## 5. Writer credits and comparison cells

Let `G=4` trajectory replicas and `c_G=(G-1)/G=3/4`. The matched broadcast
advantage is exactly

```text
A_i^broadcast = R_i - 1/G sum_j R_j
              = c_G [R_i - 1/(G-1) sum_j!=i R_j].
```

No sample-standard-deviation division, whitening, clipping, or per-group
variance normalization is applied to any main-cell credit. This makes the
`c_G` scale explicit and prevents a low-variance reward group from silently
amplifying one cell. Define

```text
Dhat_i,t^MIC = Vhat_i,t^+ - Vhat_i,t^-
Dhat_i,t^PP  = Vhat_i,t^+ - Vhat_i,t-1^+.
```

The primary cells are:

| Cell | Critic work | Writer credit `C_i,t` | Purpose |
|---|---|---|---|
| Broadcast-Ghost | fit identical `V^-`,`V^+`; discard both | terminal broadcast `A_i^broadcast` | fully compute-matched control |
| PostPost-Ghost | fit identical `V^-`,`V^+`; discard `V^-` in actor | `c_G Dhat_i,t^PP` | stale-boundary critic control |
| MIC-core | fit identical `V^-`,`V^+` | `c_G Dhat_i,t^MIC` | primary method |
| MIC-Gated | fit identical `V^-`,`V^+` | `(1-lambda_f,t) A_i^broadcast + lambda_f,t c_G Dhat_i,t^MIC` | secondary reliability wrapper |

Final-answer actions in the four cells in the table retain `A_i^broadcast`. The answer residual
`R_i-Vhat_i,T^+` is retained for decomposition and mechanism diagnostics, not
used by the primary actor update.

The gate is binary and selected only from the outer-training roots' nested
inner-CV evidence. An outer-heldout outcome is never visible. For outer fold
`f`, turn `t`, `lambda_f,t=1` only if all of the following are true on sealed
inner-OOF predictions from the selected E1 model family:

- at least 24 distinct content-root clusters have an active turn `t`;
- `V^+` improves root-weighted raw-return MSE over the turn/length model by at
  least 5%;
- both `V^-` and `V^+` have calibration slopes in `[0.5,1.5]` and absolute
  intercepts at most `0.05`;
- in each of four pre-value quartiles, the absolute conditional mean of
  `Dhat^MIC` is at most `0.03` and its root-bootstrap 90% interval lies entirely
  inside the equivalence margin `[-0.05,0.05]`; all four quartiles must pass;
- writer-credit variance is at least `0.0009` and every value/credit is finite.

Any failure sets `lambda_f,t=0`; turns are not pooled, thresholds are not
relaxed, and an undersized cell is never skipped. Gate opening rate, fold/turn
decisions, and the fraction of writer tokens falling back to broadcast are
mandatory. `MIC-Gated` is not called “safe” and does not claim guaranteed policy
improvement. If only Gated improves, the paper must be reframed as an
evidence-gated hybrid; it cannot claim broad MIC-core validity. A gate open on
fewer than 25% of active writer slots across B1--B8 is critic-delivery `NO-GO`.

All four cells in the table fit the same ghost critic pipeline with the same
frozen hyperparameters. This prevents critic compute, block delay, and
representation extraction from being confounded with writer-credit delivery.

## 6. Fixed action-slot PPO normalization

### 6.1 Scheduled slots

Every root-replica has a manifest-frozen action-slot program

```text
S_i = {(i,1), ..., (i,T_max), (i,answer)}.
```

For the frozen MemAgent protocol, `T_max=8`. It is fixed before sampling. A
missing future writer turn contributes zero
loss and zero score; no fictitious no-op action or EOS is inserted. If absence
is policy-induced, its effect must flow through the preceding realized action
and afterstate.

For `B` roots and `G` replicas,

```text
N_slot_global = B G (T_max + 1)
```

is fixed before generation and does not depend on realized writer or answer
length.

### 6.2 Token-factorized loss

For a sampled token, let

```text
r_theta = exp(log pi_theta(token | prefix) - log pi_old(token | prefix))
ppo(r,A) = min(r A, clip(r,1-epsilon,1+epsilon) A).
```

All credits are detached:

```text
C_i,t = stopgrad(C_i,t)
A_i^broadcast = stopgrad(A_i^broadcast).
```

The global policy-gradient loss is

```text
L_PG = -1 / (N_slot_global Lbar_ref) * [
          sum_i,t,m ppo(r_i,t,m, C_i,t)
          + sum_i,m ppo(r_i,m^answer, A_i^broadcast)
       ].
```

For every fixed-slot mechanism cell, the complete actor objective uses the same
denominator for policy gradient, KL, and entropy:

```text
L_actor = L_PG
          + beta_KL  /(N_slot_global Lbar_ref) * sum_valid_tokens KL_token
          - beta_ent /(N_slot_global Lbar_ref) * sum_valid_tokens H_token.
```

`beta_KL` and `beta_ent` are copied from the accepted training manifest and are
identical across cells. They are never renormalized by realized token count.
FullBranch-4-Matched replaces `N_slot_global` by its frozen global-plus-local
slot count and includes regularizers only for global actions and selected local
writer actions that enter its actor loss; reward-only continuation tokens do not
enter any actor term. LoGo-GRPO-Port is the sole exception: as part of its
delivery-faithful port, it retains the official mean token-level entropy and KL
terms alongside the valid-step actor mean. That exception is an explicit
algorithm difference, not a hidden normalization choice.

`Lbar_ref` is one positive preregistered constant that changes numerical scale
only. The sole allowed producer is a sealed, reward-blind fresh-base calibration
rollout over 64 training-disjoint roots and four independent replicas under the
final writer/answer generation and slot manifest. It is the arithmetic mean of
sampled policy-token counts over all scheduled writer and answer slots, with an
inactive slot contributing zero tokens. The producer reads no reward, gold,
critic target, S128, E1, oracle, or confirm field. Its roots, seeds, generation
manifest, statistic code SHA, output SHA, and resulting value are committed
before E1-dev. The value is shared by all blocks, seeds, and cells and is never
re-estimated. Because it is a single global scalar, it changes numerical step
size only and is reported as equivalent to a fixed learning-rate rescaling.

Every policy-sampled writer token contributes to the sequence score, including
sampled EOS when present. Padding, system-appended delimiters, parser-generated
characters, and a fictitious EOS after forced truncation do not. The raw action
bytes/tokens, sampled mask, parsed memory, parser version, and afterstate hash
are all receipted.

### 6.3 Behavior-point identity

For one detached scalar credit `D`,

```text
grad_theta ppo(r_theta,D) at theta=theta_old
= D grad_theta log pi_theta_old(token | prefix).
```

Therefore the gradient of the minimized negative objective carries the minus
sign, and summing sampled writer tokens recovers the complete writer sequence
score up to the single fixed global scale. This statement differentiates
`r_theta`; it never differentiates a constant expression `ppo(1,D)`.

### 6.4 Distributed reduction

The loss scale must be derived from the actual distributed gradient reduction.
If `W` ranks compute local numerators and DDP averages gradients, an equal-shard
implementation uses

```text
L_rank = -W / (N_slot_global Lbar_ref) * local_numerator.
```

Unequal shards require an equivalent globally reconstructed sum. Receipts record
world size, global and per-rank scheduled slots, active role slots, token counts,
local numerators, pre-DDP scale, reduction mode, and reconstructed global loss.
No coefficient may be inferred post hoc from observed gradient norms.

### 6.5 Optimization isolation

Within an actor PPO epoch, the following are immutable:

- behavior-policy representation artifacts;
- `V^-` and `V^+` predictions and head checkpoints;
- outer/inner fold assignments and gate decisions;
- old log probabilities;
- writer and answer credits;
- the append-only credit ledger.

No actor backward path may enter critic heads, representation extraction, gate
logic, or credit construction. Critic/head parameter identities must be absent
from the actor optimizer, not merely have numerically zero gradients; this also
prevents decoupled weight decay or optimizer state from changing them. Actor
backward requires `critic_grad is None`, no critic optimizer step, and unchanged
artifact/checkpoint hashes for the full block.

## 7. Direct competitors and bounded pressure tests

### 7.1 PostValue-Sibling-CF

At each turn, the formal secondary competitor uses

```text
C_i,t^PV-sib = c_G [
    Vhat^+(X_i,t^+) - 1/(G-1) sum_j!=i Vhat^+(X_j,t^+)
].
```

Sibling replicas use independent trajectory seeds. The sibling term can be a
valid action-independent control variate, but after recurrent divergence it is
not the policy mean at trajectory `i`'s own pre-write state. It is a serious
competitor, not an intentionally invalid ablation. If it matches MIC, the paper
may call `V^-` a principled state-conditional alternative, not empirically
necessary.

### 7.2 LoGo-style FullBranch-4 family: direct same-state rerollout competitors

The closest direct comparison family is a protocol-compatible adaptation of
Memory-R2's LoGo-GRPO, pinned to official arXiv
[`2605.21768v1`](https://arxiv.org/abs/2605.21768v1), PDF SHA256
`d3944bfc1887bd154aaed73bb230afd05dbeb2cdde34e537a9b9bd8ec55e2756`,
source-tar SHA256
`4222683404c2c4f3acd0e8aacca59f128e7d8e6823da1b2821736401bfd9902b`,
and official repository commit
`352896d2260da4e14ae771541e630209486f6777`. It is called
`LoGo-style FullBranch-4`, not `Memory-R2 reproduction`, because the original
method assumes session-specific QA subsets and a separate extractor--manager
memory system. MemAgent/HotpotQA has one terminal multi-hop answer and no
non-leaking per-chunk QA target. Inventing such targets would change the reward
and estimand.

Both adaptations preserve the part germane to the MIC claim:

- before any action or outcome is observed, exactly 25% of the
  `64 roots x T_max=8` root--turn positions are selected by the lowest stable
  hashes `H(block,root,turn,"logo-local-select")`; there is no random realized
  selection count or outcome-based fallback;
- for each selected position, one anchor global replica is fixed as
  `H(block,root,turn,"logo-anchor") mod 4`, and its exact `X^-` snapshot is
  restored four times;
- the four local arms use independent writer-action keys
  `H(experiment_seed,block,root,turn,"logo-writer-arm",j)`,
  `j in {0,1,2,3}`;
- each arm is continued from its materialized `X^+` through all remaining
  writer turns and the answer to the real terminal reward, with an independent
  counter-based future stream
  `H(experiment_seed,block,root,turn,"logo-future",j,future-turn,role)`;
- no gold, future chunk, answer, reward, sibling action, or sibling afterstate is
  visible before its declared environment time.

Selection and anchor mappings deliberately omit `experiment_seed`, so matched
cells and seeds branch the same structural positions. Action and future streams
include it, so fresh experiment seeds never reuse generated trajectories. Within
one experiment seed, FullBranch-4-Matched and LoGo-GRPO-Port use identical local
counter keys, providing paired common randomness across the two cells.

For selected boundary `x` and arm `j`, define the matched unstandardized and
official-style standardized local credits

```text
C_x,j^L    = R_x,j - 1/4 sum_k R_x,k
Ahat_x,j^L = C_x,j^L / [std_k(R_x,k) + 1e-6].
```

For the four ordinary global replicas of root `i`, LoGo-GRPO-Port uses

```text
Ahat_i,g^G = [R_i,g - 1/4 sum_h R_i,h] /
             [std_h(R_i,h) + 1e-6].
```

The sample standard-deviation convention (`correction=1`) is frozen for both
global and local groups. A zero-variance group receives exactly zero credit; it
is never divided into numerical noise.

Local continuation writer/answer sequences are reward-evaluation work, not
additional actor-loss slots. This corresponds to LoGo-GRPO's shared-start local
comparison when a MemAgent chunk contains one policy-controlled writer call,
while avoiding an unavailable session-local surrogate reward.

Two independently trained cells use the identical frozen branch schedule:

1. **FullBranch-4-Matched.** The detached `C^L` is delivered only to the
   complete sampled writer sequence at the selected boundary. The ordinary
   global branch is exactly Broadcast-Ghost. This cell uses the common MIC
   token-factorized loss and the fixed normalization below. It isolates an
   amortized critic from explicit same-state rerollout under matched delivery.
2. **LoGo-GRPO-Port.** Global and local group credits are divided by their own
   group reward standard deviation with `epsilon=1e-6`. Each complete generated
   action sequence `u` uses the official geometric-mean step ratio and
   dual-clipped loss:

```text
rho_u = exp[1/L_u sum_m log r_u,m]
if Ahat_u < 0:
    ell_u = min(-3*Ahat_u,
                max(-rho_u*Ahat_u,
                    -clip(rho_u,0.8,1.2)*Ahat_u))
else:
    ell_u = max(-rho_u*Ahat_u,
                -clip(rho_u,0.8,1.2)*Ahat_u)
L_PG,Port = mean over realized valid global and local action sequences of ell_u.
```

   Local `Ahat^L` is delivered to the selected writer sequence; standardized
   global group credit is delivered to all ordinary global writer and answer
   sequences. This intentionally retains the official valid-step denominator
   and sequence-ratio objective rather than the MIC fixed-slot estimator. It is
   the delivery-faithful port. Differences from original Memory-R2 remain the
   task/reward adaptation, four rather than eight global replicas, and absence
   of its extractor--manager architecture and curriculum. Those differences are
   reported, never hidden under a reproduction claim.

   Its complete objective is

```text
L_actor,Port = L_PG,Port
               + beta_KL  * mean_valid_actor_tokens KL_token
               - beta_ent * mean_valid_actor_tokens H_token.
```

   The valid-token set contains ordinary global writer/answer actions and
   selected local writer actions only; reward-evaluation continuation tokens are
   excluded. `beta_KL` and `beta_ent` use the accepted MemAgent manifest values
   for backbone-controlled comparison rather than Memory-R2's task-specific
   coefficients; this coefficient choice is explicitly listed as another port
   difference. The regularizers occur exactly once and are not already included
   in `L_PG,Port`.

For FullBranch-4-Matched, the actor numerator is the sum of the ordinary
fixed-slot global numerator and all selected local writer numerators. Its
denominator is fixed before rollout:

```text
Lbar_ref * [64*4*(T_max+1) + 4*N_selected_local_boundaries].
```

Inactive scheduled global slots and nonexistent selected writer turns contribute
zero but retain their denominator slots. A selected nonexistent writer turn
launches no branch and contributes four zero local slots. Thus selection,
termination, and generated lengths cannot change the Matched normalization.
There is no separate mixing coefficient or post-result choice: denominator
allocation between global and local slots is determined solely by frozen slot
counts. Realized gradient contributions still depend on detached credits,
sequence scores, and sampled action lengths. This is a matched mechanistic
control, not official LoGo delivery.

Both FullBranch-4 cells run from the same fresh base with three seeds and save a
checkpoint after every actor block. They are compared to MIC-core and
Broadcast-Ghost on common support. Mandatory cost coordinates are terminal
environment trajectories, generated writer/answer tokens, total model
forward/backward tokens, H20-hours, and wall time. Because a selected local arm
requires a full continuation, FullBranch may reach fewer actor blocks under the
same cap. No interpolation or last-observation carry-forward from above a cap is
allowed.

The common-support decision is fixed before confirm is opened. Resource caps are
the measured H20-hours of MIC-core at B1, B2, B4 and B8, separately within each
seed. At each cap, each FullBranch cell contributes its latest preregistered
checkpoint not exceeding that same-seed cap. A selected checkpoint is admissible
only if the unused cap is smaller than the measured H20 cost of its next complete
actor block; otherwise the cap comparison is `MISSING_COMMON_SUPPORT`. Thus the
ordinary cap statement is precisely “higher/lower F1 under a paired seed-specific
H20-hour cap,” not generic compute dominance.

For two admissible checkpoints, method A Pareto-dominates B only if
`F1_A >= F1_B`, all five declared cumulative cost coordinates for A are no
greater than B, and at least one of those six inequalities is strict. Any other
tradeoff is reported as non-dominance. This definition also governs every later
SparseBranch use of “Pareto-dominates.” The final-cap paired F1 contrast uses the
same 10,000 crossed root-bootstrap draws as Section 10.2. A positive practical
amortization statement additionally requires, for MIC against **each**
FullBranch cell: an admissible final-cap checkpoint in every seed; mean
`F1_MIC-F1_FullBranch >= 0.005`; all three seed differences positive; one-sided
95% root-bootstrap lower bound above zero; and
`cumulative_model_tokens_MIC <= cumulative_model_tokens_FullBranch` at every
paired selected checkpoint. These are secondary licensing gates, outside the
two-hypothesis primary family. A missing comparator or any failure forbids the
compute-amortization claim but does not negate the action-boundary result.

If either FullBranch cell Pareto-dominates MIC, practical amortization fails. If
MIC passes the positive licensing gates against both adaptations, the licensed
statement is limited to this MemAgent terminal-reward setting; it does not
establish superiority over original Memory-R2, its session-local reward, or
local rerollout methods in general.

### 7.3 Broadcast + SparseBranch-2x1 correction

This B4, three-seed pressure test is a bounded direct challenge, not a complete
branching algorithm or oracle upper bound. Exactly one of the four factual
replicas per root is branched: `replica_star = H(root_id,"branch-replica") mod
4`. Its fixed scheduled turn is `tau = 1 + H(root_id,replica_star,"branch-turn")
mod T_max`. Both mappings are computed before rollout and recorded. The method
may not select among turns that happen to remain active after observing the
trajectory. If the scheduled turn does not exist, the preregistered behavior is
no correction, not outcome-based fallback.

The replica and turn mappings deliberately omit `experiment_seed`, keeping the
structural pressure test paired across the three seeds; all generated action and
future streams below include `experiment_seed`, so trajectories remain fresh.

At `tau`, the selected factual replica is replaced by two writer actions sampled
from the same pre-write state. The other `G-1` root replicas remain unchanged.
The two writer actions use independent counter keys
`H(experiment_seed,root,replica_star,tau,"writer-arm",arm)` for
`arm in {0,1}`. The two leaves
then use counter-based common future randomness keyed separately by
`(experiment_seed,root,replica_star,tau,future-turn,role,future-seed-index)`
with no arm field,
so future randomness is shared across arms while different token lengths cannot
shift a global RNG stream. Their pair credit is

```text
C_1 = c_G (R_1-R_2)
C_2 = c_G (R_2-R_1).
```

For leaf `k`, define its branch-specific broadcast advantage against the other
unchanged replicas as

```text
A_k^branch = c_G [R_k - mean(return of the other G-1 root replicas)].
```

The selected replica retains one scheduled-slot weight in the global
denominator. Its common trunk contributes once with credit
`(A_1^branch+A_2^branch)/2`. The selected writer slot contributes

```text
1/2 * [sum_m ppo(r_1,m, C_1) + sum_m ppo(r_2,m, C_2)].
```

Every downstream writer/answer slot similarly contributes one half of each
leaf's token loss with its branch-specific broadcast advantage. Thus the two
leaves jointly retain the weight of one original replica; common-prefix actions
are never duplicated, and the behavior-point selected-turn contribution is

```text
c_G/2 [S_1(R_1-R_2) + S_2(R_2-R_1)].
```

Receipts distinguish trunk roots, branch leaves, writer and continuation tokens,
model forward/backward tokens, H20-hours, and wall time. SparseBranch is judged
on a compute/sample Pareto curve. If this exact bounded method Pareto-dominates
MIC under Section 7.2's definition, it is a strong counterexample to MIC's practical
amortization claim; if it loses, no claim is made about all branching methods.

### 7.4 Native token-normalized Broadcast

Realized-active-token-normalized Broadcast is retained only as a short B2
normalizer diagnostic. It is not the matched primary baseline. The historical
certified Original curve remains read-only contextual evidence and cannot
replace fresh `Broadcast-Ghost`.

## 8. E0: oracle and implementation suite

All E0 tests must pass before E1 or GPU actor training. Numeric oracle checks use
float64 and fail at maximum closure error above `1e-12` unless a stricter
machine-precision bound is specified.

| ID | Test | Required PASS evidence |
|---|---|---|
| E0-1 | Action boundary and oracle decomposition | Analytic toy with separate chunk arrival and writer action recovers `D^chunk`, `D^write`, answer residual, and full closure |
| E0-2 | Branching oracle gradient | Enumerated same-state actions/futures recover the oracle writer policy gradient; shuffled actions fail |
| E0-3 | Filtration and afterstate sufficiency | overwrite-only state is rejected; nested histories pass; raw-action/cache/parser side-channel attacks are rejected |
| E0-4 | Stable root OOF | global hash folds invariant to batch membership; aliases/content duplicates cannot cross folds; held-out outcome never enters fit |
| E0-5 | Behavior-point score identity | autograd of `ppo(r_theta,D)` at old policy equals `D` times the sampled-token score; minimized loss has the negative sign |
| E0-6 | Length, termination, and inactive slots | sampled EOS, forced truncation, empty slots, and policy/exogenous termination produce exact declared masks; no fake actions |
| E0-7 | Cross-slot length isolation | changing valid length or masked padding in an unrelated answer slot cannot rescale a fixed writer slot |
| E0-8 | Batch duplication | duplicating the complete synthetic batch and doubling `B,N_slot` leaves the normalized global gradient unchanged |
| E0-9 | Distributed reconstruction | single-process reference and two-rank distributed gradients match for unequal active lengths/slots; rank receipts reconstruct global loss |
| E0-10 | Optimization isolation | actor backward leaves critic/representation gradients absent, excludes their parameters from actor optimizer, and preserves all credit/checkpoint hashes |
| E0-11 | Role routing and broadcast scale | writer and answer masks are disjoint; the four Section 5 cells always give answer the exact unstandardized group-centered broadcast; `c_G` reconstruction is exact; every writer cell receives only its declared detached credit; LoGo-GRPO-Port alone follows its separately declared standardized global delivery |
| E0-12 | SparseBranch accounting | fixed `tau`, common trunk once, two leaf actions, future credit, and compute ledger close without prefix duplication |
| E0-13 | FullBranch-4 accounting | frozen 25% root--turn schedule and anchor, four same-`X^-` writer arms, independent counter streams, terminal continuations, and complete compute ledger reconstruct exactly; Matched recovers its fixed global+local denominator while Port recovers standardized credits, geometric-mean ratios, dual clipping, and valid-step mean |

Negative controls include future/gold/outcome feature taints, set-relative folds,
generation-cache reuse, shuffled state/outcome association, random denominators,
and rank-local denominators. A test that merely searches source strings does not
count as entry-level rejection evidence.

## 9. E1 and branching-oracle feasibility

### 9.1 E1-dev and E1-holdout

E1 uses fresh on-policy data from one frozen fresh-base behavior policy:

```text
E1-dev:     128 roots x 4 independent replicas
E1-holdout: 128 roots x 4 independent replicas.
```

The two splits are content-disjoint from each other, S128, the branching oracle,
actor training, calibration roots, and final confirm. E1-dev selects exactly one
member of the finite representation--head set in Section 4.2. E1-holdout is
opened once after the selected specification and all gates are sealed; it cannot
trigger a capacity increase, new encoder, new feature, new grid value, threshold
change, or additional E1 sample collection.

After selection, the chosen specification and the frozen turn/length comparator
are each refit on all E1-dev roots and predict E1-holdout without any holdout
refit, recalibration, intercept correction, or early stopping. Every holdout root
therefore has exactly one prediction from a model that saw no holdout outcome.

E1 reports fold/turn/root-clustered intervals for raw-return prediction,
calibration, writer-credit magnitude, gate coverage, and failure cells. A post
state's lower MSE alone is insufficient: pre/post capacity is symmetric and
paired comparisons use the same roots, turns, target, and weighting.

Minimum gates identifiable from ordinary E1-holdout trajectories are:

- both pre- and post-head root-weighted raw-return MSE improve over the frozen
  turn/length model by at least 5%, with paired root-cluster bootstrap 95% lower
  bounds above zero;
- replacing pre-write by post-write state reduces paired root-weighted
  raw-return MSE by at least 5%, with 95% lower bound above zero;
- both heads have calibration slope in `[0.5,1.5]` and absolute intercept at
  most `0.05`;
- every occupied fold--turn cell contains at least 24 distinct content roots,
  otherwise that turn is ineligible for Gated delivery; at least 75% of active
  scheduled-turn cells must be eligible for the route to continue;
- all value, credit, calibration, coverage, and closure fields are finite and
  the leakage/firewall audit is PASS.

E1-holdout cannot identify same-state action advantage, so it never reports or
gates on branching rank, sign, or advantage MSE. Those labels exist only in the
separate formal oracle below.

E1 is necessary but not sufficient. Actor experimentation is licensed only by
the conjunction `E0 PASS AND E1-holdout PASS AND formal-branching-oracle PASS`;
none alone proves a causal effect or an unbiased learned gradient.

Every E1 and formal-oracle root-bootstrap interval uses 10,000 paired
content-root resamples with seed `20260825`. A two-sided percentile 95% interval
uses the 2.5th and 97.5th percentiles; a stated one-sided 95% lower bound uses
the fifth percentile. The same root draw is used for every method/predictor in a
paired contrast. No action row, trajectory replica, turn, or future seed is
resampled as an independent root cluster.

### 9.2 Split branching oracle

The formal oracle uses

```text
64 pre-write states x 4 writer actions x 8 common future seeds
= 2048 continuations.
```

States are produced by fresh-base behavior rollouts on exactly 64 preregistered
content roots, one state per root. A stable root hash assigns a scheduled turn
before rollout, stratified equally across the frozen strata `{1,2}`, `{3,4}`,
`{5,6}`, and `{7,8}`. Turn
availability is determined from the exogenous chunk schedule before writer
generation; an unavailable root--turn pair is a manifest failure and is not
replaced after outcomes. These roots are content-disjoint from E1-dev,
E1-holdout, actor training, calibration, S128, and confirm. At each frozen
pre-write state, four writer actions are sampled independently from the frozen
behavior policy with stable action seeds. Duplicate raw or materialized actions
are retained with their sampling multiplicity and never resampled; their rate
is reported. Eight future seeds are assigned by stable hash before outcomes and
split into fixed halves `A` and `B` of four seeds each. Future randomness is
counter-keyed by state, future turn, role and seed index and shared across
action arms.

For actions `k,j`, let `d_x,kj^A` be the difference between their mean returns
over the four `A` seeds, and define `d_x,kj^B` analogously. The split-pair
action-variation estimator uses

```text
Theta_action_hat = average_x,k<j d_x,kj^A d_x,kj^B,
```

avoiding naive noise-variance subtraction and winner's-curse reuse. Inference
bootstraps pre-write states; action rows are not treated as independent.

Action importance requires both:

- `Theta_action_hat >= 0.0025` and its 95% lower bound above `0.0009`;
- cross-evaluated top--bottom return gap mean at least `0.05` and 95% lower
  bound at least `0.02`.

For the cross-evaluated gap, half A chooses the top and bottom actions and half B
evaluates their mean-return difference; the roles are reversed, and the two
held-half gaps are averaged. The prevalence of states with gap at least `0.10`
is reported, not used as a hard gate. Leave-one-out oracle action credit is

```text
A_x,k^LOO = Q_x,k - 1/(K-1) sum_j!=k Q_x,j.
```

After E1-holdout PASS, the selected critic specification is refit once on the
union of E1-dev and E1-holdout, without changing hyperparameters, and evaluated
on the disjoint oracle states. Oracle credit licensing additionally requires:

- Spearman correlation between `Dhat^MIC` and eight-seed `A^LOO` at least
  `0.30`, with state-bootstrap 95% lower bound above zero;
- cross-half sign accuracy at least `65%`: A selects pairs with
  `abs(d^A)>=0.05`, whose predicted pair-credit sign is scored against `d^B`;
  B selects and A scores analogously; the two held-half accuracies are averaged,
  with state-bootstrap 95% lower bound above `50%`;
- at least `10%` paired MSE improvement over the zero-advantage predictor, with
  state-bootstrap 95% lower bound above zero;
- calibration slope in `[0.5,1.5]` and absolute intercept at most `0.05` for
  predicted versus eight-seed LOO advantage.

Branching outcomes license or reject actor experimentation but never fit the
critic, choose an online fold/turn gate, update the actor, tune thresholds, or
enter E1-dev. In addition to rank/MSE/calibration, the oracle reports a
preregistered 16-dimensional Rademacher sketch (seed `20260825`) of writer
score-gradient alignment; rank alone cannot establish alignment with the
policy-gradient vector. The sketch is mechanism evidence, not an additional
hard gate.

Failure of action importance means the route lacks a meaningful writer-credit
object in this setting and is `NO-GO`, not an invitation to increase critic
capacity.

## 10. Actor experiment and primary estimand

### 10.1 Main budget

Each block contains 64 roots and four independent trajectory replicas, seals all
256 trajectories, performs nested cross-fitting, and then runs exactly one PPO
epoch over a frozen minibatch order. Checkpoints are evaluated at

```text
B0, B1, B2, B4, B8.
```

The three primary cells `Broadcast-Ghost`, `PostPost-Ghost`, and `MIC-core` run
to B8 with three fresh seeds. `MIC-Gated` and `PostValue-Sibling-CF` also run to
B8 with three seeds as formally secondary experiments. Native token-normalized
Broadcast stops at B2 with three seeds. SparseBranch stops at B4 with three
seeds. FullBranch-4-Matched and LoGo-GRPO-Port run under the common-support
compute rule in Section 7.2 with three seeds; each terminal block is whichever
preregistered checkpoint is the last not exceeding the matched resource budget,
never a performance-selected checkpoint.

All matched cells share fresh base revision, root order, behavior seeds, reward,
tokenizer, writer/answer generation protocol, slot program, minibatch order,
PPO epochs, optimizer hyperparameters, and checkpoint schedule. Only the
declared writer credit/delivery and the explicitly named FullBranch/SparseBranch
environment work may differ.

### 10.2 Confirmation estimand

The primary performance estimand is normalized token-F1 learning-curve AUC on a
sealed confirm set, using cumulative environment trajectories on a linear axis:

For cell `c` and seed `s`, with `F_k=F1_c,s(B_k)`, the non-equidistant
trapezoid is exactly

```text
AUC_c,s = [0.5(F_0+F_1)
           + 0.5(F_1+F_2)
           + 1.0(F_2+F_4)
           + 2.0(F_4+F_8)] / 8.
```

The axis is divided by the common B8 trajectory total, so AUC has F1 units.
Including common B0 does not change pairwise AUC differences; improvement-over-
B0 AUC is reported as an equivalent readability view. Primary inference is
`MIC-core - Broadcast-Ghost`; `MIC-core - PostPost-Ghost` is the boundary
mechanism contrast. Normalized EM is secondary and format validity is a health
metric.

Confirm is opened once, only after every cell/seed checkpoint and manifest is
frozen. All anchors are evaluated in that single opening to compute the
predeclared AUC; no best-anchor selection is licensed. A missing, corrupt, or
non-finite anchor fails that cell/seed and is never interpolated, carried
forward, or replaced.

Each `AUC_c,s` is first computed within cell and seed from the same 512 confirm
roots. Contrasts pair cells within seed. The three seeds are fixed replication
conditions, not a random sample from a seed population. The preregistered
uncertainty procedure is a crossed paired root bootstrap with 10,000 replicates
and seed `20260825`: each replicate draws one content-root index vector and uses
that identical draw across all three fixed seeds, cells, and anchors, then
recomputes root-level metrics, each seed-specific AUC contrast, and their
arithmetic mean. Seeds are never resampled. The report gives the two-sided
percentile 95% root-population interval conditional on these three seeds;
confirmatory criteria use its one-sided 95% lower bound, the fifth percentile.
No seed-population or population-wide algorithm claim is licensed.

There is one primary performance hypothesis:

```text
mean_s(AUC_MIC-core,s - AUC_Broadcast-Ghost,s) >= 0.010,
all three seed contrasts > 0,
and crossed paired-root bootstrap 95% lower bound > 0.
```

The key boundary mechanism hypothesis is

```text
mean_s(AUC_MIC-core,s - AUC_PostPost-Ghost,s) >= 0.005,
at least two of three seed contrasts > 0,
and crossed paired-root bootstrap 95% lower bound > 0.
```

The `0.010` and `0.005` requirements are smallest-effect-size-of-interest
(SESOI) point-estimate gates. The one-sided `alpha=0.05` tests concern only the
null contrast `Delta<=0`; statistical significance does not waive the SESOI or
seed-sign requirements.

The full action-boundary claim is an intersection--union claim: both
preregistered one-sided criteria must pass at `alpha=0.05`. Because the claim is
licensed only by their conjunction, no additional multiplicity correction is
applied between them; no other contrast is promoted into this confirmatory
family. MIC-core B8 token-F1 must additionally be no worse than
Broadcast-Ghost by more than `0.010`, and format validity may not fall by more
than `0.02`. Failure of the boundary test permits a generic critic result but
not the MIC action-boundary claim. All other cell comparisons are secondary and
their intervals are labelled exploratory rather than added to this family.

Primary sample efficiency uses environment trajectories for the matched main
cells. Generated writer tokens, answer tokens, total model forward/backward
tokens, H20-hours, peak memory, and wall time are mandatory secondary axes.
Both FullBranch-4 cells and SparseBranch are compared on Pareto curves and
common-support budgets rather than inserted into the main AUC family.

## 11. Data isolation and adaptive-use contract

The historical fixed S128 has been fully revealed and is permanently a
development benchmark. It is not MIC-v2 confirm and may not select
representation, head, gate, block size, normalizer, budget, seed, checkpoint, or
failure revision.

Before any run, canonical root IDs and content hashes must prove pairwise
disjointness among:

- actor-training roots;
- critic-fit/E1-dev roots;
- E1-holdout roots;
- branching-oracle roots;
- reference-length calibration roots;
- S128 and all previously viewed selection data;
- sealed confirm roots;
- external-validity roots.

Both root content `(question,context)` and full example
`(question,context,ground truth)` intersections are reported. Dataset filenames,
row numbers, or train/dev labels are not evidence of disjointness. Any direct
overlap with confirm is fail-closed. All selection exposure is append-only and
human adaptive use is disclosed.

The confirm set contains at least 512 previously untouched, content-deduplicated
examples. A separate, preregistered MuSiQue/2WikiMultiHopQA or variable-tracking
evaluation supplies minimal external validity. Exact dataset identities and
SHAs are implementation-manifest fields that must be committed before data are
opened; filling those fields may not change this scientific contract.

## 12. Interpretation and failure matrix

| Outcome | Licensed interpretation |
|---|---|
| MIC > PostPost > Broadcast | learned critic helps and action-local boundary adds finite-sample value |
| MIC ~= PostPost > Broadcast | critic helps; boundary-specific empirical contribution is weak |
| MIC > Broadcast, PostPost ~= Broadcast | evidence supports boundary-specific delivery benefit |
| PostPost > MIC | pre-head/delivery error exceeds the value of removing chunk credit; MIC-core fails |
| Sibling ~= MIC | state-conditional pre-head is principled but not empirically necessary here |
| only MIC-Gated improves | reframe as reliability-gated hybrid; no broad MIC-core claim |
| gate mostly closed | learned MIC is not deliverable in this setting |
| either FullBranch-4 cell Pareto-dominates MIC under Section 7.2 | the no-rerollout amortization claim fails in its closest protocol-compatible comparison |
| MIC passes the Section 7.2 secondary gates against both FullBranch-4 cells | evidence favors amortization only for these exact terminal-reward MemAgent adaptations, not over Memory-R2 generally |
| SparseBranch Pareto-dominates MIC under Section 7.2 | strong counterexample to practical amortization value |
| critic gates pass but actor does not improve | credit is identifiable but does not translate to useful optimization |
| branching action importance fails | no meaningful writer-credit problem was established; route `NO-GO` |

No single-seed, early-budget, S128, dense-reward, critic-loss, or training-return
result licenses superiority, convergence, sufficient-training, population, or
blind-test claims.

## 13. Frozen failure and revision rules

- Any gold/future/outcome leakage, fold breach, afterstate side channel, random
  denominator, optimizer contamination, or ledger mismatch is infrastructure
  failure and blocks all scientific interpretation.
- E0 failure must be corrected before any E1 or actor data are used. Such a
  correction receives a new commit and test receipt, not a threshold change.
- E1-dev may select only one preregistered representation--head pair. E1-holdout
  failure is scientific `NO-GO`; no encoder/capacity rescue is allowed.
- Branching-oracle importance or credit-validity failure is scientific `NO-GO`.
- T5/B1 numeric or method-inactive failure permits one preregistered engineering
  correction. Performance weakness does not.
- Confirm is never used for revision, checkpoint selection, gate selection,
  hyperparameter change, or a second attempt.
- Every algorithmic or engineering variant has a distinct commit, manifest,
  run ID, output root, ledger and report. Evidence is never overwritten.

## 14. Required immutable evidence

Each block and final report binds at minimum:

- exact Git commit and clean-tree receipt;
- base/model/tokenizer/data/reward/metric SHAs;
- stable root/content identities and all split intersections;
- behavior-policy checkpoint and full actor shard hashes;
- trajectory, generation and independent seed ledgers;
- exact `X^-`/`X^+` schemas and state hashes;
- representation/projection/serializer receipts;
- outer/inner fold assignments and fit/predict root hashes;
- critic checkpoints, OOF predictions, calibration and gate receipts;
- detached credit ledger hash before and after actor PPO;
- per-role masks, sampled-token identities and action-slot receipts;
- global/per-rank loss reconstruction and role-specific gradient ledger;
- optimizer-membership proof excluding critic parameters;
- actor/vLLM weight-sync, checkpoint/resume and append-only hash chains;
- FullBranch/SparseBranch trunk, anchor, arm, counter-stream, continuation,
  denominator, and compute receipts where applicable;
- one-shot confirm manifest, predictions, independent metric rows and paired
  comparison report.

## 15. Novelty and wording boundary

The paper may claim a new action-boundary formulation and algorithmic/evidence package
only if the preregistered gates and experiments support it. It will not claim
novelty for martingales, afterstate values, centralized critics, action-value
baselines, leave-one-out estimators, cross-fitting, PPO, reliability gating, or
turn-level credit in general.

The closest-work boundary is frozen as follows:

| Primary work | Already established | Residual MIC-v2 question |
|---|---|---|
| [Memory-R2 / LoGo-GRPO](https://arxiv.org/abs/2605.21768) | Cross-rollout memory divergence makes global group comparison unfair; a global branch plus stochastic same-snapshot local rerollouts provides session-local comparison, with a length-normalized step objective | Can own-state writer advantage be amortized across diverged states without per-state local rerollout, while separating arrived chunks from materialized writes? Direct evidence uses the explicitly adapted FullBranch-4, not a claimed Memory-R2 reproduction. |
| [Input-dependent baselines](https://arxiv.org/abs/1807.02264) | Exogenous input information can define a bias-free policy-gradient baseline and reduce variance | MIC may use only already-arrived, time-safe chunk history, never future inputs, and additionally estimates the policy-controlled materialized afterstate value |
| [COMA](https://arxiv.org/abs/1705.08926) | Centralized critic and action-marginalized counterfactual baseline | Text actions cannot be enumerated cheaply; MIC estimates pre/post values across recurrent free-text states with root-OOF evidence |
| [Reactor beta-LOO](https://arxiv.org/abs/1704.04651) | Action-value LOO estimators and explicit approximation bias--variance tradeoff | MIC's afterstate quotient, recurrent boundary, and no-rerollout OOF estimator remain application-specific residuals |
| [Posterior Value Functions](https://proceedings.mlr.press/v139/nota21a.html) | Legitimate hindsight/posterior baselines with variance guarantees | MIC's post-write term is an afterstate action-value surrogate and forbids future chunks/outcomes; it is not a posterior baseline |
| [RUDDER](https://papers.neurips.cc/paper_files/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html) | Return redistribution through differences of return predictions | MIC isolates exogenous-arrival and write-action boundaries, preserves actor roles, and requires root-cross-fitted licensing |
| [HCA](https://arxiv.org/abs/1912.02503) / [CCA](https://proceedings.mlr.press/v139/mesnard21a.html) / [QCA](https://proceedings.mlr.press/v202/mesnard23a.html) | Hindsight-, counterfactual-, or luck-conditioned credit | MIC deliberately excludes future information and must justify the resulting power/cost tradeoff |
| [Turn-level LLM-agent RL](https://arxiv.org/abs/2505.11821) | Turn/hierarchy-level credit for multi-turn agents | Exact materialized-memory afterstate, recurrent divergence, fixed-slot sequence delivery, and fail-closed OOF evidence |

MIC does not claim to discover recurrent-memory rollout divergence, same-state
rerollout credit, or exogenous-input conditioning. A publishable result requires
evidence that the amortized pre/post estimator provides a useful compute--credit
tradeoff relative to the protocol-compatible FullBranch-4 and sibling
competitors. SparseBranch remains only a bounded stress test.

The intended residual is:

> a no-online-local-rerollout action-boundary estimator for a policy-controlled
> free-text memory after time-safe exogenous information arrival; an amortized
> state-conditional baseline across recurrently diverged states; and a
> fail-closed protocol tying that estimator to root-OOF prediction, branching
> validation, and the actual sequence-action actor gradient.

Relevant primary comparisons include COMA, Reactor beta-LOO, posterior value
functions, action-dependent-baseline analyses, RUDDER, HCA/CCA/QCA, turn-level
LLM-agent credit, and recurrent memory RL. Their known components must be
acknowledged rather than renamed as MIC contributions.

## 16. Release condition

This document alone authorizes no implementation and no H20 execution. The next
steps are:

1. independent adversarial scientific review of the definitions, propositions,
   nearest-work boundary, estimator, gates, and failure interpretations;
2. resolve every scientific blocker in a new reviewed contract revision;
3. freeze exact data/split/calibration/confirm manifests without opening sealed
   outcomes;
4. only then prepare an implementation plan and code changes;
5. run the complete entry-level E0 suite before any E1 or actor launch.

Until steps 1--3 are complete, status remains
`MIC_V2_PREREGISTRATION_PENDING_REVIEW`, and all GPU entry points must remain
disabled for MIC-v2.

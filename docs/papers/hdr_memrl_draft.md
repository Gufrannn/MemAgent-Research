# When Should a Memory Rewrite? Robust Recurrent Memory Learning under Evidence-Preserving Temporal Refinements

**Stage:** pre-E0/E1 framing draft, 2026-08-22. **Scientific decision:** KEEP-FRAMING / EVIDENCE-HOLD. All empirical method results below are explicitly conditional placeholders.

## Abstract (203 words)

Recurrent language-model agents process long evidence streams by repeatedly replacing a bounded natural-language memory. Their behavior can depend not only on which evidence is observed, but also on where the stream is partitioned into rewrite events. This dependence is easy to confound with context truncation, changed queries, or additional computation, and is therefore not established by conventional length-extrapolation evaluations. We formulate **evidence-preserving temporal refinement**: multiple environments expose the same terminal query and the same ordered evidence tokens under different, pre-registered rewrite boundaries. We ask whether a memory policy trained at one rewrite horizon implements a stable transition law under such refinements. We first introduce a fail-closed paired protocol that certifies evidence and query equality, excludes root leakage and accidental truncation, and estimates within-root horizon dispersion after controlling total input tokens. Only if this diagnostic finds a nontrivial failure axis do we propose HDR-MemRL, a budget-matched recurrent policy objective combining stable-root balanced scheduling with online group-robust horizon weights. A finite-group bound separates empirical worst-horizon risk, group sampling error, and refinement shift to unseen horizons; a counterexample shows that optimal nominal-horizon return alone cannot control any of them. Experiments are pre-registered on fresh Qwen2.5-7B training with fixed trajectory budgets, five update anchors, fixed-horizon and uniform multi-horizon controls, and unseen refinements. If token-controlled dispersion is absent, this formulation is marked NO-GO and the task pivots to a stronger MemAgent+RL failure axis rather than treating variable chunking as robustness evidence.

## 1. Introduction

Long documents and interaction histories exceed the context windows of otherwise capable language models. A recurrent memory agent offers a simple linear-cost alternative: read a segment, rewrite a bounded free-text memory, discard the segment, and repeat. The final answer is produced from the terminal memory. This operator is attractive precisely because it can be applied an arbitrary number of times. Yet its learned semantics may be tied to the segmentation used during training. A writer exposed to a few large segments can rely on late reader compensation, retain different facts than a writer exposed to many small segments, or progressively erase an early bridge fact. In each case the evidence and final question may be identical while only the timing of state replacement changes.

Existing evidence does not isolate this phenomenon. MemAgent demonstrates end-to-end RL for overwrite memory and strong length extrapolation, including contexts much longer than those used in training ([paper](https://arxiv.org/abs/2507.02259), [official code](https://github.com/BytedTsinghua-SIA/MemAgent)). That is important evidence for scalable memory agents, but length extrapolation jointly changes the amount of evidence, number of rewrites, and often task composition. Recent memory systems expand the action space: UMA combines a compact summary with structured CRUD memory ([paper](https://arxiv.org/abs/2602.18493)); Memory-R1 trains explicit ADD/UPDATE/DELETE/NOOP management ([paper](https://arxiv.org/abs/2508.19828)); InfMem learns writing, retrieval, and stopping decisions ([paper](https://arxiv.org/abs/2602.02704)); and MemReread learns whether to make additional passes ([paper](https://arxiv.org/abs/2605.10268)). These systems address memory capacity and control, but none of their stated objectives makes the *same* overwrite operator invariant to evidence-preserving refinements. Conversely, generic group DRO and robust RL already optimize worst-group performance; simply labeling chunk sizes as groups is not a sufficient algorithmic contribution.

The central object in this paper is therefore not input length and not generic worst-group optimization. It is **temporal refinement of a recurrent text-state transition**. Let an ordered token sequence be partitioned into segments. A refinement splits one or more segments without deleting, duplicating, or permuting tokens. The query and terminal scoring rule remain fixed. An ideal memory transition should preserve task-relevant information across a reasonable family of such refinements, even though its intermediate natural-language states need not be textually identical. Fixed-horizon training provides no such guarantee. Indeed, a policy can be optimal when a decisive fact and its disambiguator arrive in one segment, yet fail when they arrive in consecutive segments because the first is overwritten before the second is observed.

This framing changes the experimental burden. A horizon comparison is invalid unless it can prove equality of ordered evidence and terminal query, exact closure of segment boundaries, and absence of train/evaluation root overlap. It is also invalid if a smaller segment size silently triggers a global input cap, or if extra horizons multiply trajectories per update. We introduce a stable-root suite in which each root has deterministic identity from the frozen dataset, source position, and query. Every root–horizon environment carries content hashes, contiguous boundary receipts, and terminal-query hashes. A balanced Latin rotation allocates one pre-registered horizon to each root at each update while keeping the total number of roots, rollouts, and trajectories equal to the Original configuration. Evaluation uses all horizons only outside the training budget comparison.

The diagnostic precedes the method. On frozen trajectories we estimate within-root dispersion across evidence-equated horizons. A token-controlled test compares observed dispersion against root-clustered/permutation uncertainty while conditioning on total processed input tokens and checking that no generation cap was hit. If no residual dispersion remains, there is no identified target for horizon DRO: this formulation becomes **NO-GO** and triggers a new literature/identification pivot rather than GPU training. This decision rule matters: variable chunking can easily manufacture an apparent worst group through truncation or compute imbalance.

Conditional on this gate, HDR-MemRL optimizes a constrained worst-horizon objective. Stable-root scheduling ensures that a hard group cannot dominate by receiving more roots. Online exponentiated group weights emphasize high-loss horizons within a pre-registered KL ball around the uniform distribution, avoiding an unconstrained collapse to a single noisy group. Per-sample multipliers are normalized by observed group counts so the actor objective remains budget matched. The dual state is checkpointed, resumed, and recorded in an append-only ledger. Uniform multi-horizon ERM is the decisive algorithmic control: if it matches HDR, the result supports augmentation but not distributional robustness.

Our intended theory has two modest, falsifiable parts. First, a finite recurrent counterexample demonstrates non-implication: maximizing nominal-horizon return places no lower bound on a refined-horizon return, even with identical evidence and terminal task. Second, for a finite registered horizon family, a group-risk bound decomposes worst empirical risk, root-sampling uncertainty, and a refinement regularity term for unseen horizons. The bound does not claim that generic group DRO is new. It identifies exactly what additional assumption is required to transfer robustness beyond observed boundaries and motivates an unseen-horizon test rather than implying it automatically.

The paper will make four contributions only if supported by evidence:

1. **Phenomenon and identification.** An evidence-equated temporal-refinement protocol that either establishes a real within-root failure axis or terminates the route.
2. **Theory.** A non-implication counterexample and a finite-group/unseen-refinement bound with explicit assumptions.
3. **Algorithm.** Budget-matched stable-root scheduling and checkpointable constrained group-DRO for recurrent free-text memory, compared directly with uniform multi-horizon training.
4. **Empirics.** Fresh-base, five-anchor nominal/worst/unseen curves, matched compute, mechanism ledgers, and a derived variable-rewrite-depth stress set. Method improvements remain placeholders until run: `[T5/T10/T15/T20/T25 RESULTS NOT YET OBSERVED]`.

This study has a narrow boundary. It does not claim general robustness to arbitrary context-length shift, new evidence distributions, reordered documents, or adaptive retrieval. It does not change the memory schema or add intermediate rewards. Failure to find token-controlled dispersion, failure of robust training to beat uniform ERM on worst horizons, or nominal degradation beyond the pre-registered threshold are substantive negative results.

## 2. Problem formulation and propositions

Let a root be \(x=(q,e,y)\), with terminal query \(q\), ordered evidence tokens \(e=(e_1,\ldots,e_L)\), and terminal verifier target \(y\). A boundary vector \(b^h=(0=b_0<\cdots<b_h=L)\) induces segments \(c_t=e_{b_{t-1}+1:b_t}\). A shared writer policy samples bounded memories
\(m_t\sim\pi_\theta(\cdot\mid q,m_{t-1},c_t)\), and the same policy produces terminal answer \(a\sim\pi_\theta(\cdot\mid q,m_h)\). Return is \(R(x,a)\). Environments \(h,h'\) are evidence-equated iff their query hash, ordered evidence hash, evidence token count, verifier, and decoding contract are equal and both boundary vectors close exactly at \(L\).

For registered horizons \(\mathcal H\), define \(J_h(\theta)=\mathbb E_x\mathbb E[R\mid x,b^h]\), loss \(L_h=-J_h\), and

\[
\min_\theta\max_{w\in\Delta_K: D_{KL}(w\Vert w_0)\le\rho}
\sum_{h\in\mathcal H}w_h L_h(\theta).
\]

**Proposition 1 (nominal non-implication).** For any two distinct legal refinements \(h<h'\), there exists a bounded-memory recurrent environment with identical ordered evidence and terminal query, and a policy \(\pi\), such that \(J_h(\pi)=1\) and \(J_{h'}(\pi)=0\). Moreover, \(\pi\) can be uniquely optimal in the nominal environment.

*Proof.* It suffices to construct one split and pad either environment with irrelevant segments. Let the memory alphabet be \(\{\bot,0,1\}\), and let adjacent evidence tokens be a uniformly random bit \(z\) and a marker `ASK-PREVIOUS`. The terminal answer is \(z\). Under the coarse boundary, define the unique reward-one action to write \(z\) after jointly observing `(z, ASK-PREVIOUS)`; every other write maps to \(\bot\) and reward zero. Under the refined boundary, the same deterministic writer sees \(z\) without the marker and, by its nominally unconstrained action on that observation, writes \(\bot\); when the marker arrives, \(z\) is no longer in its observation or bounded memory, so its return is zero. Add a nominal penalty \(-\epsilon\) to every write other than the prescribed one to make the policy uniquely nominal-optimal. Ordered evidence and the terminal query are unchanged. Extra irrelevant segments extend the construction to arbitrary \(h<h'\). ∎

**Proposition 2 (registered and unseen refinements).** Let loss be in \([0,1]\), let \(K\) groups be fixed before sampling, and let \(n_h\) independent stable roots be evaluated in group \(h\). For any fixed policy, with probability at least \(1-\delta\),
\[
\max_h L_h(\theta)\le \max_h\widehat L_h(\theta)+
\max_h\sqrt{\log(2K/\delta)/(2n_h)}.
\]
If an unseen boundary \(b'\) satisfies \(|L_{b'}(\theta)-L_b(\theta)|\le C d(b',b)\) for every registered \(b\), then its loss is at most the right-hand side plus \(C\min_b d(b',b)\). *Proof:* apply Hoeffding's inequality to each pre-registered group, take a union bound, then apply the stated Lipschitz condition and minimize over registered boundaries. The bound is intentionally policy-fixed; a learned-policy theorem requires a held-out root suite or a uniform complexity term. Proposition 1 shows the unseen conclusion is impossible without a regularity assumption.

**E0 closure theorem (machine-checked contract).** For every root, the registered suite contains exactly one row per horizon; each boundary sequence is positive, contiguous, non-overlapping, begins at zero and ends at the full evidence count; query/evidence hashes agree; root–horizon keys are unique; train/eval roots are disjoint.

## 3. Closest-work claim matrix and novelty boundary

| Primary source | What it solves | What it does not establish for this paper | Residual/decision |
|---|---|---|---|
| MemAgent, ICLR 2026 ([paper](https://openreview.net/pdf?id=k5nIOvYGCL), [code](https://github.com/BytedTsinghua-SIA/MemAgent)) | RL-trained overwrite memory; independent-context multi-conversation; strong length extrapolation | Same-evidence boundary intervention; token-controlled within-root dispersion; worst refinement objective | Direct substrate and strongest baseline. We must not relabel its length curve as our phenomenon. |
| UMA ([paper](https://arxiv.org/abs/2602.18493)) | End-to-end RL with summary plus structured memory bank; dynamic state tracking | Invariance of one fixed overwrite operator under legal temporal refinements | Adjacent benchmark/memory design, not an algorithmic substitute. |
| Memory-R1 ([paper](https://arxiv.org/abs/2508.19828)) | RL for structured memory-management operations | Evidence-equated rewrite-depth robustness | Shows that richer actions may solve failures; include as scope boundary, not matched main baseline. |
| InfMem ([paper](https://arxiv.org/abs/2602.02704)) | RL control over retrieval, writing, stopping | Fixed-budget boundary robustness of repeated overwrites | Adaptive compute is a different estimand; stopping must not leak into HDR comparison. |
| MemReread ([paper](https://arxiv.org/abs/2605.10268)) | Learned rereading and length extrapolation | Same single-pass evidence under alternative rewrite boundaries | Rereading is an external-validity competitor with extra compute. |
| GDR-MDP ([paper](https://arxiv.org/abs/2210.12262)) | Robust RL over task groups/beliefs, hierarchical latent groups | Free-text rewrite refinements or evidence-equated identification | Establishes that group-robust RL is prior art; our novelty cannot be exponentiated weights. |
| Group DRO ([ICLR 2020 paper](https://openreview.net/pdf?id=ryxGuJrFvS)) | Worst-group optimization and group adjustment | Sequential state rewrite, paired roots, unseen refinement | Algorithmic ancestor; uniform ERM and adjusted DRO are mandatory controls. |
| SPPO ([paper](https://arxiv.org/abs/2604.08865)) | Sequence-level PPO with scalar value for long reasoning | Robustness across environment refinements | Optimization-unit neighbor only; HDR retains accepted Original GRPO geometry to isolate the estimand. |
| Trajectory-level robust regularization analysis ([paper](https://arxiv.org/abs/2407.13399)) | Trajectory-level divergences/sample-complexity perspective | Environment-group shift induced by rewrite boundaries | Warns against attributing generic sequence weighting novelty to HDR. |
| Doob/martingale credit and actor-critic literature; PPO ([paper](https://arxiv.org/abs/1707.06347)) | Assigns return to actions/states and stabilizes policy updates | Does not identify boundary-induced environment groups; HDR deliberately keeps terminal credit fixed | Credit assignment is a competing explanation and must be measured, but is not silently changed. |
| Safe policy improvement / TRPO ([paper](https://arxiv.org/abs/1502.05477)) | Bounds degradation under policy-distribution shift using trust regions | Does not control an externally intervened rewrite-boundary distribution at a fixed policy | Policy KL and boundary robustness are distinct; KL-matched training is an ablation. |
| Information Bottleneck ([paper](https://arxiv.org/abs/physics/0004057)) and variational information bottleneck ([paper](https://arxiv.org/abs/1612.00410)) | Trades predictive sufficiency against representation rate | Does not specify repeated free-text rewrite invariance; capacity changes can mimic horizon failures | Memory token/rate matching is required to rule out a capacity-only account. |
| Robust MDP survey/formulations and GDR-MDP ([paper](https://arxiv.org/abs/2210.12262)) | Handles transition/reward or task-distribution uncertainty | Groups are generally given/latent, not generated by an evidence-equated intervention with paired roots | Our identification protocol, not minimax algebra, is the residual. |
| AMAGO ([paper](https://arxiv.org/abs/2310.09971)) | Long-sequence in-context RL and memory/meta-learning | Learned hidden-state adaptation rather than repeated destructive natural-language state replacement | Recurrent-RL neighbor; motivates testing whether the phenomenon is specific to overwrite memory. |

**Novelty boundary.** A publishable residual exists only if (i) evidence-preserving refinement reveals nontrivial failures beyond token/truncation effects, (ii) the paired protocol is materially stronger than ordinary variable-length testing, and (iii) constrained DRO improves worst/unseen refinement over *uniform multi-horizon ERM* at matched budget. If only (i) holds, the output is an analysis/benchmark paper or should merge with a broader memory-robustness project. If only uniform augmentation helps, HDR as an RL-method paper is NO-GO.

## 4. Experimental plan

### Frozen gates and data tables

| Stage/table | Rows | Columns / estimands | Required decision |
|---|---|---|---|
| E0 scheduler closure | roots × horizons | evidence/query SHA, token count, bounds, duplicates, split overlap | zero violations |
| E1 frozen dispersion | paired frozen roots | per-horizon EM/F1/format, input tokens, cap-hit, within-root range, permutation CI, token-controlled coefficient | dispersion above registered noise and nonzero after token control |
| T5 health | Original-T5 vs fixed vs uniform vs HDR | nominal/worst/unseen F1, format, group counts, dual entropy, writer tokens, trajectories | all infrastructure/mechanism gates; nominal no worse than 2pp; worst +2pp |
| Main anchors | T5/10/15/20/25 | nominal, worst registered, unseen, mean; paired root CIs | T25 success contract; no post-hoc horizon removal |
| External validity | derived Hotpot + variable-tracking | evidence-preserving refinements only | directionally consistent effect or explicit boundary |

### Main baselines

1. Accepted fixed-horizon Original, imported via file-level SHA evidence and independently recomputed metrics; never retrained.
2. Fixed-horizon fresh control where needed for method-run infrastructure only, clearly separated from the certified curve.
3. Uniform multi-horizon ERM with the exact stable-root scheduler and trajectory budget.
4. HDR-MemRL with constrained online weights.
5. Length-matched random boundary perturbation, token-count groups, and curriculum without DRO.

### Ablations tied to the insight

- Remove stable-root balancing (tests confounding by root frequency).
- Keep balancing but freeze uniform weights (isolates DRO from augmentation).
- Group by processed token count rather than rewrite depth (tests whether horizon is only length).
- Remove the KL ball / vary preregistered dual timescale once (tests collapse control).
- Evaluate seen only versus held-out intermediate horizons (tests, but cannot prove, refinement regularity).
- Shuffle horizon labels as a negative control.

### Failure criteria

- **NO-GO_E0:** any unexplained evidence/query mismatch, boundary gap/overlap, truncation, duplicate root–horizon pair, or split leakage.
- **NO-GO_E1:** dispersion is within registered permutation/noise tolerance, disappears under total-token/cap controls, or depends on post-hoc horizon selection.
- **NO-GO_FRAMING:** reviewer judges the residual to be generic variable chunking plus known group DRO.
- **NO-GO_T5:** mechanism invalid, nominal F1 drops more than 2pp, or worst F1 fails the registered +2pp gate; one revision only for the pre-registered dual-timescale/curriculum failure class.
- **MERGE recommendation:** E1 is strong but uniform ERM matches HDR, leaving a valuable benchmark finding without a standalone RL algorithm contribution.

## 5. Current scientific status

**KEEP-FRAMING / EVIDENCE-HOLD.** Temporal refinement is a precise failure axis not stated by the closest sources reviewed above; the paired intervention, nominal non-implication, and explicit unseen-refinement assumption form a coherent paper story. The optimizer alone remains known machinery. The route becomes an empirical GO only through a strong identified phenomenon, exact paired protocol, and separation from uniform multi-horizon ERM. E0/E1 and an independent framing review remain mandatory before any Method-T5 command is emitted; E1 failure triggers a further pivot rather than a false positive.

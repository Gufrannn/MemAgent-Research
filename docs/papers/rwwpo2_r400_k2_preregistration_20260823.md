# RWWPO-2 preregistration — controlled off-behavior geometry in recurrent state writing

**Status:** frozen scientific design; pre-R50 release tests, data-boundary audit,
base-protocol audit, numeric oracle, and independent numeric audit passed on H20
at commit `2d0369040dc5aacd9e07369bc90466c305e38919`. The first B/seed-2026 R50
attempt then failed closed before its first optimizer commit with
`RWWPO2_BEHAVIOR_PARAMETER_GRADIENT_SKETCH_MISMATCH`. It produced no valid R50
round. Inspection found that the live actor had duplicated the registered
numeric-oracle projection with a different, unbounded implementation and that
the oracle lacked the live seven-section long-context streaming replay. The
implementation correction was independently reviewed and released at commit
`5e9e60d1547ac82c76e762090f3d7fe6518f0692`; its first authenticated H20
release-test attempt stopped before GPU work because two direct-file tests used
an unrelated Python without Torch. Commit `d69a238743a394f6dde2f6fc98b26d5fd33d87db`
fixed that interpreter isolation and passed 126/126 authenticated tests plus
the CPU evidence gates. Its numeric oracle then failed closed without a PASS
receipt: the new 8191-token replay OOMed because the oracle used one monolithic
FSDP wrapper and a full FP32 log-softmax rather than the live actor's layered
auto-wrap, mixed precision, activation checkpointing, and selective log-prob
kernel. The correction retains 8191 tokens and seven backwards but now binds
those live execution semantics. Commit
`a512741e9d559d8a43b58695200cc512a0cb916e` then passed 126/126 authenticated
tests, the CPU evidence gates, and the two-H20 numeric oracle. Its first
B/seed-2026 R50 attempt completed round 1, then completed round-2 inner-1 with
an alpha-zero rollback and failed at the round-2 inner-2 behavior precondition.
The failed attempt contains three transaction receipts per rank, no round-10
recovery checkpoint, and no valid R50 endpoint. Read-only reconstruction showed
that the old alpha-zero receipt reused a cached pre-rollback tensor instead of
performing a committed-state forward, while the transaction snapshot omitted
forward-mutable model buffers. Therefore that receipt cannot certify complete
rollback. The current correction snapshots/restores all named model buffers,
performs an independent post-commit/post-rollback forward, and makes any
closure failure an append-only attempt-level NO_GO. Commit `b540521...`
contained that correction, but its first authenticated H20 suite stopped before
GPU work at 139/140 PASS because one synthetic test fixture attached the fresh
committed-forward prefix rows to a distinct cached trial tensor. The production
auditor correctly rejected the inconsistent fixture. The fixture is now split
into separate trial and committed certificates; the correction must pass a new
exact-commit authenticated suite and numeric oracle. Commit
`eab35b9d390a30f9c8b2e4696dfaf94988e6ffe4` passed those pre-R50 gates and its
fresh B/seed-2026 attempt completed six full rounds before round-7 inner-2 failed
the post-commit forward closure. The alpha-zero fresh forward differed from its
behavior reference by `1.0065193176269531` with frozen `tau_logprob=1e-6`.
Inspection found that behavior materialization recorded one RNG snapshot per
microbatch, while trial and fresh replay restored only one post-gradient RNG
for the whole sequence. The correction makes every diagnostic replay consume
the ordered behavior microbatch RNG schedule and binds its ordered digest vector
and aggregate in each receipt. It must pass a new exact-commit authenticated
suite and numeric oracle before another fresh-base R50 attempt. Commit
`7d7054a7a7269052f9194f59906a94c664523cd4` implemented and tested that replay
binding, but the subsequent B/seed-2026 attempt falsified replay RNG as the
complete explanation: all seven behavior-forward RNG digests were identical,
two rounds completed, and round-3 inner-2 again failed the alpha-zero fresh
closure (`max_abs=0.8136634826660156`, frozen `tau_logprob=1e-6`). That root is
also attempt-level `NO_GO` and has no valid R50 endpoint. Code audit then found
an execution-contract mismatch that the prior oracle did not test: the live
actor loads FP32 optimizer parameters and executes BF16 FSDP mixed-precision
forwards, while the prior oracle loaded the model directly in BF16; moreover,
the transaction interpolated visible rank-local FlatParameter shards outside
an FSDP writeback context. This is a leading, falsifiable FSDP derived-state
hypothesis, not yet a proven root cause. The next pre-R50 oracle therefore must
load FP32, run an 8191-token backward, contrast the legacy raw-shard path with a
unit-wise all-gather plus public `summon_full_params(writeback=True)` commit,
verify repeated candidate and alpha-zero restore forwards within the
independently calibrated tolerance, and enforce a 120-second maximum for every
full writeback. R50 remains locked until that exact-commit oracle and its
independent audit pass. No consumed
failure root or old numeric receipt may be reused. The
previous K1 hard-rollback run remains diagnostic-only and is not a performance
result for the method below.

Commit `dc14d0aab388aea0bc191ba186eded2d1439db5c` contained the reviewed
FSDP-safe writeback implementation, but its first authenticated H20 release-test
root stopped before CPU or GPU work at 154/156 PASS. The failures were confined
to test evidence: a brittle static slice selected the earlier writeback branch
rather than the actual trial replay branch, and the synthetic transaction-closure
fixture did not populate the phase fields required by the independent auditor.
The production algorithm and audit thresholds were unchanged. The consumed root
is `NO_GO`; a new exact commit must pass a fresh authenticated suite and numeric
oracle before R50 is unlocked.

Commit `4a6a72ef51aa9e8bba2b9c2efe22dc4c98b54dfe` later passed its frozen
pre-R50 gates and launched B/D/E seed-2026 assignments. The B and D attempts
reached the round-30 checkpoint boundary, where recovery-root pruning ran for
the first time and raised `NameError: append_gate_a_record is not defined`.
Rounds 10 and 20 did not exercise this branch because no recovery root was yet
old enough to prune. Both full attempts are `NO_GO`; an existing round-30 root
is only a candidate lineage parent. A read-only lineage audit alone never
authorizes a different training commit. Cross-commit resume now has a separate
fail-closed compatibility producer that requires unchanged algorithm and
resolved controller/numeric semantics, identical authenticated runtime
environments, and consumer release-test PASS evidence. No H20 compatibility
receipt or resumed endpoint currently exists, so these checkpoints remain
unauthorized until those gates actually PASS.
The repair
does not merely add the missing local import: pruning is now a two-phase
append-only transaction (`intent -> delete -> complete`) bound to the recovery
checkpoint inventory and the preserved scientific-anchor inventory. Missing,
duplicate, reordered, forged, or incomplete prune evidence is fail-closed.
Any process that already loaded the old producer remains unaffected by the
source repair and must not cross the round-30 pruning boundary.

Commit `39373a4cb133a524513cbfd9fff3be9388b16ba6` subsequently passed its
authenticated release/CPU/cross-commit gates. B round-30 lineage passed, while
D round-30 lineage failed post-step tensor reconstruction and remains
unauthorized for resume. The post-hoc B/D/E T20 fixed-S128 generations and
per-cell audits completed, but the descriptive comparison failed before any E
resume because it compared the three checkpoint-specific resolved-manifest
file hashes for equality. Those files must differ by construction. The repaired
diagnostic contract preserves each file hash as cell-specific evidence and
uses the common, validated `eval_manifest_hash` as the cross-cell example/order
identity. This is an audit correction only; S128 remains adaptive development
evidence and cannot become confirmatory performance evidence.

## 1. Scientific question

A recurrent memory writer emits free text that is materialized as the next
observation. Tokenwise PPO, a per-write joint objective, and a whole-writer-path
objective can nevertheless have exactly the same gradient at the behavior
policy. The experiment asks whether their distinct *off-behavior* geometry is
useful once the training loop deliberately, minimally, and audibly leaves that
point, and whether cumulative writer-prefix feasibility controls that movement.

The central claim is conditional:

> Recurrent whole-path objectives differ from tokenwise objectives only after
> optimization leaves the behavior policy. We test whether, in recurrent state
> writing, prefix-distribution feasibility can control this off-behavior regime
> and whether cross-turn geometry adds value beyond per-write geometry and the
> controller itself.

R400 is a preregistered medium-budget study, not a convergence claim. R50 is a
mechanism gate that never reads S128 or any confirmatory performance set.

## 2. Single-pass degeneracy proposition

**Proposition (single-pass degeneracy).** Consider C, E, and B below as
deterministic transition kernels. Suppose each behavior batch receives exactly
one actor transaction and the cells begin from identical complete algorithmic
state: model parameters and buffers, ordered parameter groups, full optimizer
and scaler state, scheduler state, clipping state, distributed-reduction state,
RNG streams, and batch. Suppose their exact full parameter-gradient tensors,
old log probabilities, advantages, active-token denominator, answer loss, and
KL/entropy additions agree. Finally, suppose the controller is an
objective-blind deterministic function only of the complete candidate state and
fixed batch, including deterministic trial forwards, and reads neither objective
value, variant label, attempt identifier, nor variant-specific randomness. Then
C, E, and B produce the same proposal, trial decision, commit, complete next
state, and by induction the same training trajectory.

The result is an algorithmic degeneracy statement, not a statement that the
objectives are globally identical. It is falsified if any assumption above is
violated. In particular, objective-dependent controller inputs, unequal
normalization, unequal dropout streams, nonidentical scheduler state, or
different optimizer moments invalidate the induction. Accepted-only scheduling
does not invalidate the theorem when its initial state and deterministic
decisions remain identical; it is undesirable in C-D because it confounds the
controller assignment with a different LR trajectory after decisions diverge.

Consequently, the old K1 results identify only the prefix-controller dynamics.
They cannot be interpreted as evidence for the whole-path surrogate.

## 3. K2 intervention and frozen cells

Each rollout round reuses its behavior batch for exactly two full-batch actor
transactions. In this implementation, **one PPO epoch is exactly one inner
transaction**, the full recurrent batch forms exactly one optimizer minibatch,
and exactly one `optimizer.step()` is called per epoch. Thus `ppo_epochs=2`
means K2 and at most 800 proposals at R400; any nested or additional step fails
the transaction-count audit. The manifest independently freezes
`optimizer_minibatches_per_inner_transaction=1` and
`optimizer_steps_per_inner_transaction=1`; neither field is inferred from the
framework name “epoch.” Old log probability, scalar GRPO advantage, masks, identities,
active denominator, answer loss, and reference-policy KL are frozen across both
transactions. They are never recomputed or renormalized between inner updates.

| Cell | Writer objective | Prefix controller | Budget |
|---|---|---|---|
| D | Original tokenwise PPO | none; telemetry only | R50 mechanism on three seeds; R400 confirm on eight paired seeds |
| C | Original tokenwise PPO | cumulative-prefix feasible backtracking | R50 mechanism on three seeds; R400 explanatory on three seeds |
| E | per-write joint ratio | same controller | R50 mechanism on three seeds; R400 confirm on eight paired seeds |
| B | whole-writer-path ratio | same controller | R50 mechanism on three seeds; R400 confirm on eight paired seeds |
| A | whole-writer-path ratio | alpha=1 hard rollback | R50-K2 short reference |

For trajectory `i`, writer turn `t`, and the active writer tokens of that turn,

```text
ell_i,t = sum_k (log pi_theta - log pi_behavior)
```

On FSDP rank `h`, E uses `-sum_i,t exp(ell_i,t) A_i / D_h`; B uses
`-sum_i exp(sum_t ell_i,t) A_i / D_h`, where `D_h` is that rank's active-token
denominator. C and D use the accepted Original tokenwise clipped-PPO reduction:
the distributed optimizer averages the two rank-local losses. This preserves the
actual Original implementation even when the two ranks have unequal token
counts; it is not relabeled as a globally token-weighted denominator. All cells
keep final-answer tokens under Original tokenwise clipped PPO, and each rank's
`D_h` is frozen across the two inner transactions and recorded in the ledger.

The direct within-round source of objective separation is inner transaction 2,
after an actual nonzero inner-1 commit. From round 2 onward independently
trained cells may differ already at their live inner-1 proposal because earlier
inner-2 updates changed their parameters, optimizer moments, and future
rollouts. That inherited divergence is legitimate and must not be mistaken for
a violation of behavior-point equality.

## 4. Proposal clock, learning rate, and deterministic replay

The immutable logical proposal clock is

```text
p = 2 * (round_id - 1) + inner_id,  inner_id in {1, 2}
eta = S(p)
```

`S` is the stateless schedule

```text
S(p) = base_lr * min(1, p / warmup_proposals)
base_lr = 1e-6, warmup_proposals = 2, 1 <= p <= 800.
```

Therefore p=1 uses `5e-7` and p>=2 uses `1e-6`. Before every proposal the runtime
writes `S(p)` directly to every optimizer param group. The stateful framework
scheduler is not advanced and is not part of a reject transaction. Immediately
before the logical reseed, the runtime freezes Python, NumPy, Torch CPU, and all
CUDA RNG states plus every persistent and non-persistent named model buffer. A
reject restores that complete transaction-entry snapshot, along with model
parameters and Adam moments, but does not change the logical `p`. Behavior
materialization records the complete RNG state immediately before every ordered
microbatch forward. Every trial and committed-state certificate starts from the
frozen entry buffer state and restores the corresponding saved behavior RNG
before each microbatch. A replay preserves and restores the algorithmic terminal
RNG and entry model buffers in `finally` paths, so diagnostic forwards cannot
advance training randomness or retain forward-mutated cache state. Each receipt
binds the ordered per-microbatch RNG digest vector,
its aggregate digest, and its exact count. After either commit or rollback,
RWWPO-2 does not mutate a visible FSDP shard and assume the next mixed-precision
forward will discover it. For each FSDP unit, every rank constructs the intended
FP32 local shard, all-gathers the intended shards, overwrites the independently
constructed full target inside public
`FSDP.summon_full_params(recurse=False, writeback=True)`, and verifies the
resharded local target exactly. This unit-wise primitive is frozen as
`fsdp_unitwise_allgather_summon_writeback_v1`; legacy RWWPO retains its old raw
shard behavior. Before R50, the two-rank numeric oracle must prove this primitive
under the live FP32-load/BF16-forward contract at 8191 tokens. The probe executes
the frozen AdamW `(lr=1e-6, betas=(0.9,0.999), weight_decay=0.01)` step after
global-norm clipping at 1.0 before testing candidate/restore closure; a backward
without that optimizer transition is insufficient evidence. Every
rank must report a positive finite gradient norm, exactly one optimizer step,
and optimizer state covering every managed FSDP unit, while every recorded
probe phase must have all FSDP units in `TrainingState.IDLE`. The independent
auditor binds these phase inventories and their cross-rank sufficient
statistics; source-level presence of an optimizer call is not evidence. Every
live writeback is timed with a cross-rank maximum and must finish within 120
seconds. The complete ordered writeback-plus-trial-forward search is separately
capped at 600 seconds per transaction; both limits are manifest-, resolved-
contract-, receipt-, and formal-audit-bound. The final tested feasible trial is
already the committed parameter state and is not redundantly written a second
time; the oracle nevertheless tests repeat-write idempotence. Alpha-zero still
performs an explicit complete-state writeback/restore. After either commit or rollback,
the runtime performs a separate forward from the committed parameter/buffer
state; only that forward may populate the post-prefix certificate. Its active
log-probability difference from the corresponding trial/behavior reference
must not exceed the pre-R50 numeric-oracle `tau_logprob`. For controller cells
A/B/C/E the reconstructed prefix distribution must also be feasible; cell D is
the preregistered no-controller telemetry baseline and does not acquire a
feasibility constraint through this evidence check. Named buffers are treated as
non-optimizable implementation/cache state: every trial begins from the entry
snapshot and the terminal buffer digest must equal the entry digest for both
accepted and rejected transactions. Otherwise an append-only failure receipt
is written and the attempt is ineligible for PASS. The ledger separately binds
entry, post-reseed, post-gradient/pre-trial, ordered behavior-forward, and
terminal RNG digests, plus entry
and terminal buffer digests. An accepted transaction commits its deterministic
post-trial RNG state; the next transaction is independently reset by its logical
seed. The accepted optimizer clock `u` increments only after a
nonzero commit and is diagnostic, never an input to `S`.

For a feasible partial step `0 < alpha < 1`, model parameters commit the
interpolated displacement `theta_b + alpha(theta_full-theta_b)`, while Adam
moments commit the full-proposal gradient update. This is a frozen part of the
controller package and is identical in C/E/B; moment interpolation is not
performed or selected after observing R50.

Attempt IDs only name evidence roots. Rollout and actor randomness use immutable
logical coordinates:

```text
seed = H(experiment_seed, round_id, inner_id, rank, stream)
```

The attempt ID, process ID, wall time, GPU index, and resume count are excluded.
If round 27 fails and a new attempt replays from the round-20 recovery
checkpoint, logical rounds 21 onward receive exactly their original proposal
clocks and seeds. The new attempt preserves the failed root but cannot alter the
algorithmic stream.

## 5. Local shadow-gradient identification

Every shadow record names its `host_variant`. At a host behavior point
`(theta_b^H, B^H)`, C/E/B loss coefficients and parameter-gradient sketches are
computed on the same parameters and batch and must agree within calibrated
numeric tolerance. These are full actual-loss shadows: the identical
reference-policy KL addition is included in all three gradients, while entropy
is frozen to zero. At the host off-behavior point `(theta_1^H, B^H)`, the same
three shadows quantify higher-order separation. No cross-run live gradient is
presented as a same-point comparison.

Cross-run R1 equality is not inferred from matching vLLM seeds. Exact R1
proposal equality is tested in a common-host replay oracle using one immutable
behavior batch and complete state. Independent live runs require matching
protocol/identity receipts and calibrated numerical health, not byte equality
across separate BF16/FSDP/vLLM executions. Later live proposals are compared
only distributionally.

The released R50 contract does not add a separate frozen-D displacement probe.
Its local evidence is the common-host C/E/B shadow computed at both inner
transactions of every R50 round in every host cell. Adding a separate probe
later would be a new, independently versioned diagnostic and cannot repair or
reinterpret the frozen R50 gate.

## 6. Off-behavior exposure and geometry activation

After inner 1 define relative parameter displacement

```text
d_theta = ||theta_1 - theta_b||_2 / (||theta_b||_2 + epsilon)
```

and trajectory-balanced writer-token movement

```text
d_logp = sqrt(mean_i mean_{k in writer(i)} (log pi_theta1 - log pi_behavior)^2).
```

A round is exposed only when both exceed frozen tolerances. `alpha > 0` alone is
inadmissible because BF16 quantization can erase a nominal displacement.
`tau_theta`, `tau_logp`, and the gradient-separation threshold `tau_g` are fixed
from no-op, save/load, all-reduce, and BF16/FSDP numeric-oracle tests before R50;
they may not be selected from R50 outcomes. The registered parameter-gradient
projection is one shared chunk-bounded implementation used by both oracle and
live actor. Its oracle additionally repeats a synthetic, label-free 8191-token,
seven-microbatch streaming backward—the maximum frozen R50 actor section count—
and the independent auditor reconstructs the unchanged 16-times-noise threshold
rule. The same oracle also tests the live FP32 optimizer-shard to BF16 forward
transition, an 8191-token no-update backward, candidate recommit, alpha-zero
restore, a second fresh forward, exact behavior-reference integrity, and the
frozen writeback-time ceiling. None of these closure outcomes enters the numeric
threshold calculation. Runtime mismatch diagnostics report only aggregate coefficient and
projection statistics, never examples, tokens, rewards, or outcomes.

An eligible round has a complete two-rank inner-1 commit/reject record, at least
one writer token for every included trajectory, and a complete inner-2 shadow
record. Every eligible round is in the ExposureRate denominator; rejected or
quantized inner-1 updates count as not exposed. Exposure and conditional
activation are reported separately for each host cell and are descriptive
diagnostics, never a causal cross-cell contrast. The R50 report includes

```text
ExposureRate = exposed rounds / eligible rounds
GeometryActivation = P(shadow separation | exposed)
```

with shadow separation quantified by fixed coefficient-space differences,
cosines, and deterministic local-FSDP gradient sketches. If exposure is too low
to identify the geometry, the conclusion is insufficient activation under the
frozen R400-K2 contract—not failure of B as an abstract objective.

If inner 1 rejects, inner 2 still executes from `theta_b` with the same old log
probability and advantage but the next proposal-clock learning rate. C/E/B are
then still locally degenerate, and the ledger must record that fact.

## 7. Controller and estimands

The descending grid is `[1, 1/2, 1/4, 1/8, 1/16, 1/32]`. Every tested point is
realized by a fresh forward. The controller commits the first feasible point in
that descending order, i.e. the largest *tested* feasible point. It makes no
monotonicity or globally maximal-step claim.

Feasibility requires each cumulative writer-prefix turn to satisfy frozen
normalized ESS and absolute log-ratio-cap conditions. The controller must not
read objective value, variant label, S128, reward aggregates, or future/gold
information.

Preregistered longitudinal intention-to-treat (ITT) contrasts are:

- `C-D`: effect of assigning the prefix-feasibility controller package;
- `E-C`: effect of assigning the per-write joint objective under that controller;
- `B-E`: effect of assigning the whole-path rather than per-write objective;
- `B-A`: feasible backtracking versus hard-rollback training dynamics.

The longitudinal contrasts include all downstream rollout and optimizer-state
mediation. They do not by themselves isolate a pure geometry mediator. Only the
same-host shadow and shared-trial evidence identifies the local geometry. `B-A`
is not a pure long-run single-step causal effect once rollouts diverge.
Its clean local counterfactual is the shared trial ledger: alpha=1 infeasible
but a smaller alpha feasible means B salvages a proposal that A would reject.

## 8. R50 mechanism gate and uncertainty unit

R50 contains 50 behavior batches and 100 transactions per run. Cell A uses the
same three explicitly frozen mechanism seeds as the initial matrix. Inner
transactions share a batch; they are not independent observations. ESS
stability and geometry uncertainty are summarized by rollout-round/root
clusters. No observation-level leave-one-out inference is permitted.

R50 is continued only if all preregistered, performance-free gates pass:

1. finite losses, gradients, proposals, and trial forwards;
2. exact behavior-point C/E/B shadow equality;
3. nontrivial off-behavior exposure and measurable conditional geometry;
4. trajectory-concentration ESS and prompt-root ESS pass the frozen
   leave-one-prompt-root-out stability audit;
5. backtracking has nonzero aperture and is not effectively frozen;
6. two-rank decisions, weights, optimizer state, proposal clock, seeds, ledger,
   checkpoint, and resume replay all close.

Numeric exposure/separation tolerances are outputs of the pre-R50 oracle and
must be hash-bound in the resolved experiment contract before launch. The
performance-free gates are already frozen at 40 or more eligible rounds, at
least 10 exposed rounds, ExposureRate at least 0.20, at least five activated
exposed rounds, and GeometryActivation at least 0.50 per host; the maximum
leave-one-root-out feasibility flip fraction is 0.25. If the
approximately eight trajectories but only four independent prompt roots per turn
make ESS unstable, the run is NO-GO for the chosen statistical unit. Trajectory
ESS is never presented as eight independent roots. `q_min` and the log-ratio cap are never relaxed
to rescue it.

A turn is `root_loo_supported` only when at least three independent prompt
roots have writer tokens, so deleting one root leaves at least two. Sparse late
turns remain subject to full-data trajectory/root ESS and log-ratio-cap tests,
but do not enter the LOO flip-rate denominator. Every audited rollout round must
contain at least one supported turn, and supported/unsupported turn counts are
both reported.

## 9. R400 budget, checkpoints, and lineage

- 400 rollout rounds, four prompt groups per round, two independent trajectories
  per prompt group, at most 3,200 trajectories per cell/seed;
- two full-batch actor proposals per round, at most 800 proposals;
- one optimizer minibatch per inner transaction;
- no learned critic, prior, reward model, or auxiliary fit;
- full recovery checkpoints every 10 rounds, retaining the newest two;
- immutable actor-only scientific anchors at 5/10/15/20/25/50/100/200/300/400;
- every attempt has a new immutable root. Canonical lineage is an authenticated
  DAG: an audited recovery checkpoint and all ledger prefixes anchored by it may
  become a parent of a new attempt; the failed suffix after that checkpoint is
  permanently excluded. A checkpoint without prefix/tail and parent-attempt
  binding can never be resumed.

Non-multiple-of-10 full checkpoint materializations used to create scientific
anchors may be deleted only after their actor shards are hard-linked/copied into
an immutable anchor inventory and audited. Recovery deletion is never allowed
to delete a scientific actor anchor or evidence ledger. Every recovery-root
prune must first append an authenticated intent bound to both the checkpoint and
scientific-anchor inventories, then delete the exact resolved root, and finally
append the immediately following completion record proving the root is absent.
An intent without completion, a completion without its exact intent digest, or
any unexpected prune makes the full endpoint ineligible for PASS.

The actual-loss evidence is stored as immutable tensor shards (round x inner x
rank) plus small append-only JSONL receipts and a hash chain. Each tensor shard
contains behavior/current/reference/proposed/committed log probabilities,
advantages, role masks, and stable identities. The independent auditor rebuilds
the selected surrogate, low-variance reference KL, total policy loss, and active
log-probability gradient norm from those tensors; self-reported scalar loss is
not accepted. Large arrays and masks are forbidden in JSONL. Checkpoint receipts
anchor the tensor-shard inventory and the JSONL prefix/tail.

## 10. Performance protocol and confirmatory hypotheses

S128 has already been inspected and used during the K1 controller pivot. It is
an adaptive development benchmark and is forbidden during R50 and during all
R400 training runs. It cannot select hyperparameters, capacity, stopping,
seed, or checkpoint.

After the performance-free R50 mechanism gate passes and before any R400
training starts, an at-least-512-example confirmation set must be materialized,
sealed, and shown content/root-disjoint from actor training, S128, Capture32, and
all prior inspected inventories using canonical normalized question/context/answer
content hashes and question/context root hashes. R50 may not read the candidate
confirmation rows or their outcome labels. The generation producer,
greedy decoding parameters, tokenizer/model revisions, prompt rendering,
maximum lengths, missingness rule, stable-root join, evaluator code SHA, and one
fixed generation seed are included in the seal. It is opened once only after all
required seeds, mechanism analyses, code, and checkpoint rules are frozen.
The protocol binds `data.max_prompt_length=8192` before dataset construction and
the production-effective `MemoryDataset` prompt-filter length of 40,000 after
its deterministic in-place configuration update. The producer execution
summary records both values; protocol equality is rejected if either changes.
The same canonical protocol also binds vLLM dtype/load format, eager/cache
mode, sampling flags, memory utilization, batching limits, and validation
decoding; the repository-local reward source is represented by its relative
path and byte SHA rather than an absolute checkout path.

Metrics are frozen as primary macro token-F1; key secondary strict normalized
EM; macro precision/recall decomposition; format safety; and historical SubEM
diagnostic. Dense training reward is never performance.

The co-primary confirmatory hypotheses are:

1. overall utility: `B-D` on confirmatory macro token-F1;
2. whole-path-package assignment: `B-E` on confirmatory macro token-F1.

The initial D/C/E/B matrix uses three seeds for mechanism estimation. It cannot
support confirmatory algorithm-level significance. Before the one-time
confirmation, B/D/E are run for the eight independently frozen paired training
seeds `2026..2033`, without selecting additional seeds from the three-seed
outcomes. The confirmatory statistic is the seed-level paired difference in
root-macro token-F1. For each contrast, its frozen practical margin is
subtracted from each paired seed difference (`0.02` for B-D; `0.01` for B-E).
The one-sided null is that the mean paired difference is no larger than that
margin. The two margin-centered statistics use exhaustive paired sign-flip
enumeration over the eight seed differences and Holm family-wise correction at
0.05. Its finite-seed inferential validity assumes independent training seeds
and sign symmetry of the margin-centered differences at the boundary null; it
is not presented as a distribution-free randomized-assignment test. Exact zero centered seed differences remain
zero in every sign assignment; raw exact p-values are ordered ascending for the
Holm step-down rule. Roots quantify within-seed evaluation
uncertainty but are not substituted for independent algorithm assignments.
Both co-primary hypotheses must pass for the full RWWPO claim. `C-D` and `E-C`
are explanatory contrasts and cannot be promoted after viewing results.

## 11. Explicit failure boundaries

- K1 results may not support a whole-path-surrogate claim.
- Low exposure may not be relabeled objective failure or rescued by increasing K.
- A controller package result may not be presented as pure objective geometry.
- S128 may not be relabeled blind, held-out, or confirmatory.
- R400 may not be called converged or sufficiently trained.
- No result may be selected by SubEM after primary token-F1 fails.
- Missing tensor shards, rank coverage, logical seeds, host identity, or lineage
  evidence invalidate the affected claim.

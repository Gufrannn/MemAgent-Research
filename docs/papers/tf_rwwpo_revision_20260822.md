# Prefix-Divergence Trust Regions for State-Writing Policies

## Status and frozen negative evidence

This revision is an unrun, falsifiable algorithm proposal.  The prior
whole-prefix + binary hard-rollback run (`b485f2e`, run
`rwwpo_seed2026_metricfix1_b485f2e`) remains diagnostic-only and formally
NO-GO.  It committed seven nonzero updates out of 25 and rejected all five
anchor proposals.  Its fixed-S128 F1 differences versus certified Original at
T5/10/15/20/25 were respectively +0.04796, +0.00585, -0.00343, -0.05119, and
-0.05063.  These observations motivate the controller revision but do not
identify a causal mechanism or establish that backtracking works.

## Reframed claim

Let `Z_1:t` be the writer path and `S_t = F_t(Z_1:t)` the materialized memory
state.  Under absolute continuity and a policy-independent environment kernel,
the data-processing inequality implies that an f-divergence between writer
path distributions upper-bounds the corresponding divergence between their
push-forward memory-state distributions.  Path divergence is not equal to
state divergence.  Finite-minibatch ESS is a certificate statistic, not a
population chi-square divergence.

The final answer is not materialized into the current episode state.  It keeps
Original tokenwise clipped PPO.  Because writer and answer share parameters,
every proposed update is nevertheless certified by a fresh forward over all
writer prefixes.

## Transactional feasible-step controller

For each optimizer update, TF-RWWPO computes the actual-loss gradient and a
full Adam proposal.  It then tests the frozen descending grid
`[1, 1/2, 1/4, 1/8, 1/16, 1/32]`.  Each candidate is realized as
`theta_old + alpha (theta_full - theta_old)` and evaluated by a fresh forward.
The controller commits the largest tested feasible grid point satisfying both
prefix ESS >= 0.5 and the frozen absolute cumulative log-ratio cap.  It never
claims a globally maximal feasible step.  A nonzero commit retains the full
Adam proposal's moments while damping only that update's parameter
displacement; this is not a permanent global learning-rate change.  If no
nonzero candidate is feasible, model, optimizer, scheduler, and RNG states are
restored.  BF16 training has no AMP scaler; the ledger records this explicitly.

## Corrected propositions

1. Tokenwise clipped PPO is generally not a function of a sampled whole-write
   likelihood ratio.  A single sampled path does not specify the full joint
   policy distribution.
2. At the behavior point, the unclipped whole-prefix writer term has the same
   per-token coefficient as Original token-mean PPO when each trajectory's
   terminal prefix appears once and the common active-token denominator is
   used.  Summing multiple prefix objectives would repeat-count early tokens
   and break this identity.
3. A nonzero locally feasible step exists only under continuity, finite-batch
   evaluation, and strict feasibility at the behavior point.  The fixed grid
   may still contain no nonzero feasible point.
4. Writer-path f-divergence bounds its memory-state push-forward divergence
   only under absolute continuity and a fixed measurable environment/state
   materialization kernel.

## Mechanism measurements and decision experiment

Ledger analysis reports per-write log-ratio, cumulative prefix log-ratio,
their variances and cross-turn covariance contribution, per-write versus
cumulative-prefix ESS, writer length/turn count, token KL and clip fraction.
All predictive diagnostics are grouped by trajectory and optimizer step; no
observation-level random leave-one-out analysis is permitted.  Covariance is
descriptive, not evidence of causal recurrent feedback.

The minimum decisive comparison is: (A) whole-prefix + hard rollback, (B)
whole-prefix + feasible backtracking, and (C) Original tokenwise objective +
the same feasible controller, against the read-only certified Original curve.
`B>A` identifies a controller contribution; `B>C` is required for an
independent whole-prefix-objective contribution.  `B≈C` reframes the paper
around prefix certification.  `C>B` eliminates the whole-prefix candidate.
Persistent alpha <= 1/32 motivates a prefix-metric preconditioned direction;
no result permits relaxing q_min or the cap.

# Closed-loop randomness estimand contract v6

The primary mode is frozen before outcomes. D is temperature-0 deterministic-protocol value, S is seed-marginal stochastic-policy value, and F is one frozen-seed realization for screening only. D is the default for a future Audit16 and needs no stochastic rollout.

S requires policy-specific independent non-overlapping seed namespaces and a prefrozen K. Replicates are averaged within each policy × stable example before example-level comparisons; seeds never increase independent n. Same-integer-seed CRN is only a corrected-trajectory-seed, BCI `PASS_COUPLED`, disjoint-namespace coupling sensitivity.

Every stochastic request is keyed by `H(experiment, mode, policy_or_CRN, example, replicate, turn, component, request_role)`. Writer, certificate, and terminal-reader requests are independently addressable at every turn. Sequential PRNG consumption position is never trajectory identity. A D-primary stochastic reversal is `STOCHASTIC_NONTRANSPORT` and does not erase deterministic IUT; invalid S is `STOCHASTIC_POLICY_MEAN_INVALID`; F is always single-seed realized screening.

The exact CPU audit records that Bernoulli(.8) minus Bernoulli(.6) has seed-marginal difference .2 although a single realization can differ by one or zero. It also verifies that one extra sequential random call can desynchronize turn two, while keyed addressing restores the intended one-to-one request identity.

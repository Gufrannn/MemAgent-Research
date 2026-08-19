# Horizon estimands and sole Shape A primary

Every replay row declares one estimand mode. E0 is an immediate probe. EH is the regime-conditional total execution effect of one update under a frozen exogenous future regime. EF is a forced-realized-suffix controlled effect and is construct-only. VG is full closed-loop fixed-policy value and requires a real fixed policy run from a common initial state.

For EH, the pair join fixes the upstream state, one materialized candidate, writer/reader coupling and endpoint version, plus `exogenous_suffix_contract_hash`, `future_policy_hash`, and `horizon`. The exogenous suffix contract hashes only future chunks, order, and budget—not a realized transcript. Commit and retain may induce different future prompts, candidates, memory states, and decisions; each arm records its own realized trajectory hash and future hash chain. Those realized hashes are never pair keys. Legacy `suffix_hash` is rejected as ambiguous.

Shape A has one primary: EH terminal continuous normalized-F1 harm, `H_H=F1_retain-F1_commit`, where larger is more harmful. `tau_H=F1_commit-F1_retain=-H_H` is a direction-reversed secondary alias only. The primary is the held-out continuous prediction increment of `B0+D_star` over B0. T0 B0 contains only checkpoint, turn, support position, pre-state/chunk/old-memory lengths, horizon, and pre-action difficulty. Candidate length/edit/validity, branch completion, qualification, truncation, and failure reasons cannot enter B0.

The v3 row unit is `(stable_example_id, checkpoint_hash, write_id)`, while independent n remains stable examples. The initial outcome-blind manifest fixes `m_i^target`; all R=0 D-star exclusions remain in that denominator and are not arm failures. The prebranch R=1 measurable stratum fixes `m_i^elig`, with weight `1/m_i^elig`. Reports separately show measurement availability and R=0 exclusions, paired closure, full-closure target weight, postbranch missing weight, and arm asymmetry. Missing weight is never reallocated; complete-case, IPW, and imputation primaries are forbidden.

`D_star` is continuously calibrated within a frozen `turn_type × required-component-pattern` joint-null stratum as `(raw−median)/MAD`, clipped to `[-5,5]`; larger means worse. It is not truncated at zero. Legacy `max(0,W−q95)` is forbidden and q95 is secondary anomaly flag only. Missing components yield R=0 rather than zero fill, and insufficient joint-null yield cannot borrow another turn's scale.

E0 F1/probes, terminal EM, grounding, distribution divergence, and terminal-minus-immediate are secondary. EF is construct-only and VG is separate. Checkpoints, turns, writes, reader repeats, and horizon rows do not increase n; cross-fitting/bootstrap cluster by stable example. Result-driven endpoint/subset choice or row deletion is `HORIZON_ENDPOINT_SELECTION_INVALID`.

The included freeze is outcome-blind and authorizes neither training nor reading NOOP scientific outcomes.

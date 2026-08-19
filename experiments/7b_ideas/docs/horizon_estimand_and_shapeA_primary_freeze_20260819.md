# Horizon estimands and sole Shape A primary

Every replay row declares one estimand mode. E0 is an immediate probe. EH is the regime-conditional total execution effect of one update under a frozen exogenous future regime. EF is a forced-realized-suffix controlled effect and is construct-only. VG is full closed-loop fixed-policy value and requires a real fixed policy run from a common initial state.

For EH, the pair join fixes the upstream state, one materialized candidate, writer/reader coupling and endpoint version, plus `exogenous_suffix_contract_hash`, `future_policy_hash`, and `horizon`. The exogenous suffix contract hashes only future chunks, order, and budget—not a realized transcript. Commit and retain may induce different future prompts, candidates, memory states, and decisions; each arm records its own realized trajectory hash and future hash chain. Those realized hashes are never pair keys. Legacy `suffix_hash` is rejected as ambiguous.

Shape A has one primary: EH terminal continuous normalized-F1 harm, `H_H=F1_retain-F1_commit`, tested as the held-out continuous prediction increment of `B0+D_star` over `B0`. B0 contains checkpoint, turn type, support position, length, horizon, truncation/validity, and pre-frozen difficulty. Candidate descendants and post-outcome fields are forbidden.

E0 F1/probes, terminal EM, grounding, distribution divergence, and terminal-minus-immediate are secondary. EF is construct-only and VG is a separate closed-loop study. The independent unit is always `stable_example_id`; turns, writes, reader repeats, and horizon rows do not increase n, and cross-fitting/bootstrap cluster by example. Choosing E0 versus EH, endpoint, horizon, turn subset, or deleting negative/invalid rows after results is `HORIZON_ENDPOINT_SELECTION_INVALID`.

The included freeze is outcome-blind and authorizes neither training nor reading NOOP scientific outcomes.

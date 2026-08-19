# Closed-loop terminal attribution v8 / resource-mode IUT

Closed-loop terminal output is attributed only to the total difference between complete policy packages. The contrasts `GC-GF`, `GC-GN`, and `GC-GS` are frozen before outcomes. Each reports win/tie/loss and positive/negative terminal-effect mass. The point summary `GC-max(control)=min(GC-GF,GC-GN,GC-GS)` may also be reported.

Oracle semantics are orthogonal. D mode permits a deterministic pointwise package oracle. S mode must estimate conditional policy means with independent assignment/evaluation seed folds; raw stochastic maxima are coupling-dependent hindsight luck. F mode permits only a seed-manifest hindsight envelope. Oracle failure appends `WITH_ORACLE_OPPORTUNITY_INVALID` and never revokes a passed terminal IUT.

“GC beats the best control” is an intersection-union claim. All three pairwise contrasts must pass the same frozen SESOI and interval-lower-bound gate. Any failure yields `CLOSED_LOOP_CONTROL_DOMINANCE_NO_GO`. Selecting one control after outcomes or substituting an ordinary best-control interval is forbidden.

Earlier policy actions cause state, prompt, candidate, certificate, and response paths to diverge. Therefore terminal outputs may not be named harmful commit, beneficial rejection, turn-local credit, rollback-caused rescue, or certificate-mediated gain. Local action attribution would require a new same-state, same-materialized-candidate commit/NOOP intervention at the visited closed-loop state; this suite does not add it. The hard gate name is `terminal_pairwise_IUT`.

Oracle assignments from the same Confirm32 may not train or tune a selector, D-star, certificate, or gate. A future package selector would have to be frozen before outcomes and tested on a new independent confirmation set; it is not authorized here. No oracle quantity rescues a failed terminal IUT.

Composition transport is an orthogonal audit. Qualified `V_splice-V_direct_GC` signed gap and MAE may show myopic nontransport, but neither statistic is an actionability gate. A direct GC terminal IUT pass remains actionability even when splice reverses; a failed terminal IUT remains `NO_GO` even when splice agrees.

Resource mode is also frozen before outcomes. A accuracy-first retains raw QA; incomplete cost qualification adds `WITH_COST_UNQUALIFIED` without erasing IUT. B fixed-budget requires a common vector and keeps every over-budget example under frozen truncation/skip/fallback rules; failure is `FIXED_BUDGET_POLICY_VALUE_INVALID`. Raw protocol QA is never silently relabeled equal-budget, utility, or practical advantage. The parent schema is `closed-loop-commit-v8-resource-mode`.

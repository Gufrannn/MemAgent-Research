# Closed-loop terminal attribution v4 / IUT

Closed-loop terminal output is attributed only to the total difference between complete policy packages. The contrasts `GC-GF`, `GC-GN`, and `GC-GS` are frozen before outcomes. Each reports win/tie/loss and positive/negative terminal-effect mass. The point summary `GC-max(control)=min(GC-GF,GC-GN,GC-GS)` may also be reported.

Oracle semantics are separated. `V_fixed*=max_g E[Y_g]` is the best value among the four executed fixed policies. `V_clair=E[max_g Y_ig]` selects a package only after seeing every example's terminal potential outcomes, so it is a clairvoyant sample upper bound, not an executable fixed policy. `Opportunity_package=V_clair-V_fixed*`; `Regret_GC_clair=V_clair-V_GC` is a descriptive clairvoyant gap and must not be called regret to the best fixed policy. Complementary outcomes can have `V_fixed*=.5`, `V_clair=1`, and opportunity `.5` with no fixed policy of value one.

“GC beats the best control” is an intersection-union claim. All three pairwise contrasts must pass the same frozen SESOI and interval-lower-bound gate. Any failure yields `CLOSED_LOOP_CONTROL_DOMINANCE_NO_GO`. Selecting one control after outcomes or substituting an ordinary best-control interval is forbidden.

Earlier policy actions cause state, prompt, candidate, certificate, and response paths to diverge. Therefore terminal outputs may not be named harmful commit, beneficial rejection, turn-local credit, rollback-caused rescue, or certificate-mediated gain. Local action attribution would require a new same-state, same-materialized-candidate commit/NOOP intervention at the visited closed-loop state; this suite does not add it. The hard gate name is `terminal_pairwise_IUT_and_regret`.

Clairvoyant assignments from the same Confirm32 may not train or tune a selector, D-star, certificate, or gate. A future package selector would have to be frozen before outcomes and tested on a new independent confirmation set; it is not authorized here. High `V_clair` never rescues a failed terminal IUT.

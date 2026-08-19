# Closed-loop composition-gap qualification firewall

Closed-loop actionability and composition transport are orthogonal. Real end-to-end GC/GF/GN/GS executions and their three-way terminal IUT alone decide closed-loop actionability. `V_splice` and its composition gap answer only whether a frozen single-step construction transports to the realized closed-loop package. A bad splice cannot veto direct actionability, and a good splice cannot rescue failed terminal IUT.

A composition gap is identified only when direct and splice rows share the same initial manifest, checkpoint, writer-reader contract, horizon, endpoint, and missingness rule. The splice algorithm and exact source-row hashes must be frozen before outcomes, and stable examples, direct source rows, and splice rows must form complete one-to-one mappings. Any mismatch yields `COMPOSITION_GAP_NOT_IDENTIFIED`; it cannot be described as feedback.

The frozen direction is `V_splice-V_direct_GC`. Every qualified report includes its signed mean and mean absolute error/error mass. Cancellation is not transport: gaps `+1,-1` have signed mean zero and MAE one. An endpoint-definition mismatch can create an apparent gap `.6`, but the firewall rejects it before reading outcomes. If direct GC passes every terminal control while a qualified splice reverses direction or has large frozen-SESOI error, the legal conclusion is `CLOSED_LOOP_ACTIONABILITY_WITH_MYOPIC_NONTRANSPORT`.

This audit adds no local intervention, rollout, optimizer step, training, or selector authorization.

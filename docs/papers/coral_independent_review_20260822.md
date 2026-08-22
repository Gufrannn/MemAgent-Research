# CORAL independent adversarial review record

Date: 2026-08-22

Scope: paper framing, E0 construction, E1 estimand and producer, actor/trainer
integration, distributed sketch oracle, fail-closed audit, and locked source
release. The reviewer inspected the real trainer and H20 entry paths rather
than only helper functions. No H20 execution or performance result was shown
to the reviewer.

## Final source conclusions

- Paper framing: **GO**.
- Corrected CPU E0: **GO** as an exact-enumeration mechanism counterexample,
  not as trainer integration evidence.
- E1 producer source: **GO**, conditional on the runtime gates below.
- Locked source release: **GO**.
- Method-T5 training: **not yet authorized**.

The review initially rejected the standalone COSI certificate formulation and
then repeatedly returned CORAL for reframe. Blockers included claim/role-mask
mismatch, a separable-parameter theory for a shared LM, an inert inactive-role
KL claim, an incorrectly normalized E0, terminal-action sampling confounded
with occupancy in E1, step-dependent root identity, correlated projection
bases mislabeled independent, root pseudoreplication, and an oracle sealer that
could accept a hand-written self-hashed PASS. The final source closes these
issues by narrowing the method to nonfinal writer -> terminal answer, retaining
Original's full-token denominator, rebuilding the shared-parameter E0,
resampling both terminal branches at fixed proposal weights, binding
step-independent dataset roots, using deterministic nonlinearly
basis-separated sketches, clustering inference by eight writer proposals, and
strictly validating and embedding the complete oracle v3 report. The first real
two-rank H20 execution exposed a `7.45e-09` difference between NCCL and
sequential rank0 FP32 reduction orders; v3 freezes a `1e-7` assembly aperture
while leaving optimizer/checkpoint reversibility and all substantive projection,
FSDP, denominator, and calibration gates unchanged.

## Mandatory runtime conditions

The source review does not certify GPU execution. Before T5, a canonical
ascending pair of H20 GPUs must pass the two-rank NCCL/FSDP oracle, including
the 1,000,003-coordinate collision calibration and writer-plus-answer
denominator closure. The real trainer must then produce exactly eight writer
proposals and 32 never-reused roots, and the sealer and final E1 auditor must
PASS. The Original read-only import and an externally issued, content-addressed
paper-framing certificate are also required. The committed manifest remains
`training_authorized=false`; only external authenticated gates can unlock the
T5 entry.

## Local verification observed by the final review

The final reviewed suite reported 19 passing tests and three Torch-dependent
skips, plus Python compilation. After that review, the source added stricter
JSON integer typing and external-gate authorization tests; these must be
included in the final pre-commit regression record.

# CORAL release status

- `43b722d`: **SUPERSEDED / NOT RELEASE**.
- `d6cccf99d39859b6bc843fd3abee2d92e9f7e4e4`: **SUPERSEDED / NOT RELEASE** after the unified Original evidence and continuous-T25 protocol correction.
- `b019d925ebf32b92b8a97c1c5a2a70a752880b62`: **SUPERSEDED / NOT RELEASE** because its frozen Original-ledger SHA contained a one-character transcription error detected by H20 fail-closed preflight.
- `3f6597f3bd35e58aac387d1651e0c9f9db6600b7`: **SUPERSEDED / NOT RELEASE** because the supplied Original5 metric-row digest contained an extra character; H20 fail-closed materialization detected it before E1 or Method training.
- `20382a5e035cd08fe90852267c142e51d7bbaa38`: **SUPERSEDED / NOT RELEASE** after the first real two-rank H20 oracle showed a `7.45e-09` FP32 reduction-order difference against a sequential rank0 reference; its exact-zero comparison was numerically invalid while all substantive oracle errors passed.
- Only the exact remote SHA identified as `RELEASE` in the final handoff may be used on H20.

The released entry must authenticate the shared read-only Original/S128 evidence roots, materialize the Original comparison curve from its certified artifact inventory, train Method once from the fresh base through T25, apply the cheap numerical/ledger/weight-sync T5 health check, and run all five fixed-S128 evaluations only after T25.

# CORAL release status

- `43b722d`: **SUPERSEDED / NOT RELEASE**.
- `d6cccf99d39859b6bc843fd3abee2d92e9f7e4e4`: **SUPERSEDED / NOT RELEASE** after the unified Original evidence and continuous-T25 protocol correction.
- `b019d925ebf32b92b8a97c1c5a2a70a752880b62`: **SUPERSEDED / NOT RELEASE** because its frozen Original-ledger SHA contained a one-character transcription error detected by H20 fail-closed preflight.
- `3f6597f3bd35e58aac387d1651e0c9f9db6600b7`: **SUPERSEDED / NOT RELEASE** because the supplied Original5 metric-row digest contained an extra character; H20 fail-closed materialization detected it before E1 or Method training.
- `20382a5e035cd08fe90852267c142e51d7bbaa38`: **SUPERSEDED / NOT RELEASE** after the first real two-rank H20 oracle showed a `7.45e-09` FP32 reduction-order difference against a sequential rank0 reference; its exact-zero comparison was numerically invalid while all substantive oracle errors passed.
- `2d0d89cd6cb67fa0981f0a1ebc65d5a9fdd71ff6`: **SUPERSEDED / NOT RELEASE** after its real H20 E1 v4 run exposed TensorDict 0.5/0.6 serialization failure when cloning the intentional zero-tensor root-metadata batch. The failure occurred at update 1 before any optimizer update; the v4 output is failure evidence and must never be reused.
- `d9a6b72cbdee67fe68b2aa60dffab4bc0fa05a0c`: **SUPERSEDED / NOT RELEASE** after E1 v5 passed research preflight but its new standalone clone oracle could not import the repository-local `verl` package. No distributed oracle or trainer update started; v5 remains append-only failure evidence.
- `67c90d008f8424ae21f28aec768184495bfcc534`: **SUPERSEDED / NOT RELEASE** after E1 v6 passed both runtime oracles and performed its first writer update, then exposed a missing per-turn `uid` in the cached recurrent materialization before writing a proposal receipt. The v6 weights/output are diagnostic failure evidence and must never be resumed or reused.
- Only the exact remote SHA identified as `RELEASE` in the final handoff may be used on H20.

The released entry must authenticate the shared read-only Original/S128 evidence roots, materialize the Original comparison curve from its certified artifact inventory, train Method once from the fresh base through T25, apply the cheap numerical/ledger/weight-sync T5 health check, and run all five fixed-S128 evaluations only after T25.

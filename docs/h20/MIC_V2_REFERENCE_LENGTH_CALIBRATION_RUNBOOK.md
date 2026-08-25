# MIC-v2 reference-length calibration (H20)

This is the first GPU-producing stage after the E0 and data-freeze certificates.
It does not authorize E1 or actor training. It runs the frozen fresh
Qwen2.5-7B base on the 64 label-blind calibration roots with four independent
replicas and computes `Lbar_ref` over all `64*4*9=2304` scheduled action slots.
Each active slot receipts the exact sampled-token prefix mask. Each writer slot
also receipts the raw token action, parsed materialized-memory tokens, frozen
parser version, and the nested post-write afterstate hash.
Before the CPU finalizer can sign the statistic, a second fresh strict-vLLM
process independently regenerates every active action from the same frozen
prompt/seed schedule and requires canonical token-for-token equality.

CPU preflight projects the frozen split into a label-blind source artifact. The
GPU producer can read only the certified question, context, stable identity, and
source-position fields in that artifact; it cannot open the source parquet or
read rewards, gold answers, the exposed S128 set, E1, oracle, confirm, or metrics.

## Fresh launch on physical H20 4,5

```bash
set -euo pipefail

export MEMAGENT_MIC_V2_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_MIC_V2_REPO_DIR=/data/cw/memagent_work/code/MemAgent-MIC-Audit
export MEMAGENT_MIC_V2_EXPECTED_COMMIT=REPLACE_WITH_APPROVED_EXACT_COMMIT
export MEMAGENT_MIC_V2_CALIBRATION_RUN_ID=mic-v2-lbar-20260825-r1
export MEMAGENT_MIC_V2_GPU_PAIR=4,5
unset MEMAGENT_MIC_V2_CALIBRATION_RESUME

cd "$MEMAGENT_MIC_V2_REPO_DIR"
test "$(git rev-parse HEAD)" = "$MEMAGENT_MIC_V2_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

screen -dmS mic-v2-lbar-r1 bash -lc '
set -euo pipefail
cd "$MEMAGENT_MIC_V2_REPO_DIR"
bash scripts/h20/run_qwen25_7b_mic_v2_reference_length_calibration.sh
'
```

The entry acquires both canonical locks
`locks/memagent_h20_gpu_4.lock` and `locks/memagent_h20_gpu_5.lock`. An occupied
GPU or lock is `NO_GO`; the entry never kills another process.

## Monitor

```bash
export MIC_V2_LBAR_ROOT=/data/cw/memagent_work/logs/mic_v2_reference_length/mic-v2-lbar-20260825-r1
screen -ls | grep mic-v2-lbar-r1 || true
tail -n 80 -F "$MIC_V2_LBAR_ROOT/calibration.log"
```

Progress lines report `completed` out of 256 trajectories. The expected wall
time on two H20s is approximately 60--180 minutes including the independent
full replay; output length determines the
real duration.

## Recover after a connection or process failure

Do not delete, edit, or move the attempt directory. Completed trajectory
records form an append-only hash chain and are reused only when they are the
exact expected prefix.

```bash
export MEMAGENT_MIC_V2_CALIBRATION_RESUME=1
screen -dmS mic-v2-lbar-r1-resume bash -lc '
set -euo pipefail
cd "$MEMAGENT_MIC_V2_REPO_DIR"
bash scripts/h20/run_qwen25_7b_mic_v2_reference_length_calibration.sh
'
```

## Success evidence

```bash
python -m json.tool "$MIC_V2_LBAR_ROOT/certificates/reference_length.json"
sha256sum \
  "$MIC_V2_LBAR_ROOT/certificates/p0.json" \
  "$MIC_V2_LBAR_ROOT/certificates/execution.json" \
  "$MIC_V2_LBAR_ROOT/certificates/gpu_replay.json" \
  "$MIC_V2_LBAR_ROOT/certificates/reference_length.json" \
  "$MIC_V2_LBAR_ROOT/trajectories/length_receipts.jsonl"
```

Proceed to E1-dev implementation/execution only after the certificate reports
`MIC_V2_REFERENCE_LENGTH_CALIBRATION_PASS` and independent review accepts the
exact evidence SHA.

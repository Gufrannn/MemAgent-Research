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
Fresh-base verification recursively requires the exact 11 loading-effective
files plus four SHA-pinned inert repository artifacts; any unknown file,
symlink, adapter, tokenizer override, chat template, or remote code is `NO_GO`.
The launcher pins `VLLM_USE_MODELSCOPE=False`; the GPU runner and both execution
receipts reject any other config-loader environment before model loading.
Every writer afterstate is bound to the same prepared content root's frozen
chunk-token receipt; a cross-root receipt substitution is `NO_GO`.

## Fresh launch on physical H20 4,5

```bash
set -euo pipefail

export MEMAGENT_MIC_V2_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_MIC_V2_REPO_DIR=/data/cw/memagent_work/code/MemAgent-MIC-Audit
export MEMAGENT_MIC_V2_EXPECTED_COMMIT=REPLACE_WITH_APPROVED_EXACT_COMMIT
export MEMAGENT_MIC_V2_CALIBRATION_RUN_ID=mic-v2-lbar-20260825-r2
export MEMAGENT_MIC_V2_GPU_PAIR=4,5
unset MEMAGENT_MIC_V2_CALIBRATION_RESUME

cd "$MEMAGENT_MIC_V2_REPO_DIR"
test "$(git rev-parse HEAD)" = "$MEMAGENT_MIC_V2_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

screen -dmS mic-v2-lbar-r2 bash -lc '
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
export MIC_V2_LBAR_ROOT=/data/cw/memagent_work/logs/mic_v2_reference_length/mic-v2-lbar-20260825-r2
screen -ls | grep mic-v2-lbar-r2 || true
tail -n 80 -F "$MIC_V2_LBAR_ROOT/calibration.log"
```

Progress lines report `completed` out of 256 trajectories. The expected wall
time on two H20s is approximately 60--180 minutes including the independent
full replay; output length determines the
real duration.

## Recover after a connection or process failure

The historical `mic-v2-lbar-20260825-r1` attempt contains cross-root afterstate
receipts rejected by independent replay. It is immutable failed evidence: never
resume, edit, move, or delete it. The recovery procedure below applies only to
a valid partial `r2` prefix created by the exact corrected commit.

Do not delete, edit, or move the attempt directory. Completed trajectory
records form an append-only hash chain and are reused only when they are the
exact expected prefix.

```bash
export MEMAGENT_MIC_V2_CALIBRATION_RESUME=1
screen -dmS mic-v2-lbar-r2-resume bash -lc '
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

The accepted r2 raw/canonical SHA bundle and `Lbar_ref` are frozen in
`manifests/h20/qwen25_7b_mic_v2_reference_length_authority.json` at immutable
Git commit `ecd9c5e7829245da9fc158e1da0cc1953054faf7`; the raw authority-file SHA is
`c8b4f3d3e93099cffc65e2d4ad2465b42598cca381b2e9b70d81a45adf162096`.
The verifier reads the authority and all producer code from their historical
Git objects, requires both producer and authority commits to be ancestors of
the current checkout, and reconstructs the 64 label-blind inputs from the
frozen data split. A later E1 checkout may change implementation files but may
not alter or self-resign this authority.

Every E1-dev preregistration and P0 must independently pin those two authority
constants and run the read-only authority entry before opening E1 data:

```bash
bash scripts/h20/verify_qwen25_7b_mic_v2_reference_length_authority.sh
```

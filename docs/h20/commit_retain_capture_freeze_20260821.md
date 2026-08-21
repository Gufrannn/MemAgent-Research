# Qwen2.5-7B base-I COMMIT(C) versus exact RETAIN(old) capture

This closure runs the next minimum evidence experiment only. It uses the same
Qwen2.5-7B-Instruct base-I checkpoint, the certified stable-I S128 identity,
the existing fixed HotpotQA S128 parquet, and the same four outcome-blind
writer-turn-0 prompt-length strata. It runs strict vLLM 0.8.2 with TP=2 on
the explicitly supplied pair of physical H20 GPUs. The pair may be
non-contiguous (for example `2,4`) but must be written in canonical ascending
`A,B` form. It never attaches a trainer, restores an optimizer,
updates the actor, selects a method, or changes `sources/`.

The Python coordinator imports the native recurrent utilities before it builds
the vLLM engine. Those utilities import PyTorch/TensorDict and may initialize
CUDA in the coordinator. vLLM 0.8.2 otherwise defaults local workers to
`fork`, which is unsafe after CUDA initialization. This contract therefore
freezes `VLLM_WORKER_MULTIPROC_METHOD=spawn`. P0 asks the installed vLLM
runtime which method it observes and verifies a real Python multiprocessing
`spawn` context. Every capture/run receipt records that method and whether CUDA
was already initialized in the coordinator immediately before engine creation.

## Native-interface evidence and minimal patch

The native `recurrent.impls.memory.MemoryAgent.update` has one non-final state
transition: it assigns `unpad(...responses...)` directly into every active
memory slot. There is no COMMIT/RETAIN argument and no exact-byte RETAIN path.
P0 records the complete method-source digest and fails if this fact changes.

The minimum patch therefore lives outside the training path in
`recurrent.research.commit_retain_capture`. It gives token state one canonical
serialization (`memagent.state-token-u32le.v1`: unsigned 32-bit little-endian
token IDs), then requires a loaded-state receipt to contain those exact bytes,
the byte SHA-256, the token IDs, and the token-array SHA-256. RETAIN loads the
already-materialized old-state blob directly; it never decodes/re-encodes it,
casts it through a tensor, or regenerates the writer. COMMIT loads the one
already-materialized candidate blob. The ordinary MemoryAgent/trainer path is
unchanged.

## Frozen intervention

P0 resolves the same four S128 examples and freezes one outcome-blind writer
timepoint per example using `max(1, (total_writer_turns - 1) // 2)`. P0 rejects
an example unless there is at least one prefix writer turn and at least one
future writer chunk.

For each frozen stable write, one coordinator invocation constructs one
strict-vLLM engine (TP=2 may create its normal local worker processes) and
executes this exact order:

1. Execute and record every prefix writer prompt/output to obtain old state.
2. At the frozen writer turn, generate candidate C exactly once and materialize
   its state bytes before either arm starts.
3. Run COMMIT(C), loading exact candidate bytes.
4. Run RETAIN(old), loading exact pre-write bytes.
5. Give both arms the identical future chunks, future horizon, base checkpoint,
   writer/reader decode settings, per-turn request seeds, disabled-cache
   contract, and configured generation budgets.

Every record carries stable example/root/write/turn IDs; old/candidate bytes and
token hashes; writer prompt text/tokens/checkpoint; each loaded-state receipt;
every prefix/future writer output; both final reader outputs; recomputed HotpotQA
outcomes; actual call/token-cost receipts; GPU UUID/name; engine/process/call
identity; and explicit zero-training receipts. Realized token counts are
measurements and may differ due to the state intervention; the number of calls,
horizon, and configured budgets are equal by contract.

The pair artifact is hash-chained as it is generated. Immediately before the
GPU child, its direct parent shell issues a single-use credential and records
that authorization in the append-only supervisor ledger. The same ledger then
records the completed four-pair capture and the read-only audit. The audit
authenticates the original parent/child PID and credential before it
reconstructs identities, byte states, state chains, source
chunks, prompts/outputs, RNG, decode, cost, and outcomes. A handwritten JSON
`PASS` field is an unknown field and fails validation.

## H20 commands

Replace `<FULL_40_CHAR_COMMIT>` with the pushed commit. Use a new run ID; an
existing run directory is never resumed or overwritten.

```bash
export MEMAGENT_COMMIT_RETAIN_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_COMMIT_RETAIN_REPO_DIR=/data/cw/memagent_work/code/MemAgent-Research
export MEMAGENT_COMMIT_RETAIN_EXPECTED_COMMIT=<FULL_40_CHAR_COMMIT>
export MEMAGENT_COMMIT_RETAIN_RUN_ID=commitretain4_20260821r1
export MEMAGENT_COMMIT_RETAIN_GPU_PAIR=2,4

cd "$MEMAGENT_COMMIT_RETAIN_REPO_DIR"
git fetch origin
git switch h20/qwen25-7b-commit-retain-capture-20260821
git pull --ff-only
test "$(git rev-parse HEAD)" = "$MEMAGENT_COMMIT_RETAIN_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

bash scripts/h20/preflight_qwen25_7b_commit_retain.sh

screen -L -Logfile \
  "$MEMAGENT_COMMIT_RETAIN_WORK_ROOT/logs/commit_retain_capture_screen_20260821.log" \
  -dmS xinman-commit-retain-0821 \
  bash -lc '
cd "$MEMAGENT_COMMIT_RETAIN_REPO_DIR" || {
  echo "NO_GO: repository unavailable"
  exec bash
}
bash scripts/h20/run_qwen25_7b_commit_retain.sh
rc=$?
echo
echo "COMMIT_RETAIN_EXIT_CODE=$rc"
echo "screen 已保留；输入 exit 才会关闭"
exec bash
'

screen -D -r xinman-commit-retain-0821
```

P0 asks `nvidia-smi` to authenticate both requested indices, UUIDs, and H20
names. The wrappers hold one lock per physical GPU, in ascending order, for the
whole invocation. Thus runs on `4,5` and `5,6` cannot overlap. The immutable
run root ends in `<run-id>_gpuA_B`; changing the pair requires a new P0 and a
new root while leaving the scientific protocol unchanged.

After screen exits, the metadata-only re-audit is:

```bash
"$MEMAGENT_COMMIT_RETAIN_WORK_ROOT/.venv/bin/python" \
  tools/h20/audit_qwen25_7b_commit_retain.py \
  --manifest manifests/h20/qwen25_7b_commit_retain_capture_seed2026.json \
  --verify-existing
```

Expected wall time on two H20s is approximately 25–60 minutes: P0 and the full
model/data/code binding about 2–6 minutes, one model load plus roughly 50–60
strict single-request generations about 20–50 minutes, and the CPU audit under
2 minutes. The only terminal decision is
`COMMIT_RETAIN_CAPTURE_AUDIT_COMPLETE`; it is an evidence-completeness decision,
not a method or performance claim.

# Qwen2.5-7B exact-32 COMMIT versus RETAIN capture

This is the reviewed GPU producer for the preregistered paired-effect
development evidence. It runs the existing Qwen2.5-7B-Instruct base recurrent
interface on exactly 32 frozen rows from the existing HotpotQA S128 authority.
For every row it materializes one candidate before branching, then runs both
COMMIT(candidate) and exact RETAIN(old-state) through one strict-vLLM engine.
It does not attach a trainer, update an actor, authorize a new method, or create
a paper-performance result.

The 32 rows, stable IDs, intervention turns, 4 folds of 8, decode contract and
admissibility thresholds are frozen in
`manifests/h20/qwen25_7b_paired_effect_capture32_preregistration.json` before
the first generation. The earlier capture4 is pipeline-pilot evidence only: it
cannot fill a missing row, replace a failed row, or be stitched into this run.

## GPU and lock contract

The operator must explicitly provide exactly two different physical GPU
indices in ascending order, for example `4,5`, `6,7`, or `5,6`. There is no
automatic GPU selection. P0 verifies that both selected devices report as
NVIDIA H20, freezes their physical indices and UUIDs, and binds
`CUDA_VISIBLE_DEVICES` to the same ordered pair. Compute mode must be
`Default`; MIG must be disabled or unsupported.

Before either P0 or capture touches a GPU, the shell holds and exports an
inherited file descriptor for each
`locks/memagent_h20_gpu_N.lock`. To remain mutually exclusive with older H20
jobs, choosing either GPU 4 or 5 also holds
`locks/memagent_gate_a_gpu_4_5.lock`; choosing either GPU 6 or 7 also holds
`locks/memagent_gate_a_gpu_6_7.lock`. Thus a cross-pair such as `5,6` holds two
per-device locks and both legacy aggregate locks. A busy device or held lock is
a fail-closed P0 error; the scripts never kill another user's process.

## H20 commands

Use the pushed full commit and a never-before-used run ID. P0 is deliberately a
standalone command. Do not start capture unless it reports both the P0
certificate and the external preregistration anchor.

```bash
export MEMAGENT_CAPTURE32_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_CAPTURE32_REPO_DIR=/data/cw/memagent_work/code/MemAgent-Research
export MEMAGENT_CAPTURE32_EXPECTED_COMMIT=<FULL_40_CHAR_COMMIT>
export MEMAGENT_CAPTURE32_RUN_ID=commitretain32_20260821r1
export MEMAGENT_CAPTURE32_PHYSICAL_GPUS=6,7

cd "$MEMAGENT_CAPTURE32_REPO_DIR"
test "$(git branch --show-current)" = h20/qwen25-7b-paired-effect-pipeline-20260821
test "$(git rev-parse HEAD)" = "$MEMAGENT_CAPTURE32_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

bash scripts/h20/preflight_qwen25_7b_commit_retain_capture32.sh
```

After P0 passes, run the single uninterrupted exact-32 capture in `screen`:

```bash
screen -L -Logfile \
  "$MEMAGENT_CAPTURE32_WORK_ROOT/logs/commit_retain_capture32_screen_${MEMAGENT_CAPTURE32_RUN_ID}.log" \
  -dmS "xinman-commit-retain32-$MEMAGENT_CAPTURE32_RUN_ID" \
  bash -lc '
cd "$MEMAGENT_CAPTURE32_REPO_DIR" || {
  echo "CAPTURE32_NO_GO: repository unavailable"
  exec bash
}
bash scripts/h20/run_qwen25_7b_commit_retain_capture32.sh
rc=$?
echo
echo "CAPTURE32_EXIT_CODE=$rc"
echo "screen retained; type exit to close it"
exec bash
'

screen -D -r "xinman-commit-retain32-$MEMAGENT_CAPTURE32_RUN_ID"
```

The capture wrapper requires the exact standalone P0 prefix, obtains the same
GPU and legacy locks, has its direct parent issue a single-use child
credential, and launches one coordinator/one vLLM engine. It accepts only a
complete artifact containing all 32 frozen stable examples, both arms, every
required prefix/future turn, both final readers, all outcomes and receipts.
The read-only audit then writes a final report and an external terminal anchor.

## Attrition and provenance

P0 exclusively creates
`provenance/commit_retain_capture32/<run-id>.preregistration.json` outside the
run directory. The successful terminal audit exclusively creates the matching
`.terminal.json`, binding the P0, resolved manifest, complete capture ledger,
run receipt, supervisor ledger, final report, stable inventories and exact GPU
identity. Both files state `PENDING_EXTERNAL_SIGNATURE`: they are export
candidates in a location outside the run root, not a fabricated claim that a
third party has already signed them.

Once the preregistration anchor exists, a partial capture is `FAIL/ATTRITION`,
not a smaller analysis set. Missing one example, one arm, one required turn,
one reader outcome or one receipt is fatal. The failed run is never resumed or
supplemented; retry only under a new run ID and regenerate the complete same 32
rows. An empty but valid model answer remains an observed zero-score outcome;
missing token/hash evidence does not.

The hash-chained supervisor state machine is exactly `s0_preflight`,
`capture_authorization`, `capture_started`, `capture_complete`, `audit_result`.
Every persisted supervisor record and both local export anchors are validated
at runtime against their frozen Draft 2020-12 schemas; the audit also
independently reconstructs their paths, hashes, identities, exact 353-call
count, and claim-firewall fields.
The last two states are admissible only with exactly 32 unique pair IDs and 32
unique stable-write IDs. A ledger stopped after authorization or start is
therefore durable attrition evidence; it is not a resumable prefix.

Only a final report with 32 unique frozen stable examples and a matching local
terminal export candidate is capture-complete; externally trusted provenance
remains pending until an independent system signs or immutably stores that
candidate. Even then it is development
evidence for reviewing the preregistered paired-effect scorer, not permission
to train, not a causal claim, and not an untouched S128 paper result.

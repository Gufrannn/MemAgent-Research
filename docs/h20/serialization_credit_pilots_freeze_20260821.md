# Qwen2.5-7B serialization-credit mechanism pilots (SMSB4 → Tetrad4)

This closure tests the new serialization-credit idea without training or changing the
Gate A checkpoint. It uses the existing fixed HotpotQA S128, the project-native
recurrent memory templates, Qwen2.5-7B-Instruct, strict vLLM 0.8.2, and physical H20
GPUs 6 and 7. It never changes `sources/` and never calls an actor update.

## Scientific boundary

SMSB4 first captures four outcome-blind, prompt-length-stratified recurrent examples.
It then reconstructs the final reader prompt directly from token IDs in twelve fresh
Python/vLLM processes (four examples × temperature-zero, matched-seed, and independent-
seed regimes). The gate requires exact prompt-token identity and exact temperature-zero
answer-token identity for all four examples. Matched-seed identity is diagnostic only.

Tetrad4 is generated only after the exact SMSB decision
`PASS_E_DET_SINGLE_REQUEST`. It executes generated, empty, irrelevant, shuffled, and
canonical-gold memories for each of the four examples in twenty additional fresh
Python/vLLM processes. This is a construction/competence pilot, not a performance or
causal-effect result. Audit32, method selection, and method training remain pending.

Every child re-authenticates the Git checkout, fixed parquet, runtime versions, actual
GPU UUID/name and `CUDA_DEVICE_ORDER`, and a full SHA-256 inventory of every model file.
Each GPU child is started by a new parent-supervisor Python process. The supervisor
issues one HMAC-authenticated, single-use credential, observes the actual child PID,
PPID and exit code, captures child stdout in an append-only file, and writes an
HMAC-authenticated receipt binding the result and stdout digests. The audit requires
unique supervisor PIDs, child PIDs, credential IDs and receipt IDs; result fields alone
cannot satisfy either gate. Evidence also binds actual
prompt/output token IDs and hashes, trajectory and per-turn seeds, generate-call indices,
one engine construction, and one generate call for each fresh replay. Tetrad scores are
recomputed from decoded answer tokens and the S128 ground truth, while Tetrad authoring
is independently rebuilt from the immutable parquet instead of trusting persisted pass
flags.

## H20 execution

Use a new task-scoped run ID. The expected commit must be the exact final commit on the
frozen branch; placeholders below must not be copied literally.

```bash
export MEMAGENT_SERIAL_CREDIT_WORK_ROOT=/data/cw/memagent_work
export MEMAGENT_SERIAL_CREDIT_REPO_DIR=/data/cw/memagent_work/code/MemAgent-Research
export MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT=<FULL_40_CHAR_COMMIT>
export MEMAGENT_SERIAL_CREDIT_RUN_ID=scpilot4_20260821r1

cd "$MEMAGENT_SERIAL_CREDIT_REPO_DIR"
git switch h20/qwen25-7b-serialization-credit-pilots-20260821
test "$(git rev-parse HEAD)" = "$MEMAGENT_SERIAL_CREDIT_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

bash scripts/h20/preflight_qwen25_7b_serialization_credit.sh
bash scripts/h20/run_qwen25_7b_smsb4.sh
bash scripts/h20/run_qwen25_7b_tetrad4.sh
```

The last script writes and authenticates the final report, then a read-only re-audit can
be run without starting a GPU engine:

```bash
/data/cw/memagent_work/.venv/bin/python \
  tools/h20/audit_qwen25_7b_serialization_credit.py \
  --manifest manifests/h20/qwen25_7b_serialization_credit_pilots_seed2026.json
```

Expected two-H20 wall time is approximately 60–120 minutes end to end: P0/full hashing
about 2–5 minutes, SMSB4 about 25–50 minutes, and conditional Tetrad4 plus final audit
about 30–65 minutes. The wide range is deliberate because the contract starts 32 fresh
TP=2 vLLM processes; model-load and process-cleanup time dominate. A SMSB failure stops
before all Tetrad work, and no failed run directory is overwritten or resumed in place.

The only admissible terminal claims are:

- `PASS_E_DET_SINGLE_REQUEST` for SMSB4;
- `TETRAD_PILOT4_CONSTRUCTION_GATE_PASS` for Tetrad4;
- `SERIALIZATION_CREDIT_PILOT4_PASS` for the authenticated combined closure.

None of these claims selects NCR or another method, authorizes training, or reports a
paper performance effect.

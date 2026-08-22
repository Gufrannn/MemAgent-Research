# TF-RWWPO H20 runbook (release-gated)

This runbook is not a claim that TF-RWWPO works.  Substitute only the exact
reviewed release SHA after PAPER GO, RELEASE GO, and a clean pushed branch.
Never reuse a run ID or output root.  Never kill another GPU process.

## Identity and evidence

```bash
cd /home/test001/memagent-rwwpo-release
git fetch origin h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822
git switch h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822 || \
  git switch --track -c h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822 \
    origin/h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822
git merge --ff-only origin/h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822
test "$(git rev-parse HEAD)" = '<REVIEWED_RELEASE_SHA>'
test -z "$(git status --porcelain)"

export RWWPO_REPO_DIR="$PWD"
export RWWPO_WORK_ROOT=/data/cw/memagent_work
export RWWPO_EXPECTED_COMMIT='<REVIEWED_RELEASE_SHA>'
export RWWPO_EXPECTED_BRANCH=h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822
export GPU_PAIR='<LOWER_IDLE_GPU>,<HIGHER_IDLE_GPU>'
export RWWPO_RUN_ID='tf_rwwpo_seed2026_primary_<SHORT_SHA>'
export RWWPO_PHASE=full
export RWWPO_ATTEMPT_ID=tf_rwwpo_primary
export RWWPO_MANIFEST="$RWWPO_REPO_DIR/manifests/h20/qwen25_7b_tf_rwwpo_seed2026.json"

export RWWPO_ORIGINAL_RESOLVED_MANIFEST=/data/cw/memagent_work/logs/original_t25_2gpu_frozen_20260821/certificates/p0_resolved_manifest.json
export RWWPO_ORIGINAL_RESOLVED_SHA256="$(sha256sum "$RWWPO_ORIGINAL_RESOLVED_MANIFEST" | awk '{print $1}')"
```

Materialize a new commit-bound E0 receipt and baseline-import receipt from the
already authenticated read-only bundle.  Do not modify Original artifacts and
do not rerun Original.  Both receipts must name `<REVIEWED_RELEASE_SHA>`.

## GPU preflight and one continuous fresh run

```bash
nvidia-smi -i "$GPU_PAIR" --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader
python3 tools/h20/audit_tf_rwwpo_source_firewall.py

screen -dmS tf-rwwpo-t25 bash -lc '
set -euo pipefail
cd "$RWWPO_REPO_DIR"
bash scripts/h20/run_qwen25_7b_tf_rwwpo.sh
'
```

The launcher takes both canonical per-GPU locks before CUDA initialization and
refuses occupied cards.  Training starts from the frozen fresh base, enables
the method at update 1, and targets T25 in one process.  At T5 it saves the
checkpoint and runs only the cheap mechanism gate: at least four nonzero
commits, median committed alpha >= 1/8, ESS/cap/behavior identity, rank
agreement, rollback closure, checkpoint inventory, and weight sync.  A failure
terminates the run and preserves its entire root.  A pass continues without
S128 evaluation and saves T10/T15/T20/T25.

## Monitoring and recovery

```bash
screen -r tf-rwwpo-t25
tail -F "$RWWPO_WORK_ROOT/logs/rwwpo/$RWWPO_RUN_ID/train.log"
```

If the process fails, do not append or resume that root.  Preserve screen log,
ledger, locks, checkpoint, and tombstone.  After processes exit, locks release,
and both GPUs are empty, fix only on a new reviewed commit and use a new semantic
run ID from fresh P0.  An unmatched transaction intent makes audit fail closed.

## Post-training audit and evaluation

After T25, run `audit_rwwpo_actual_loss.py` and `audit_rwwpo_run.py --target-step
25`, then a second read-only re-audit.  Only after both pass, evaluate the saved
T5/10/15/20/25 checkpoints using the same fixed-S128 manifest and strict-vLLM
launcher, import the certified Original five-anchor bundle read-only, and run
the per-anchor plus five-anchor auditors.  Performance rows remain separate
from the mechanism ledger.  B and C require different run IDs, manifests,
output roots, and release commits.

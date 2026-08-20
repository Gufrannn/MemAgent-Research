#!/usr/bin/env bash
set -euo pipefail

# Build and atomically publish a small, reviewable Gate A evidence archive.
# This never copies checkpoint shards and never modifies the original evidence.
# Run this script; do not source it into a shared login shell.

: "${MEMAGENT_GATEA_WORK_ROOT:?set MEMAGENT_GATEA_WORK_ROOT explicitly}"
: "${MEMAGENT_GATEA_REPO_DIR:?set MEMAGENT_GATEA_REPO_DIR explicitly}"
: "${MEMAGENT_GATEA_EXPECTED_COMMIT:?set MEMAGENT_GATEA_EXPECTED_COMMIT explicitly}"

case "$MEMAGENT_GATEA_WORK_ROOT:$MEMAGENT_GATEA_REPO_DIR" in
  /*:/*) ;;
  *) echo 'GATE_A_ARCHIVE_FAIL: work root and repo dir must be absolute paths' >&2; exit 2 ;;
esac

GATEA_ARCHIVE_EXPECTED_COMMIT=c3f987be5513cad2a9e95622dd6773726a7bf12e
GATEA_ARCHIVE_EXPECTED_BRANCH=h20/qwen25-7b-gatea-2gpu-frozen-20260820
GATEA_ARCHIVE_RELEASE_NAME=gate_a_r5_c3f987be
GATEA_ARCHIVE_PARENT="$MEMAGENT_GATEA_WORK_ROOT/archives"
GATEA_ARCHIVE_RELEASE="$GATEA_ARCHIVE_PARENT/$GATEA_ARCHIVE_RELEASE_NAME"
GATEA_ARCHIVE_LOG_ROOT="$MEMAGENT_GATEA_WORK_ROOT/logs/gate_a_2gpu_frozen_20260821r5"
GATEA_ARCHIVE_CERT_ROOT="$GATEA_ARCHIVE_LOG_ROOT/certificates"
GATEA_ARCHIVE_LEDGER="$GATEA_ARCHIVE_LOG_ROOT/gate_a_execution_ledger.jsonl"
GATEA_ARCHIVE_FRESH="$MEMAGENT_GATEA_WORK_ROOT/logs/memory_agent/qwen25_7b_h20_2gpu_gatea_fresh2_strictvllm_naive_indseed_seed2026_20260821r5"
GATEA_ARCHIVE_RESUME="$MEMAGENT_GATEA_WORK_ROOT/logs/memory_agent/qwen25_7b_h20_2gpu_gatea_resume2to3_strictvllm_naive_indseed_seed2026_20260821r5"
GATEA_ARCHIVE_FRESH_LOG="$GATEA_ARCHIVE_LOG_ROOT/qwen25_7b_h20_2gpu_gatea_fresh2_strictvllm_naive_indseed_seed2026_20260821r5.log"
GATEA_ARCHIVE_RESUME_LOG="$GATEA_ARCHIVE_LOG_ROOT/qwen25_7b_h20_2gpu_gatea_resume2to3_strictvllm_naive_indseed_seed2026_20260821r5.log"
GATEA_ARCHIVE_FINAL="$GATEA_ARCHIVE_CERT_ROOT/gate_a_final_report.json"
GATEA_ARCHIVE_PY="$MEMAGENT_GATEA_WORK_ROOT/.venv/bin/python"
GATEA_ARCHIVE_STAGE=
GATEA_ARCHIVE_PUBLISHED=0

cleanup_archive_stage() {
  if [[ $GATEA_ARCHIVE_PUBLISHED -eq 0 && -n $GATEA_ARCHIVE_STAGE && -d $GATEA_ARCHIVE_STAGE ]]; then
    rm -rf -- "$GATEA_ARCHIVE_STAGE"
  fi
}
trap cleanup_archive_stage EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$MEMAGENT_GATEA_EXPECTED_COMMIT" == "$GATEA_ARCHIVE_EXPECTED_COMMIT" ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: expected commit environment does not name r5' >&2; exit 3;
}
[[ -x "$GATEA_ARCHIVE_PY" ]] || {
  echo "GATE_A_ARCHIVE_FAIL: missing Python: $GATEA_ARCHIVE_PY" >&2; exit 4;
}
[[ "$(git -C "$MEMAGENT_GATEA_REPO_DIR" branch --show-current)" == "$GATEA_ARCHIVE_EXPECTED_BRANCH" ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: checkout is not on the frozen r5 branch' >&2; exit 5;
}
[[ "$(git -C "$MEMAGENT_GATEA_REPO_DIR" rev-parse HEAD)" == "$GATEA_ARCHIVE_EXPECTED_COMMIT" ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: checkout is not at the frozen r5 commit' >&2; exit 6;
}
[[ -z "$(git -C "$MEMAGENT_GATEA_REPO_DIR" status --porcelain)" ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: frozen checkout is dirty' >&2; exit 7;
}
git -C "$MEMAGENT_GATEA_REPO_DIR" cat-file -e "$GATEA_ARCHIVE_EXPECTED_COMMIT^{commit}"
[[ ! -e "$GATEA_ARCHIVE_RELEASE" && ! -L "$GATEA_ARCHIVE_RELEASE" ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: archive release already exists; nothing was overwritten' >&2; exit 8;
}

required_sources=(
  "$GATEA_ARCHIVE_CERT_ROOT/p0_preflight.json"
  "$GATEA_ARCHIVE_CERT_ROOT/p0_resolved_manifest.json"
  "$GATEA_ARCHIVE_CERT_ROOT/p1_audit_report.json"
  "$GATEA_ARCHIVE_FINAL"
  "$GATEA_ARCHIVE_LEDGER"
  "$GATEA_ARCHIVE_FRESH/rollout_seed_audit.jsonl"
  "$GATEA_ARCHIVE_RESUME/rollout_seed_audit.jsonl"
  "$GATEA_ARCHIVE_FRESH/latest_checkpointed_iteration.txt"
  "$GATEA_ARCHIVE_RESUME/latest_checkpointed_iteration.txt"
  "$GATEA_ARCHIVE_FRESH_LOG"
  "$GATEA_ARCHIVE_RESUME_LOG"
)
for source_path in "${required_sources[@]}"; do
  [[ -f "$source_path" && ! -L "$source_path" ]] || {
    echo "GATE_A_ARCHIVE_FAIL: source is missing, non-regular, or a symlink: $source_path" >&2; exit 9;
  }
done
[[ "$(<"$GATEA_ARCHIVE_FRESH/latest_checkpointed_iteration.txt")" == 2 ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: fresh checkpoint pointer is not step2' >&2; exit 10;
}
[[ "$(<"$GATEA_ARCHIVE_RESUME/latest_checkpointed_iteration.txt")" == 3 ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: resume checkpoint pointer is not step3' >&2; exit 11;
}

umask 027
mkdir -p "$GATEA_ARCHIVE_PARENT"
[[ -d "$GATEA_ARCHIVE_PARENT" && ! -L "$GATEA_ARCHIVE_PARENT" ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: archive parent is not a real directory' >&2; exit 10;
}
GATEA_ARCHIVE_STAGE=$(mktemp -d "$GATEA_ARCHIVE_PARENT/.gate_a_r5_c3f987be.stage.XXXXXX")
GATEA_ARCHIVE_EVIDENCE="$GATEA_ARCHIVE_STAGE/evidence"
mkdir -p \
  "$GATEA_ARCHIVE_EVIDENCE/certificates" \
  "$GATEA_ARCHIVE_EVIDENCE/logs" \
  "$GATEA_ARCHIVE_EVIDENCE/seeds" \
  "$GATEA_ARCHIVE_EVIDENCE/contract"

copy_verified() {
  local source_path=$1
  local destination_path=$2
  cp -p -- "$source_path" "$destination_path"
  cmp -s -- "$source_path" "$destination_path" || {
    echo "GATE_A_ARCHIVE_FAIL: copied bytes differ: $source_path" >&2; return 1;
  }
}

copy_verified "$GATEA_ARCHIVE_CERT_ROOT/p0_preflight.json" \
  "$GATEA_ARCHIVE_EVIDENCE/certificates/p0_preflight.json"
copy_verified "$GATEA_ARCHIVE_CERT_ROOT/p0_resolved_manifest.json" \
  "$GATEA_ARCHIVE_EVIDENCE/certificates/p0_resolved_manifest.json"
copy_verified "$GATEA_ARCHIVE_CERT_ROOT/p1_audit_report.json" \
  "$GATEA_ARCHIVE_EVIDENCE/certificates/p1_audit_report.json"
copy_verified "$GATEA_ARCHIVE_FINAL" \
  "$GATEA_ARCHIVE_EVIDENCE/certificates/gate_a_final_report.json"
copy_verified "$GATEA_ARCHIVE_LEDGER" \
  "$GATEA_ARCHIVE_EVIDENCE/gate_a_execution_ledger.jsonl"
copy_verified "$GATEA_ARCHIVE_FRESH/rollout_seed_audit.jsonl" \
  "$GATEA_ARCHIVE_EVIDENCE/seeds/fresh.jsonl"
copy_verified "$GATEA_ARCHIVE_RESUME/rollout_seed_audit.jsonl" \
  "$GATEA_ARCHIVE_EVIDENCE/seeds/resume.jsonl"
copy_verified "$GATEA_ARCHIVE_FRESH/latest_checkpointed_iteration.txt" \
  "$GATEA_ARCHIVE_EVIDENCE/fresh_latest.txt"
copy_verified "$GATEA_ARCHIVE_RESUME/latest_checkpointed_iteration.txt" \
  "$GATEA_ARCHIVE_EVIDENCE/resume_latest.txt"
copy_verified "$GATEA_ARCHIVE_FRESH_LOG" \
  "$GATEA_ARCHIVE_EVIDENCE/logs/$(basename "$GATEA_ARCHIVE_FRESH_LOG")"
copy_verified "$GATEA_ARCHIVE_RESUME_LOG" \
  "$GATEA_ARCHIVE_EVIDENCE/logs/$(basename "$GATEA_ARCHIVE_RESUME_LOG")"

GATEA_ARCHIVE_AUDIT_CHECKOUT="$GATEA_ARCHIVE_STAGE/audit_checkout"
GATEA_ARCHIVE_NEW_REAUDIT="$GATEA_ARCHIVE_EVIDENCE/gate_a_r5_readonly_reaudit.json"
git clone --quiet --shared --no-checkout \
  "$MEMAGENT_GATEA_REPO_DIR" "$GATEA_ARCHIVE_AUDIT_CHECKOUT"
git -C "$GATEA_ARCHIVE_AUDIT_CHECKOUT" checkout --quiet --detach \
  "$GATEA_ARCHIVE_EXPECTED_COMMIT"
[[ "$(git -C "$GATEA_ARCHIVE_AUDIT_CHECKOUT" rev-parse HEAD)" == "$GATEA_ARCHIVE_EXPECTED_COMMIT" ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: immutable audit checkout is at the wrong commit' >&2; exit 12;
}
(
  cd "$GATEA_ARCHIVE_AUDIT_CHECKOUT"
  PYTHONDONTWRITEBYTECODE=1 "$GATEA_ARCHIVE_PY" \
    tools/h20/audit_qwen25_7b_gatea.py \
    --manifest manifests/h20/qwen25_7b_gatea_seed2026.yaml \
    --phase final >"$GATEA_ARCHIVE_NEW_REAUDIT"
)

rm -rf -- "$GATEA_ARCHIVE_AUDIT_CHECKOUT"

source_snapshot_pairs=(
  "$GATEA_ARCHIVE_CERT_ROOT/p0_preflight.json|$GATEA_ARCHIVE_EVIDENCE/certificates/p0_preflight.json"
  "$GATEA_ARCHIVE_CERT_ROOT/p0_resolved_manifest.json|$GATEA_ARCHIVE_EVIDENCE/certificates/p0_resolved_manifest.json"
  "$GATEA_ARCHIVE_CERT_ROOT/p1_audit_report.json|$GATEA_ARCHIVE_EVIDENCE/certificates/p1_audit_report.json"
  "$GATEA_ARCHIVE_FINAL|$GATEA_ARCHIVE_EVIDENCE/certificates/gate_a_final_report.json"
  "$GATEA_ARCHIVE_LEDGER|$GATEA_ARCHIVE_EVIDENCE/gate_a_execution_ledger.jsonl"
  "$GATEA_ARCHIVE_FRESH/rollout_seed_audit.jsonl|$GATEA_ARCHIVE_EVIDENCE/seeds/fresh.jsonl"
  "$GATEA_ARCHIVE_RESUME/rollout_seed_audit.jsonl|$GATEA_ARCHIVE_EVIDENCE/seeds/resume.jsonl"
  "$GATEA_ARCHIVE_FRESH/latest_checkpointed_iteration.txt|$GATEA_ARCHIVE_EVIDENCE/fresh_latest.txt"
  "$GATEA_ARCHIVE_RESUME/latest_checkpointed_iteration.txt|$GATEA_ARCHIVE_EVIDENCE/resume_latest.txt"
  "$GATEA_ARCHIVE_FRESH_LOG|$GATEA_ARCHIVE_EVIDENCE/logs/$(basename "$GATEA_ARCHIVE_FRESH_LOG")"
  "$GATEA_ARCHIVE_RESUME_LOG|$GATEA_ARCHIVE_EVIDENCE/logs/$(basename "$GATEA_ARCHIVE_RESUME_LOG")"
)
for source_snapshot_pair in "${source_snapshot_pairs[@]}"; do
  source_path=${source_snapshot_pair%%|*}
  snapshot_path=${source_snapshot_pair#*|}
  cmp -s -- "$source_path" "$snapshot_path" || {
    echo "GATE_A_ARCHIVE_FAIL: source changed during immutable audit: $source_path" >&2; exit 13;
  }
done
[[ "$(<"$GATEA_ARCHIVE_EVIDENCE/fresh_latest.txt")" == 2 ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: archived fresh checkpoint pointer is not step2' >&2; exit 14;
}
[[ "$(<"$GATEA_ARCHIVE_EVIDENCE/resume_latest.txt")" == 3 ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: archived resume checkpoint pointer is not step3' >&2; exit 15;
}

"$GATEA_ARCHIVE_PY" - \
  "$GATEA_ARCHIVE_EVIDENCE/certificates/gate_a_final_report.json" \
  "$GATEA_ARCHIVE_NEW_REAUDIT" \
  "$GATEA_ARCHIVE_EVIDENCE/gate_a_execution_ledger.jsonl" <<'PY'
import hashlib
import json
import sys

final = json.load(open(sys.argv[1], encoding="utf-8"))
reaudit = json.load(open(sys.argv[2], encoding="utf-8"))
ledger_bytes = open(sys.argv[3], "rb").read()
ledger_rows = [json.loads(line) for line in ledger_bytes.splitlines() if line.strip()]
for label, report in (("final", final), ("reaudit", reaudit)):
    assert report.get("status") == "PASS", label
    assert report.get("decision") == "GATE_A_PASS", label
    assert report.get("gates") == {"P0": "PASS", "P1": "PASS", "P2": "PASS"}, label
    assert report.get("ledger_failures") == [], label
    assert all(
        report.get("audits", {}).get(name, {}).get("status") == "PASS"
        for name in ("A1", "A2", "A3", "A4", "A5")
    ), label
for key in ("status", "decision", "gates", "audits", "ledger_failures",
            "step2_inventory", "step3_inventory"):
    assert reaudit.get(key) == final.get(key), key
assert hashlib.sha256(ledger_bytes).hexdigest() == reaudit["ledger_sha256"]
assert len(ledger_rows) == reaudit["ledger_record_count"]
assert ledger_rows and ledger_rows[-1]["record_sha256"] == reaudit["ledger_tail_record_sha256"]
print("GATE_A_ARCHIVE_SNAPSHOT_REPORTS_AND_LEDGER=PASS")
PY

contract_files=(
  manifests/h20/qwen25_7b_gatea_seed2026.yaml
  manifests/h20/qwen25_7b_gatea_commands.json
  gate_a_execution_ledger.schema.json
  scripts/h20/gatea_frozen_common.sh
  scripts/h20/run_qwen25_7b_gatea_fresh2.sh
  scripts/h20/resume_qwen25_7b_gatea_step2_to3.sh
  tools/h20/preflight_qwen25_7b_gatea.py
  tools/h20/audit_qwen25_7b_gatea.py
)
git -C "$MEMAGENT_GATEA_REPO_DIR" archive --format=tar \
  "$GATEA_ARCHIVE_EXPECTED_COMMIT" -- "${contract_files[@]}" \
  | tar -xf - -C "$GATEA_ARCHIVE_EVIDENCE/contract"

GATEA_ARCHIVE_BUNDLE="$GATEA_ARCHIVE_EVIDENCE/repository.bundle"
git -C "$MEMAGENT_GATEA_REPO_DIR" bundle create \
  "$GATEA_ARCHIVE_BUNDLE" "refs/heads/$GATEA_ARCHIVE_EXPECTED_BRANCH"
git -C "$MEMAGENT_GATEA_REPO_DIR" bundle verify "$GATEA_ARCHIVE_BUNDLE" >/dev/null
GATEA_ARCHIVE_BUNDLE_HEADS=$(git bundle list-heads "$GATEA_ARCHIVE_BUNDLE")
[[ "$GATEA_ARCHIVE_BUNDLE_HEADS" == \
  "$GATEA_ARCHIVE_EXPECTED_COMMIT refs/heads/$GATEA_ARCHIVE_EXPECTED_BRANCH" ]] || {
  echo "GATE_A_ARCHIVE_FAIL: bundle head drifted: $GATEA_ARCHIVE_BUNDLE_HEADS" >&2; exit 13;
}
git -C "$MEMAGENT_GATEA_REPO_DIR" show --no-patch --format=fuller \
  "$GATEA_ARCHIVE_EXPECTED_COMMIT" >"$GATEA_ARCHIVE_EVIDENCE/git_commit.txt"

(
  cd "$GATEA_ARCHIVE_EVIDENCE"
  find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS
)

GATEA_ARCHIVE_TARBALL_NAME=gate_a_r5_c3f987be_evidence.tar.gz
GATEA_ARCHIVE_TARBALL_TMP="$GATEA_ARCHIVE_STAGE/.$GATEA_ARCHIVE_TARBALL_NAME.incomplete"
GATEA_ARCHIVE_TARBALL="$GATEA_ARCHIVE_STAGE/$GATEA_ARCHIVE_TARBALL_NAME"
tar -C "$GATEA_ARCHIVE_STAGE" -czf "$GATEA_ARCHIVE_TARBALL_TMP" evidence
gzip -t "$GATEA_ARCHIVE_TARBALL_TMP"
tar -tzf "$GATEA_ARCHIVE_TARBALL_TMP" >/dev/null
mv -- "$GATEA_ARCHIVE_TARBALL_TMP" "$GATEA_ARCHIVE_TARBALL"
(
  cd "$GATEA_ARCHIVE_STAGE"
  sha256sum "$GATEA_ARCHIVE_TARBALL_NAME" >"$GATEA_ARCHIVE_TARBALL_NAME.sha256"
  sha256sum -c "$GATEA_ARCHIVE_TARBALL_NAME.sha256"
)

[[ ! -e "$GATEA_ARCHIVE_RELEASE" && ! -L "$GATEA_ARCHIVE_RELEASE" ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: archive destination appeared during construction' >&2; exit 14;
}
mv -T --no-clobber "$GATEA_ARCHIVE_STAGE" "$GATEA_ARCHIVE_RELEASE"
[[ -d "$GATEA_ARCHIVE_RELEASE" && ! -e "$GATEA_ARCHIVE_STAGE" ]] || {
  echo 'GATE_A_ARCHIVE_FAIL: atomic archive publication did not complete' >&2; exit 15;
}
GATEA_ARCHIVE_PUBLISHED=1

(
  cd "$GATEA_ARCHIVE_RELEASE/evidence"
  sha256sum -c SHA256SUMS >/dev/null
)
(
  cd "$GATEA_ARCHIVE_RELEASE"
  sha256sum -c "$GATEA_ARCHIVE_TARBALL_NAME.sha256" >/dev/null
)

echo "GATE_A_ARCHIVE_PASS: $GATEA_ARCHIVE_RELEASE"
echo "GATE_A_ARCHIVE_SHA256: $(cut -d ' ' -f 1 "$GATEA_ARCHIVE_RELEASE/$GATEA_ARCHIVE_TARBALL_NAME.sha256")"
echo 'Checkpoint shards were not copied; their inventory remains frozen in gate_a_final_report.json.'

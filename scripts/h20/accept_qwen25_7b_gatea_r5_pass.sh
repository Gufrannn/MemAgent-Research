#!/usr/bin/env bash
set -euo pipefail

# Read-only acceptance for the frozen Qwen2.5-7B Gate A r5 run.
# Run this script; do not source it into a shared login shell.

: "${MEMAGENT_GATEA_WORK_ROOT:?set MEMAGENT_GATEA_WORK_ROOT explicitly}"
: "${MEMAGENT_GATEA_REPO_DIR:?set MEMAGENT_GATEA_REPO_DIR explicitly}"
: "${MEMAGENT_GATEA_EXPECTED_COMMIT:?set MEMAGENT_GATEA_EXPECTED_COMMIT explicitly}"

case "$MEMAGENT_GATEA_WORK_ROOT:$MEMAGENT_GATEA_REPO_DIR" in
  /*:/*) ;;
  *) echo 'GATE_A_POST_PASS_FAIL: work root and repo dir must be absolute paths' >&2; exit 2 ;;
esac

GATEA_POST_EXPECTED_COMMIT=c3f987be5513cad2a9e95622dd6773726a7bf12e
GATEA_POST_EXPECTED_BRANCH=h20/qwen25-7b-gatea-2gpu-frozen-20260820
GATEA_POST_PY="$MEMAGENT_GATEA_WORK_ROOT/.venv/bin/python"
GATEA_POST_MANIFEST="$MEMAGENT_GATEA_REPO_DIR/manifests/h20/qwen25_7b_gatea_seed2026.yaml"
GATEA_POST_CERT_ROOT="$MEMAGENT_GATEA_WORK_ROOT/logs/gate_a_2gpu_frozen_20260821r5/certificates"
GATEA_POST_FINAL="$GATEA_POST_CERT_ROOT/gate_a_final_report.json"
GATEA_POST_P0="$GATEA_POST_CERT_ROOT/p0_preflight.json"
GATEA_POST_P1="$GATEA_POST_CERT_ROOT/p1_audit_report.json"
GATEA_POST_LEDGER="$MEMAGENT_GATEA_WORK_ROOT/logs/gate_a_2gpu_frozen_20260821r5/gate_a_execution_ledger.jsonl"

[[ "$MEMAGENT_GATEA_EXPECTED_COMMIT" == "$GATEA_POST_EXPECTED_COMMIT" ]] || {
  echo 'GATE_A_POST_PASS_FAIL: expected commit environment does not name the frozen r5 commit' >&2
  exit 3
}
[[ -x "$GATEA_POST_PY" ]] || { echo "GATE_A_POST_PASS_FAIL: missing Python: $GATEA_POST_PY" >&2; exit 4; }
[[ -f "$GATEA_POST_MANIFEST" ]] || { echo "GATE_A_POST_PASS_FAIL: missing manifest: $GATEA_POST_MANIFEST" >&2; exit 5; }
[[ -f "$GATEA_POST_P0" && -f "$GATEA_POST_P1" && -f "$GATEA_POST_FINAL" && -f "$GATEA_POST_LEDGER" ]] || {
  echo 'GATE_A_POST_PASS_FAIL: P0, P1, final report, or execution ledger is missing' >&2
  exit 6
}

[[ "$(git -C "$MEMAGENT_GATEA_REPO_DIR" branch --show-current)" == "$GATEA_POST_EXPECTED_BRANCH" ]] || {
  echo 'GATE_A_POST_PASS_FAIL: checkout is not on the frozen two-GPU branch' >&2
  exit 7
}
[[ "$(git -C "$MEMAGENT_GATEA_REPO_DIR" rev-parse HEAD)" == "$GATEA_POST_EXPECTED_COMMIT" ]] || {
  echo 'GATE_A_POST_PASS_FAIL: checkout is not at the frozen r5 commit' >&2
  exit 8
}
[[ -z "$(git -C "$MEMAGENT_GATEA_REPO_DIR" status --porcelain)" ]] || {
  echo 'GATE_A_POST_PASS_FAIL: checkout is dirty' >&2
  exit 9
}

GATEA_POST_LIVE_REPORT=$(mktemp)
trap 'rm -f -- "$GATEA_POST_LIVE_REPORT"' EXIT
(
  cd "$MEMAGENT_GATEA_REPO_DIR"
  PYTHONDONTWRITEBYTECODE=1 "$GATEA_POST_PY" tools/h20/audit_qwen25_7b_gatea.py \
    --manifest "$GATEA_POST_MANIFEST" --phase final >"$GATEA_POST_LIVE_REPORT"
)

"$GATEA_POST_PY" - \
  "$GATEA_POST_P0" "$GATEA_POST_P1" "$GATEA_POST_FINAL" \
  "$GATEA_POST_LIVE_REPORT" "$GATEA_POST_LEDGER" "$GATEA_POST_MANIFEST" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

p0_path, p1_path, final_path, live_path, ledger_path, manifest_path = map(
    pathlib.Path, sys.argv[1:]
)


def fail(message: str) -> None:
    raise SystemExit(f"GATE_A_POST_PASS_FAIL: {message}")


def load_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        fail(f"cannot read {path}: {error}")


p0 = load_json(p0_path)
p1 = load_json(p1_path)
final = load_json(final_path)
live = load_json(live_path)
manifest = load_json(manifest_path)
resume_experiment = manifest["experiments"]["resume"]

if p0.get("status") != "PASS" or p0.get("decision") != "P0_PASS":
    fail("P0 certificate is not PASS/P0_PASS")
if p1.get("status") != "PASS" or p1.get("decision") != "P1_AUDIT_PASS":
    fail("P1 certificate is not PASS/P1_AUDIT_PASS")

expected_gates = {"P0": "PASS", "P1": "PASS", "P2": "PASS"}
for label, report in (("frozen final report", final), ("fresh read-only audit", live)):
    if report.get("phase") != "final":
        fail(f"{label} is not a final-phase report")
    if report.get("status") != "PASS" or report.get("decision") != "GATE_A_PASS":
        fail(f"{label} is not PASS/GATE_A_PASS")
    if report.get("gates") != expected_gates:
        fail(f"{label} gate statuses are {report.get('gates')!r}")
    if report.get("ledger_failures") != []:
        fail(f"{label} reports ledger failures: {report.get('ledger_failures')!r}")
    audits = report.get("audits") or {}
    if set(audits) != {"A1", "A2", "A3", "A4", "A5"}:
        fail(f"{label} does not contain exactly A1-A5")
    for audit_name in ("A1", "A2", "A3", "A4", "A5"):
        audit = audits[audit_name]
        if audit.get("status") != "PASS" or audit.get("failures") != []:
            fail(f"{label} {audit_name} is not a clean PASS: {audit!r}")
    if audits["A3"].get("applicable") is not True:
        fail(f"{label} did not apply the explicit-resume audit A3")

if final.get("step2_inventory") != p1.get("step2_inventory"):
    fail("final step2 inventory differs from the P1-frozen inventory")
if final.get("step2_inventory") != live.get("step2_inventory"):
    fail("current step2 files differ from the frozen final report")
if not final.get("step3_inventory") or final.get("step3_inventory") != live.get("step3_inventory"):
    fail("current step3 checkpoint inventory differs from the frozen final report")
for key in ("status", "decision", "gates", "audits", "ledger_failures"):
    if live.get(key) != final.get(key):
        fail(f"fresh read-only audit differs from the frozen final report at {key}")

required_checkpoint_files = {"data.pt"}
for component in ("model", "optim", "extra_state"):
    required_checkpoint_files.update(
        f"actor/{component}_world_size_2_rank_{rank}.pt" for rank in (0, 1)
    )
for inventory_name in ("step2_inventory", "step3_inventory"):
    inventory = final[inventory_name]
    if not required_checkpoint_files.issubset(item.get("path") for item in inventory):
        fail(f"{inventory_name} is missing a rank or checkpoint component")
    if any(
        int(item.get("size", 0)) <= 0
        or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None
        for item in inventory
    ):
        fail(f"{inventory_name} contains an empty file or invalid SHA-256")
step2_data = next(item for item in final["step2_inventory"] if item["path"] == "data.pt")
step3_data = next(item for item in final["step3_inventory"] if item["path"] == "data.pt")
if step2_data["sha256"] == step3_data["sha256"]:
    fail("data cursor did not advance from checkpoint2 to checkpoint3")

raw_lines = ledger_path.read_bytes().splitlines(keepends=True)
frozen_count = final.get("ledger_record_count")
if not isinstance(frozen_count, int) or frozen_count < 1 or len(raw_lines) < frozen_count:
    fail("invalid final-report ledger prefix length")
frozen_prefix_sha = hashlib.sha256(b"".join(raw_lines[:frozen_count])).hexdigest()
if frozen_prefix_sha != final.get("ledger_sha256"):
    fail("append-only ledger prefix changed after final certification")

try:
    ledger = [json.loads(line) for line in raw_lines]
except Exception as error:
    fail(f"execution ledger is not valid JSONL: {error}")
if ledger[frozen_count - 1].get("record_sha256") != final.get("ledger_tail_record_sha256"):
    fail("final-certified ledger tail record changed")

suffix = ledger[frozen_count:]
if [row.get("record_type") for row in suffix] != [
    "checkpoint_inventory", "checkpoint_inventory", "audit_result"
]:
    fail("final report is not followed by exactly the expected three append-only ledger records")
if [row.get("global_step") for row in suffix[:2]] != [2, 3]:
    fail("final checkpoint-inventory ledger suffix is not step2 then step3")
if suffix[0].get("inventory") != final.get("step2_inventory"):
    fail("final step2 ledger inventory disagrees with the final report")
if suffix[1].get("inventory") != final.get("step3_inventory"):
    fail("final step3 ledger inventory disagrees with the final report")
if (
    suffix[2].get("phase") != "final"
    or suffix[2].get("status") != "PASS"
    or suffix[2].get("decision") != "GATE_A_PASS"
):
    fail("final audit-result ledger suffix is not PASS/GATE_A_PASS")
if any(row.get("experiment_name") != resume_experiment for row in suffix):
    fail("final ledger suffix is not bound to the frozen resume experiment")
if pathlib.Path(str(suffix[2].get("report", ""))).resolve() != final_path.resolve():
    fail("final audit-result ledger record is not bound to the exact final report")

resume_rows = [
    row for row in ledger
    if row.get("experiment_name") == resume_experiment
]


def version_acks(version: int, sync_kind: str) -> list[dict]:
    return [
        row for row in resume_rows
        if row.get("record_type") == "weight_sync_ack"
        and row.get("actor_version") == version
        and row.get("sync_kind") == sync_kind
    ]


loaded = version_acks(2, "resume_loaded")
updated = version_acks(3, "post_actor_update")
required_parameters = sorted(manifest["weight_sync"]["parameter_names"])
required_transfer_format = manifest["weight_sync"]["transfer_format"]
required_parameter_count = manifest["weight_sync"]["expected_loaded_parameter_count"]
master_digest_by_version = {}
for label, rows, version in (("resume-loaded", loaded, 2), ("post-update", updated, 3)):
    if sorted(row.get("vllm_worker_rank") for row in rows) != [0, 1]:
        fail(f"{label} actor version {version} does not have exactly worker-rank 0 and 1 acks")
    for row in rows:
        if row.get("vllm_ack_version") != version:
            fail(f"{label} worker ack version does not equal actor version {version}")
        effective_digests = {
            row.get("actor_rollout_sampled_tensor_digest"),
            row.get("actor_sampled_tensor_digest"),
            row.get("vllm_sampled_tensor_digest"),
        }
        if None in effective_digests or len(effective_digests) != 1:
            fail(f"{label} rank {row.get('vllm_worker_rank')} actor/vLLM digests disagree")
        if not row.get("actor_master_sampled_tensor_digest"):
            fail(f"{label} rank {row.get('vllm_worker_rank')} actor master digest is missing")
        if row.get("weight_transfer_format") != required_transfer_format:
            fail(f"{label} rank {row.get('vllm_worker_rank')} weight-transfer format drifted")
        if (
            row.get("loaded_parameter_count") != required_parameter_count
            or row.get("model_parameter_count") != required_parameter_count
        ):
            fail(f"{label} rank {row.get('vllm_worker_rank')} did not cover the full model")
        if row.get("loaded_parameter_names_sha256") != row.get("model_parameter_names_sha256"):
            fail(f"{label} rank {row.get('vllm_worker_rank')} loaded/model name sets disagree")
        if sorted(row.get("audited_loaded_parameters") or []) != required_parameters:
            fail(f"{label} rank {row.get('vllm_worker_rank')} sampled parameter coverage drifted")
        if sorted((row.get("sampled_parameter_dtypes") or {}).keys()) != required_parameters:
            fail(f"{label} rank {row.get('vllm_worker_rank')} sampled dtype coverage drifted")

    master_digests = {row["actor_master_sampled_tensor_digest"] for row in rows}
    if len(master_digests) != 1:
        fail(f"{label} actor master digest is inconsistent across workers")
    master_digest_by_version[version] = next(iter(master_digests))
    if any(row.get("global_step") != version for row in rows):
        fail(f"{label} actor version {version} is recorded at the wrong global step")

loaded_digest = loaded[0]["actor_sampled_tensor_digest"]
updated_digest = updated[0]["actor_sampled_tensor_digest"]
if loaded_digest == updated_digest:
    fail("sampled actor digest did not change from resumed version 2 to updated version 3")
if master_digest_by_version[2] == master_digest_by_version[3]:
    fail("actor master sampled digest did not change from version 2 to version 3")

for rank in (0, 1):
    before = next(row for row in loaded if row["vllm_worker_rank"] == rank)
    after = next(row for row in updated if row["vllm_worker_rank"] == rank)
    if before.get("optimizer_step_max") is None or after.get("optimizer_step_max") is None:
        fail(f"rank {rank} optimizer-step evidence is missing")
    if after["optimizer_step_max"] <= before["optimizer_step_max"]:
        fail(f"rank {rank} optimizer did not advance across explicit resume")
    if after.get("lr_scheduler_last_epoch") is None or before.get("lr_scheduler_last_epoch") is None:
        fail(f"rank {rank} scheduler evidence is missing")
    if after["lr_scheduler_last_epoch"] <= before["lr_scheduler_last_epoch"]:
        fail(f"rank {rank} scheduler did not advance across explicit resume")
    if after.get("optimizer_state_entry_count") != before.get("optimizer_state_entry_count"):
        fail(f"rank {rank} optimizer state entry count changed across resume")
    if after.get("optimizer_step_histogram") == before.get("optimizer_step_histogram"):
        fail(f"rank {rank} optimizer-step histogram did not advance across resume")

summary_rows = [
    row for row in resume_rows
    if row.get("record_type") == "weight_sync_summary"
    and row.get("actor_version") == 3
    and row.get("sync_kind") == "post_actor_update"
]
if len(summary_rows) != 1 or sorted(summary_rows[0].get("worker_ranks") or []) != [0, 1]:
    fail("actor version 3 does not have one complete two-worker sync summary")
if summary_rows[0].get("global_step") != 3:
    fail("actor version 3 sync summary is recorded at the wrong global step")
if summary_rows[0].get("sampled_tensor_digest") != updated_digest:
    fail("actor version 3 sync summary digest disagrees with worker acks")
if summary_rows[0].get("actor_master_sampled_tensor_digest") != updated[0].get(
    "actor_master_sampled_tensor_digest"
):
    fail("actor version 3 sync summary master digest disagrees with worker acks")

step3_signal = [
    row for row in resume_rows
    if row.get("record_type") == "execution_signal"
    and row.get("global_step") == 3
    and row.get("actor_version") == 3
]
if len(step3_signal) != 1 or step3_signal[0].get("nonfinite_metric_names") != []:
    fail("step3 execution signal is absent, duplicated, or non-finite")

resume_load_rows = [row for row in resume_rows if row.get("record_type") == "resume_load"]
rollout_start_rows = [
    row for row in resume_rows
    if row.get("record_type") == "rollout_start"
    and row.get("global_step") == 3
    and row.get("actor_version") == 2
]
loaded_summary_rows = [
    row for row in resume_rows
    if row.get("record_type") == "weight_sync_summary"
    and row.get("sync_kind") == "resume_loaded"
    and row.get("global_step") == 2
    and row.get("actor_version") == 2
]
if len(resume_load_rows) != 1 or len(loaded_summary_rows) != 1 or len(rollout_start_rows) != 1:
    fail("resume-load, version-2 sync summary, or step3 rollout-start evidence is not unique")
semantic_order = [
    resume_load_rows[0]["record_index"],
    max(row["record_index"] for row in loaded),
    loaded_summary_rows[0]["record_index"],
    rollout_start_rows[0]["record_index"],
    max(row["record_index"] for row in updated),
    summary_rows[0]["record_index"],
    step3_signal[0]["record_index"],
    suffix[0]["record_index"],
]
if semantic_order != sorted(semantic_order) or len(set(semantic_order)) != len(semantic_order):
    fail("resume/load/rollout/update/signal/final-inventory records are not in semantic order")

def file_sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

result = {
    "status": "PASS",
    "decision": "GATE_A_PASS_ACCEPTED_READ_ONLY",
    "git_commit": p0["evidence"]["git_commit"],
    "gates": final["gates"],
    "audits": {name: final["audits"][name]["status"] for name in sorted(final["audits"])},
    "version_2_digest": loaded_digest,
    "version_3_digest": updated_digest,
    "version_2_master_digest": master_digest_by_version[2],
    "version_3_master_digest": master_digest_by_version[3],
    "vllm_worker_ranks_acknowledged": [0, 1],
    "step3_checkpoint_files": len(final["step3_inventory"]),
    "final_report_sha256": file_sha(final_path),
    "ledger_current_sha256": file_sha(ledger_path),
}
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo 'GATE_A_POST_PASS_OK: acceptance was read-only; no step4 or new experiment was started'

#!/usr/bin/env python3
"""Raw-artifact-backed B-D/B-E RWWPO-2 confirmatory finalizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.rwwpo2_confirmation import (
    holm_two_test_decisions, one_sided_exact_paired_sign_flip,
    sha256_file, signed_report,
)
from recurrent.research.s128_hotpot_metrics import score_terminal_output, summarize_hotpot_metrics
from recurrent.research.stable_eval_identity import canonical_sha256, validate_resolved_manifest
from tools.h20.preflight_qwen25_7b_stable_i4x2 import _load_parquet_rows


CELLS = ("B", "D", "E")
SEEDS = tuple(range(2026, 2034))


def mapping(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("source mapping malformed")
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-report", action="append", required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--seal-sha256", required=True)
    parser.add_argument("--base-resolved", required=True)
    parser.add_argument("--base-resolved-sha256", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if head != args.expected_commit or dirty:
        raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:checkout")
    if any(Path(value).is_symlink() for value in (
            args.seal, args.base_resolved, args.validation, args.manifest)):
        raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:source symlink")
    if sha256_file(args.seal) != args.seal_sha256 \
            or sha256_file(args.base_resolved) != args.base_resolved_sha256:
        raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:seal/base SHA")
    try:
        seal = signed_report(
            args.seal, decision="RWWPO2_CONFIRMATION_SEAL_PASS", commit=head
        )
    except ValueError as error:
        raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:" + str(error)) from error
    base = validate_resolved_manifest(json.loads(
        Path(args.base_resolved).read_text(encoding="utf-8")
    ))
    if seal.get("resolved_identity_manifest_sha256") != args.base_resolved_sha256 \
            or seal.get("eval_manifest_hash") != base["eval_manifest_hash"] \
            or sha256_file(args.validation) != seal.get("confirmation_data_sha256"):
        raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:sealed identity binding")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if sha256_file(args.manifest) != seal.get("manifest_sha256") \
            or manifest.get("program") != "RWWPO-2" \
            or manifest["training"]["confirmatory_seed_values"] != list(SEEDS):
        raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:manifest")
    expected_assignments = {(cell, seed) for cell in CELLS for seed in SEEDS}
    reports = {}
    for path_value in args.evaluation_report:
        if Path(path_value).is_symlink():
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:evaluation report symlink")
        path = Path(path_value).resolve()
        try:
            report = signed_report(
                path, decision="RWWPO2_CONFIRMATION_EVAL_PASS", commit=head
            )
        except ValueError as error:
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:" + str(error)) from error
        key = (str(report.get("cell")), int(report.get("experiment_seed", -1)))
        if key not in expected_assignments or key in reports:
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:assignment coverage")
        reports[key] = {**report, "path": str(path), "file_sha256": sha256_file(path)}
    if set(reports) != expected_assignments:
        raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:missing assignments")

    raw_rows = _load_parquet_rows(Path(args.validation).resolve())
    frozen = base["identity_payload"]["rows"]
    examples = len(frozen)
    truth = {}
    for row in frozen:
        reward = mapping(raw_rows[int(row["raw_row_position"])]["reward_model"])
        ground_truth = reward.get("ground_truth")
        if canonical_sha256(ground_truth) != row["ground_truth_hash"]:
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:ground truth hash")
        truth[int(row["source_order_index"])] = ground_truth

    protocols = set()
    stable_inventories = set()
    aggregates = {}
    receipt_inventory = {}
    for key, report in sorted(reports.items()):
        raw_resolved_path = Path(report["resolved_manifest_path"])
        if raw_resolved_path.is_symlink():
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:eval resolved symlink")
        resolved_path = raw_resolved_path.resolve()
        if sha256_file(resolved_path) != report["resolved_manifest_sha256"]:
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:eval resolved SHA")
        resolved = validate_resolved_manifest(json.loads(resolved_path.read_text()))
        if resolved["identity_payload"] != base["identity_payload"] \
                or resolved["eval_manifest_hash"] != base["eval_manifest_hash"]:
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:stable identity drift")
        execution = resolved.get("execution_binding", {})
        trainer = execution.get("trainer_configuration", {})
        if execution.get("cell") != key[0] \
                or int(execution.get("experiment_seed", -1)) != key[1] \
                or execution.get("interface_id") != report.get("interface_id") \
                or trainer.get("generation_protocol_sha256") != report.get(
                    "generation_protocol_sha256") \
                or canonical_sha256(trainer.get("generation_protocol", {})) != report.get(
                    "generation_protocol_sha256") \
                or report.get("confirmation_data_sha256") != seal.get(
                    "confirmation_data_sha256"):
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:execution/protocol binding")
        raw_training_audit = Path(execution.get("training_attempt_audit_path", ""))
        if raw_training_audit.is_symlink():
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:training audit symlink")
        training_audit_path = raw_training_audit.resolve()
        if not training_audit_path.is_file() \
                or sha256_file(training_audit_path) != execution.get(
                    "training_attempt_audit_sha256"):
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:training audit bytes")
        try:
            training_audit = signed_report(
                training_audit_path,
                decision="RWWPO2_R400_ATTEMPT_AUDIT_PASS", commit=head,
            )
        except ValueError as error:
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:" + str(error)) from error
        if training_audit.get("cell") != key[0] \
                or int(training_audit.get("experiment_seed", -1)) != key[1] \
                or training_audit.get("report_sha256") != execution.get(
                    "training_attempt_audit_report_sha256") \
                or training_audit.get("source_manifest_sha256") != seal.get(
                    "manifest_sha256") \
                or training_audit.get("confirmation_seal_file_sha256") != \
                    args.seal_sha256 \
                or training_audit.get("confirmation_seal_report_sha256") != \
                    seal.get("report_sha256") \
                or training_audit.get("s128_consumed") is not False \
                or training_audit.get("performance_evaluated") is not False:
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:training audit identity")
        terminal_path = Path(report["eval_root"]) / "terminal/400.jsonl"
        metric_path = Path(report["metric_rows_path"]).resolve()
        if terminal_path.is_symlink() or metric_path.is_symlink():
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:symlink raw artifact")
        if sha256_file(terminal_path) != report["terminal_sha256"] \
                or sha256_file(metric_path) != report["metric_rows_sha256"]:
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:raw artifact SHA")
        terminal = read_jsonl(terminal_path)
        if len(terminal) != examples \
                or [int(row.get("source_order_index", -1)) for row in terminal] != list(
                    range(examples)):
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:terminal denominator/order")
        recomputed_rows = []
        for source in terminal:
            order = int(source["source_order_index"])
            expected = frozen[order]
            if str(source.get("example_id")) != str(expected["example_id"]) \
                    or source.get("eval_manifest_hash") != base["eval_manifest_hash"] \
                    or not isinstance(source.get("output"), str):
                raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:terminal stable join")
            scored = score_terminal_output(source["output"], truth[order])
            recomputed_rows.append({
                "stable_key": json.dumps([
                    base["eval_manifest_hash"], str(expected["example_id"]), 0
                ], separators=(",", ":")),
                "source_order_index": order,
                "example_id": str(expected["example_id"]),
                **{name: scored[name] for name in (
                    "exact_match", "token_f1", "precision", "recall",
                    "format_success", "sub_exact_match", "extraction_route",
                )},
            })
        recorded_rows = read_jsonl(metric_path)
        if recorded_rows != recomputed_rows \
                or canonical_sha256(recorded_rows) != report[
                    "metric_rows_canonical_sha256"]:
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:metric row reconstruction")
        stable_inventory_sha = canonical_sha256(
            [row["stable_key"] for row in recomputed_rows]
        )
        if stable_inventory_sha != report.get("stable_key_inventory_sha256"):
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:stable-key inventory")
        aggregate = summarize_hotpot_metrics(
            recomputed_rows, expected_denominator=examples
        )
        if aggregate != report["metrics"]:
            raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:aggregate reconstruction")
        protocols.add(report["generation_protocol_sha256"])
        stable_inventories.add(report["stable_key_inventory_sha256"])
        aggregates[f"{key[0]}:seed{key[1]}"] = aggregate
        receipt_inventory[f"{key[0]}:seed{key[1]}"] = {
            "path": report["path"], "file_sha256": report["file_sha256"],
            "report_sha256": report["report_sha256"],
            "terminal_sha256": report["terminal_sha256"],
            "metric_rows_sha256": report["metric_rows_sha256"],
        }
    if len(protocols) != 1 or len(stable_inventories) != 1:
        raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:protocol/stable-key drift")

    minimums = manifest["performance"]["minimum_effects"]
    contrast_results = {}
    pvalues = {}
    for contrast, comparator in (("B-D", "D"), ("B-E", "E")):
        differences = [
            aggregates[f"B:seed{seed}"]["token_f1"]
            - aggregates[f"{comparator}:seed{seed}"]["token_f1"]
            for seed in SEEDS
        ]
        margin = float(minimums[contrast])
        centered = [value - margin for value in differences]
        pvalue = one_sided_exact_paired_sign_flip(centered)
        pvalues[contrast] = pvalue
        contrast_results[contrast] = {
            "paired_seed_differences": differences,
            "mean_difference": sum(differences) / 8,
            "minimum_effect_margin": margin,
            "margin_centered_differences": centered,
            "margin_centered_mean": sum(centered) / 8,
            "one_sided_exact_paired_sign_flip_p": pvalue,
            "seed_unit": "paired independent training seed",
            "inference_assumption": manifest["performance"]["sign_flip_assumption"],
        }
    holm = holm_two_test_decisions(pvalues, alpha=0.05)
    passes = all(item["reject"] for item in holm.values()) and all(
        contrast_results[name]["mean_difference"] > contrast_results[name][
            "minimum_effect_margin"] for name in ("B-D", "B-E")
    )
    for name in contrast_results:
        contrast_results[name]["holm"] = holm[name]
        contrast_results[name]["passes_effect_and_fwer"] = bool(
            holm[name]["reject"] and contrast_results[name]["mean_difference"]
            > contrast_results[name]["minimum_effect_margin"]
        )
    report = {
        "schema_version": "rwwpo2-confirmation-final-v1",
        "status": "PASS" if passes else "NO_GO",
        "decision": (
            "RWWPO2_CONFIRMATORY_CO_PRIMARY_PASS" if passes
            else "RWWPO2_CONFIRMATORY_CO_PRIMARY_NO_GO"
        ),
        "git_commit": head,
        "seal_sha256": args.seal_sha256,
        "base_resolved_sha256": args.base_resolved_sha256,
        "eval_manifest_hash": base["eval_manifest_hash"],
        "confirmation_data_sha256": seal["confirmation_data_sha256"],
        "denominator": examples,
        "generation_protocol_sha256": next(iter(protocols)),
        "stable_key_inventory_sha256": next(iter(stable_inventories)),
        "co_primary_contrasts": contrast_results,
        "family_wise_procedure": (
            "margin-centered one-sided exact paired sign flip; Holm alpha=0.05"
        ),
        "seed_aggregates": aggregates,
        "evaluation_receipts": receipt_inventory,
        "claim_scope": "medium-budget R400 on one sealed confirmation population; not convergence",
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["report_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise SystemExit("RWWPO2_CONFIRM_FINAL_NO_GO:append-only output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "decision": report["decision"],
        "co_primary_contrasts": contrast_results, "output": str(output.resolve()),
    }, sort_keys=True))
    if not passes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

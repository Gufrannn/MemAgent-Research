#!/usr/bin/env python3
"""Generate P1 or final P0/P1/P2 + A1-A5 Gate A certificates without rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recurrent.research.gate_a_execution import (
    checkpoint_inventory,
    load_frozen_manifest,
    sha256_file,
    validate_jsonl_chain,
)
from recurrent.research.trajectory_seeding import build_trajectory_seed_records, derive_turn_request_seeds


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl_records_sha256(records: list[dict]) -> str:
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_type_matches(value: object, expected: str) -> bool:
    return {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)


def validate_ledger_schema(records: list[dict], schema: dict) -> list[str]:
    """Validate the concrete schema subset used by the frozen ledger without a new dependency."""
    failures: list[str] = []
    properties = schema.get("properties", {})
    base_required = set(schema.get("required", []))
    record_types = set(properties.get("record_type", {}).get("enum", []))
    for index, record in enumerate(records):
        required = set(base_required)
        record_type = record.get("record_type")
        if record_type not in record_types:
            failures.append(f"ledger record {index} has unknown record_type {record_type!r}")
        for conditional in schema.get("allOf", []):
            expected_type = (
                conditional.get("if", {}).get("properties", {}).get("record_type", {}).get("const")
            )
            if record_type == expected_type:
                required.update(conditional.get("then", {}).get("required", []))
        missing = sorted(required - record.keys())
        if missing:
            failures.append(f"ledger record {index} missing schema fields {missing}")
        for name, value in record.items():
            rule = properties.get(name)
            if not rule:
                continue
            allowed_types = rule.get("type")
            if isinstance(allowed_types, str):
                allowed_types = [allowed_types]
            if allowed_types and not any(_json_type_matches(value, item) for item in allowed_types):
                failures.append(f"ledger record {index} field {name} has invalid type")
                continue
            if "enum" in rule and value not in rule["enum"]:
                failures.append(f"ledger record {index} field {name} is outside its enum")
            if isinstance(value, str):
                if len(value) < int(rule.get("minLength", 0)):
                    failures.append(f"ledger record {index} field {name} is too short")
                if rule.get("pattern") and re.fullmatch(rule["pattern"], value) is None:
                    failures.append(f"ledger record {index} field {name} does not match its pattern")
                if rule.get("format") == "date-time":
                    try:
                        datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except ValueError:
                        failures.append(f"ledger record {index} field {name} is not date-time")
            if isinstance(value, (int, float)) and "minimum" in rule and value < rule["minimum"]:
                failures.append(f"ledger record {index} field {name} is below its minimum")
            if isinstance(value, list):
                if len(value) < int(rule.get("minItems", 0)):
                    failures.append(f"ledger record {index} field {name} has too few items")
                if rule.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
                    failures.append(f"ledger record {index} field {name} has duplicate items")
            if isinstance(value, dict) and len(value) < int(rule.get("minProperties", 0)):
                failures.append(f"ledger record {index} field {name} has too few properties")
            if isinstance(value, dict) and isinstance(rule.get("additionalProperties"), dict):
                child_rule = rule["additionalProperties"]
                child_types = child_rule.get("type")
                if isinstance(child_types, str):
                    child_types = [child_types]
                for child_name, child_value in value.items():
                    if child_types and not any(
                        _json_type_matches(child_value, child_type) for child_type in child_types
                    ):
                        failures.append(
                            f"ledger record {index} field {name}.{child_name} has invalid type"
                        )
                    if (
                        isinstance(child_value, (int, float))
                        and "minimum" in child_rule
                        and child_value < child_rule["minimum"]
                    ):
                        failures.append(
                            f"ledger record {index} field {name}.{child_name} is below its minimum"
                        )
    return failures


def component_inventory(step_dir: Path, expected_world_size: int) -> tuple[list[dict], list[str]]:
    inventory = checkpoint_inventory(step_dir) if step_dir.is_dir() else []
    names = [item["path"] for item in inventory if item["size"] > 0]
    missing = []
    component_prefixes = {
        "model": "model",
        "optim": "optim",
        "extra": "extra_state",
    }
    expected_ranks = list(range(expected_world_size))
    for component, prefix in component_prefixes.items():
        pattern = re.compile(
            rf"^actor/{prefix}_world_size_{expected_world_size}_rank_(\d+)\.pt$"
        )
        ranks = sorted(int(match.group(1)) for name in names if (match := pattern.match(name)))
        if ranks != expected_ranks:
            missing.append(f"{component}_ranks_{ranks}")
    if "data.pt" not in names:
        missing.append("data")
    return inventory, missing


def audit_seeds(
    seed_records: list[dict],
    seed: int,
    rollout_n: int,
    *,
    expected_steps: list[int],
    expected_batch_size: int,
    expected_dataset_cursor: Sequence[int],
) -> tuple[bool, list[str]]:
    failures = []
    if not seed_records:
        return False, ["missing rollout seed records"]
    base_records = [row for row in seed_records if row.get("record_type", "trajectory_seed") == "trajectory_seed"]
    turn_records = [row for row in seed_records if row.get("record_type") == "trajectory_turn_seed"]
    if not base_records:
        return False, ["missing base trajectory seed records"]
    if not turn_records:
        failures.append("missing recurrent turn seed records")
    expected_step_set = set(expected_steps)
    base_step_set = {int(row.get("global_step", -1)) for row in base_records}
    turn_step_set = {int(row.get("global_step", -1)) for row in turn_records}
    if base_step_set != expected_step_set:
        failures.append(
            f"base trajectory step coverage {sorted(base_step_set)} != {sorted(expected_step_set)}"
        )
    if turn_step_set != expected_step_set:
        failures.append(
            f"turn trajectory step coverage {sorted(turn_step_set)} != {sorted(expected_step_set)}"
        )
    for step in sorted(expected_step_set):
        rows = [row for row in base_records if int(row.get("global_step", -1)) == step]
        if len(rows) != expected_batch_size:
            failures.append(
                f"step {step} base trajectory count {len(rows)} != {expected_batch_size}"
            )
        required = {
            "row", "group", "replica", "global_step", "trajectory_seed", "uid",
            "dataset_index", "mode",
        }
        for index, row in enumerate(rows):
            missing = sorted(required - row.keys())
            if missing:
                failures.append(f"seed row {index} at step {step} missing identity fields {missing}")
        actual_rows = sorted(int(row.get("row", -1)) for row in rows)
        expected_rows = list(range(expected_batch_size))
        if actual_rows != expected_rows:
            failures.append(f"step {step} trajectory rows {actual_rows} != {expected_rows}")
        identity_keys = [(row.get("uid"), int(row.get("replica", -1)), int(row.get("row", -1))) for row in rows]
        if len(identity_keys) != len(set(identity_keys)):
            failures.append(f"trajectory identity collision at step {step}")
        rows_by_index = {int(row.get("row", -1)): row for row in rows}
        actual = [
            int(rows_by_index[row]["trajectory_seed"])
            for row in expected_rows
            if row in rows_by_index
        ]
        if len(actual) != len(set(actual)):
            failures.append(f"trajectory seed collision at step {step}")
        expected = build_trajectory_seed_records(
            base_seed=seed,
            global_step=step,
            batch_size=expected_batch_size,
            rollout_n=rollout_n,
            mode="independent",
        )
        if len(actual) != expected_batch_size or actual != [int(row["trajectory_seed"]) for row in expected]:
            failures.append(f"seed schedule is not reconstructable at step {step}")

        base_by_row = rows_by_index
        group_uids: dict[int, set[str]] = defaultdict(set)
        group_dataset_indices: dict[int, set[int]] = defaultdict(set)
        uid_groups: dict[str, set[int]] = defaultdict(set)
        for source_row, row in base_by_row.items():
            expected_group, expected_replica = divmod(source_row, rollout_n)
            actual_group = int(row.get("group", -1))
            actual_replica = int(row.get("replica", -1))
            if (actual_group, actual_replica) != (expected_group, expected_replica):
                failures.append(
                    f"step {step} row {source_row} group/replica "
                    f"{(actual_group, actual_replica)} != {(expected_group, expected_replica)}"
                )
            if row.get("mode") != "independent":
                failures.append(f"step {step} row {source_row} is not independently seeded")
            uid = str(row.get("uid", ""))
            group_uids[actual_group].add(uid)
            uid_groups[uid].add(actual_group)
            group_dataset_indices[actual_group].add(int(row.get("dataset_index", -1)))
        for group, uids in sorted(group_uids.items()):
            if len(uids) != 1:
                failures.append(f"step {step} group {group} does not share one prompt uid: {sorted(uids)}")
        for uid, groups in uid_groups.items():
            if len(groups) != 1:
                failures.append(f"step {step} prompt uid {uid!r} is reused across groups {sorted(groups)}")
        groups_per_step = expected_batch_size // rollout_n
        for group, indices in sorted(group_dataset_indices.items()):
            cursor_offset = (step - 1) * groups_per_step + group
            if cursor_offset < 0 or cursor_offset >= len(expected_dataset_cursor):
                failures.append(
                    f"step {step} group {group} cursor offset {cursor_offset} is outside "
                    f"the frozen prefix of length {len(expected_dataset_cursor)}"
                )
                continue
            expected_index = int(expected_dataset_cursor[cursor_offset])
            if indices != {expected_index}:
                failures.append(
                    f"step {step} group {group} semantic dataset cursor "
                    f"{sorted(indices)} != frozen semantic index {expected_index} "
                    f"at cursor position {cursor_offset}"
                )
        step_turns = [row for row in turn_records if int(row.get("global_step", -1)) == step]
        turn_keys = [
            (int(row.get("sample_index", -1)), int(row.get("turn", -1)))
            for row in step_turns
        ]
        if len(turn_keys) != len(set(turn_keys)):
            failures.append(f"trajectory turn identity collision at step {step}")
        active_turns_by_row: dict[int, list[int]] = defaultdict(list)
        final_turns_by_row: dict[int, list[int]] = defaultdict(list)
        for row in step_turns:
            source_row = int(row.get("sample_index", -1))
            base = base_by_row.get(source_row)
            if base is None:
                failures.append(f"turn record references unknown sample_index {source_row} at step {step}")
                continue
            if (
                row.get("uid") != base.get("uid")
                or int(row.get("group", -1)) != int(base["group"])
                or int(row.get("replica", -1)) != int(base["replica"])
                or int(row.get("trajectory_seed", -1)) != int(base["trajectory_seed"])
                or int(row.get("dataset_index", -1)) != int(base["dataset_index"])
                or row.get("mode") != "independent"
            ):
                failures.append(f"turn/base trajectory identity mismatch at step {step}, row {source_row}")
            turn = int(row.get("turn", -1))
            if turn < 0:
                failures.append(f"invalid recurrent turn at step {step}, row {source_row}: {turn}")
                continue
            is_final = row.get("is_final")
            if not isinstance(is_final, bool):
                failures.append(
                    f"turn record is missing boolean final identity at step {step}, "
                    f"row {source_row}, turn {turn}"
                )
            elif is_final:
                final_turns_by_row[source_row].append(turn)
            else:
                active_turns_by_row[source_row].append(turn)
            expected_request_seed = derive_turn_request_seeds(
                [int(base["trajectory_seed"])], [0], turn
            )[0]
            if int(row.get("request_seed", -1)) != expected_request_seed:
                failures.append(f"request seed is not reconstructable at step {step}, row {source_row}, turn {turn}")
        common_final_turns = set()
        for source_row in base_by_row:
            active_turns = sorted(active_turns_by_row[source_row])
            final_turns = sorted(final_turns_by_row[source_row])
            if active_turns != list(range(len(active_turns))):
                failures.append(
                    f"active trajectory turns are missing or non-contiguous at step {step}, "
                    f"row {source_row}: {active_turns}"
                )
            if len(final_turns) != 1:
                failures.append(
                    f"trajectory must have exactly one final turn at step {step}, "
                    f"row {source_row}: {final_turns}"
                )
                continue
            final_turn = final_turns[0]
            common_final_turns.add(final_turn)
            if active_turns and final_turn <= active_turns[-1]:
                failures.append(
                    f"trajectory final turn does not follow active turns at step {step}, "
                    f"row {source_row}: active={active_turns}, final={final_turn}"
                )
        if len(common_final_turns) != 1:
            failures.append(
                f"memory trajectories do not share one global final turn at step {step}: "
                f"{sorted(common_final_turns)}"
            )
    return not failures, failures


def audit_sync(
    records: list[dict],
    required_versions: list[int],
    ranks: list[int],
    required_syncs: list[tuple[str, int, str]] | None = None,
    required_parameters: list[str] | None = None,
    required_transfer_format: str | None = None,
    expected_loaded_parameter_count: int | None = None,
) -> tuple[bool, list[str], dict[int, str]]:
    failures = []
    digests_by_version: dict[int, set[str]] = defaultdict(set)
    master_digests_by_version: dict[int, set[str]] = defaultdict(set)
    loaded_name_digests: set[str] = set()
    syncs = required_syncs or [("", version, "") for version in required_versions]
    for experiment, version, sync_kind in syncs:
        acks = [
            row for row in records
            if row.get("record_type") == "weight_sync_ack"
            and int(row.get("actor_version", -1)) == version
            and (not experiment or row.get("experiment_name") == experiment)
            and (not sync_kind or row.get("sync_kind") == sync_kind)
        ]
        actual_ranks = sorted({int(row["vllm_worker_rank"]) for row in acks})
        context = f"{experiment or '*'}:{version}:{sync_kind or '*'}"
        if actual_ranks != ranks:
            failures.append(f"sync {context} ack ranks {actual_ranks} != {ranks}")
        if len(acks) != len(ranks):
            failures.append(f"sync {context} expected {len(ranks)} unique acks, found {len(acks)}")
        for row in acks:
            actor_rollout_digest = row.get("actor_rollout_sampled_tensor_digest")
            if row.get("actor_sampled_tensor_digest") != actor_rollout_digest:
                failures.append(
                    f"actor effective-digest alias mismatch at sync {context}, "
                    f"rank {row['vllm_worker_rank']}"
                )
            if actor_rollout_digest != row["vllm_sampled_tensor_digest"]:
                failures.append(
                    f"effective actor-rollout/vLLM digest mismatch at sync {context}, "
                    f"rank {row['vllm_worker_rank']}"
                )
            master_digest = row.get("actor_master_sampled_tensor_digest")
            if not master_digest:
                failures.append(f"missing actor master digest at sync {context}, rank {row['vllm_worker_rank']}")
            else:
                master_digests_by_version[version].add(master_digest)
            if required_parameters is not None:
                audited_loaded = sorted(row.get("audited_loaded_parameters") or [])
                if audited_loaded != sorted(required_parameters):
                    failures.append(
                        f"sampled load coverage mismatch at sync {context}, rank "
                        f"{row['vllm_worker_rank']}: {audited_loaded} != {sorted(required_parameters)}"
                    )
                if int(row.get("loaded_parameter_count", -1)) < len(required_parameters):
                    failures.append(
                        f"loaded parameter count is too small at sync {context}, "
                        f"rank {row['vllm_worker_rank']}: {row.get('loaded_parameter_count')}"
                    )
                sampled_dtypes = row.get("sampled_parameter_dtypes") or {}
                if sorted(sampled_dtypes) != sorted(required_parameters):
                    failures.append(
                        f"sampled dtype coverage mismatch at sync {context}, "
                        f"rank {row['vllm_worker_rank']}"
                    )
            if expected_loaded_parameter_count is not None:
                loaded_count = int(row.get("loaded_parameter_count", -1))
                model_count = int(row.get("model_parameter_count", -1))
                if loaded_count != expected_loaded_parameter_count or model_count != expected_loaded_parameter_count:
                    failures.append(
                        f"full load count mismatch at sync {context}, rank {row['vllm_worker_rank']}: "
                        f"loaded={loaded_count}, model={model_count}, "
                        f"expected={expected_loaded_parameter_count}"
                    )
                loaded_names_digest = row.get("loaded_parameter_names_sha256")
                model_names_digest = row.get("model_parameter_names_sha256")
                if not loaded_names_digest or loaded_names_digest != model_names_digest:
                    failures.append(
                        f"full load name-set digest mismatch at sync {context}, "
                        f"rank {row['vllm_worker_rank']}"
                    )
                else:
                    loaded_name_digests.add(loaded_names_digest)
            if required_transfer_format is not None and row.get("weight_transfer_format") != required_transfer_format:
                failures.append(
                    f"weight transfer format mismatch at sync {context}, rank "
                    f"{row['vllm_worker_rank']}: {row.get('weight_transfer_format')} "
                    f"!= {required_transfer_format}"
                )
            if int(row["vllm_ack_version"]) != version:
                failures.append(f"ack version mismatch at actor version {version}")
            if actor_rollout_digest:
                digests_by_version[version].add(actor_rollout_digest)
        if len(digests_by_version[version]) != 1:
            failures.append(f"sync {context} has split or missing actor digests")
        if len(master_digests_by_version[version]) != 1:
            failures.append(f"sync {context} has split or missing actor master digests")
        summaries = [
            row for row in records
            if row.get("record_type") == "weight_sync_summary"
            and int(row.get("actor_version", -1)) == version
            and (not experiment or row.get("experiment_name") == experiment)
            and (not sync_kind or row.get("sync_kind") == sync_kind)
        ]
        if len(summaries) != 1:
            failures.append(f"sync {context} expected one driver summary, found {len(summaries)}")
        else:
            summary = summaries[0]
            if sorted(summary.get("worker_ranks") or []) != ranks:
                failures.append(f"sync {context} summary worker ranks do not match {ranks}")
            if (
                summary.get("sampled_tensor_digest") not in digests_by_version[version]
                or summary.get("actor_master_sampled_tensor_digest")
                not in master_digests_by_version[version]
            ):
                failures.append(f"sync {context} driver summary digest does not match worker acks")
    if expected_loaded_parameter_count is not None and len(loaded_name_digests) != 1:
        failures.append("vLLM full parameter name-set changed across workers or actor versions")
    collapsed = {version: next(iter(values)) for version, values in digests_by_version.items() if len(values) == 1}
    return not failures, failures, collapsed


def build_report(manifest: dict, phase: str) -> tuple[dict, list[dict]]:
    paths = manifest["paths"]
    ledger_path = Path(paths["execution_ledger"])
    ledger = read_jsonl(ledger_path)
    ledger_schema = json.loads((REPO_ROOT / manifest["ledger_schema"]).read_text(encoding="utf-8"))
    ledger_failures = validate_jsonl_chain(ledger) + validate_ledger_schema(ledger, ledger_schema)
    fresh_experiment = manifest["experiments"]["fresh"]
    resume_experiment = manifest["experiments"]["resume"]
    accepted_experiments = {fresh_experiment} | ({resume_experiment} if phase == "final" else set())
    execution_records = [row for row in ledger if row.get("experiment_name") in accepted_experiments]
    fresh_dir = Path(paths["fresh_output"])
    resume_dir = Path(paths["resume_output"])
    fresh_seeds = read_jsonl(fresh_dir / "rollout_seed_audit.jsonl")
    resume_seeds = read_jsonl(resume_dir / "rollout_seed_audit.jsonl")
    expected_trajectory_count = int(manifest["training"]["train_batch_size"]) * int(
        manifest["training"]["rollout_n"]
    )
    fresh_a1_ok, fresh_a1_failures = audit_seeds(
        fresh_seeds,
        manifest["training"]["seed"],
        manifest["training"]["rollout_n"],
        expected_steps=[1, 2],
        expected_batch_size=expected_trajectory_count,
        expected_dataset_cursor=manifest["data"]["train_cursor_prefix"],
    )
    a1_ok = fresh_a1_ok
    a1_failures = list(fresh_a1_failures)
    if phase == "final":
        resume_a1_ok, resume_a1_failures = audit_seeds(
            resume_seeds,
            manifest["training"]["seed"],
            manifest["training"]["rollout_n"],
            expected_steps=[3],
            expected_batch_size=expected_trajectory_count,
            expected_dataset_cursor=manifest["data"]["train_cursor_prefix"],
        )
        a1_ok = a1_ok and resume_a1_ok
        a1_failures.extend(f"resume: {failure}" for failure in resume_a1_failures)

    expected_world_size = int(manifest["gpu"]["world_size"])
    step2_inventory, step2_missing = component_inventory(
        fresh_dir / "global_step_2", expected_world_size
    )
    step3_inventory, step3_missing = (
        component_inventory(resume_dir / "global_step_3", expected_world_size)
        if phase == "final" else ([], [])
    )
    a2_failures = [f"step2 missing {item}" for item in step2_missing]
    if phase == "final":
        a2_failures.extend(f"step3 missing {item}" for item in step3_missing)
        p1_report_path = Path(paths["certificate_root"]) / "p1_audit_report.json"
        if not p1_report_path.is_file():
            a2_failures.append("missing immutable P1 audit report")
        else:
            p1_report = json.loads(p1_report_path.read_text())
            frozen_step2 = p1_report.get("step2_inventory", [])
            if frozen_step2 != step2_inventory:
                a2_failures.append("step2 checkpoint inventory changed after P1 certification")
            frozen_count = p1_report.get("ledger_record_count")
            if not isinstance(frozen_count, int) or frozen_count < 1 or frozen_count > len(ledger):
                a2_failures.append(f"invalid P1 ledger prefix length: {frozen_count}")
            else:
                frozen_prefix = ledger[:frozen_count]
                if jsonl_records_sha256(frozen_prefix) != p1_report.get("ledger_sha256"):
                    a2_failures.append("execution ledger P1 prefix changed after certification")
                if frozen_prefix[-1].get("record_sha256") != p1_report.get("ledger_tail_record_sha256"):
                    a2_failures.append("execution ledger P1 tail changed after certification")
            p1_inventory_records = [
                row for row in ledger
                if row.get("record_type") == "checkpoint_inventory"
                and row.get("experiment_name") == fresh_experiment
                and int(row.get("global_step", -1)) == 2
            ]
            if len(p1_inventory_records) != 1 or p1_inventory_records[0].get("inventory") != frozen_step2:
                a2_failures.append("ledger does not contain exactly one matching P1 checkpoint inventory")
            p1_audit_records = [
                row for row in ledger
                if row.get("record_type") == "audit_result"
                and row.get("experiment_name") == fresh_experiment
                and row.get("phase") == "p1"
                and row.get("status") == "PASS"
            ]
            if len(p1_audit_records) != 1:
                a2_failures.append("ledger does not contain exactly one P1 PASS audit result")

    a3_failures = []
    if phase == "final":
        loads = [
            row for row in execution_records
            if row.get("record_type") == "resume_load" and row.get("experiment_name") == resume_experiment
        ]
        expected_source = str(Path(paths["resume_source"]).resolve())
        if len(loads) != 1:
            a3_failures.append(f"expected exactly one resume_load record, found {len(loads)}")
        else:
            load = loads[0]
            if load.get("resume_source") != expected_source or int(load.get("global_step", -1)) != 2:
                a3_failures.append(f"resume source/global step drift: {load}")
            load_acks = load.get("actor_load_worker_acks") or []
            expected_ranks = manifest["weight_sync"]["required_worker_ranks"]
            actual_ranks = sorted(int(ack.get("rank", -1)) for ack in load_acks)
            if actual_ranks != expected_ranks:
                a3_failures.append(f"resume actor load ack ranks {actual_ranks} != {expected_ranks}")
            for ack in load_acks:
                if not all(ack.get(key) for key in ("model_loaded", "optimizer_loaded", "extra_loaded")):
                    a3_failures.append(f"incomplete actor checkpoint load ack: {ack}")
                state_count = int(ack.get("optimizer_state_entry_count", 0))
                step_count = int(ack.get("optimizer_step_entry_count", 0))
                histogram = ack.get("optimizer_step_histogram") or {}
                valid_histogram = isinstance(histogram, dict) and all(
                    isinstance(value, int) and value > 0 for value in histogram.values()
                )
                if (
                    state_count < 1
                    or step_count != state_count
                    or not valid_histogram
                    or sum(histogram.values()) != step_count
                ):
                    a3_failures.append(f"incomplete loaded optimizer state evidence: {ack}")
                if ack.get("optimizer_step_max") is None or int(ack["optimizer_step_max"]) < 1:
                    a3_failures.append(f"loaded optimizer never advanced before resume: {ack}")
                if ack.get("lr_scheduler_last_epoch") is None or int(ack["lr_scheduler_last_epoch"]) < 1:
                    a3_failures.append(f"missing/non-positive loaded scheduler epoch: {ack}")
            data_item = next((item for item in step2_inventory if item["path"] == "data.pt"), None)
            if not data_item or load.get("data_sha256") != data_item["sha256"]:
                a3_failures.append("loaded data cursor SHA does not match P1 step2 data.pt")
            step3_data = next((item for item in step3_inventory if item["path"] == "data.pt"), None)
            if data_item and step3_data and data_item["sha256"] == step3_data["sha256"]:
                a3_failures.append("step3 data cursor did not advance from step2")

        resume_syncs = [
            row for row in execution_records
            if row.get("record_type") == "weight_sync_ack"
            and row.get("experiment_name") == resume_experiment
            and int(row.get("actor_version", -1)) in (2, 3)
        ]
        for rank in manifest["weight_sync"]["required_worker_ranks"]:
            loaded = next(
                (row for row in resume_syncs if int(row["vllm_worker_rank"]) == rank and int(row["actor_version"]) == 2),
                None,
            )
            updated = next(
                (row for row in resume_syncs if int(row["vllm_worker_rank"]) == rank and int(row["actor_version"]) == 3),
                None,
            )
            if loaded is None or updated is None:
                a3_failures.append(f"rank {rank} missing resume optimizer continuity evidence")
                continue
            loaded_step = loaded.get("optimizer_step_max")
            updated_step = updated.get("optimizer_step_max")
            loaded_epoch = loaded.get("lr_scheduler_last_epoch")
            updated_epoch = updated.get("lr_scheduler_last_epoch")
            if loaded_step is None or updated_step is None or int(updated_step) <= int(loaded_step):
                a3_failures.append(
                    f"rank {rank} optimizer step did not advance across resume: {loaded_step} -> {updated_step}"
                )
            loaded_state_count = int(loaded.get("optimizer_state_entry_count", 0))
            updated_state_count = int(updated.get("optimizer_state_entry_count", 0))
            if loaded_state_count < 1 or updated_state_count != loaded_state_count:
                a3_failures.append(
                    f"rank {rank} optimizer state entry count changed across resume: "
                    f"{loaded_state_count} -> {updated_state_count}"
                )
            if loaded.get("optimizer_step_histogram") == updated.get("optimizer_step_histogram"):
                a3_failures.append(f"rank {rank} optimizer step histogram did not change across resume")
            if loaded_epoch is None or updated_epoch is None or int(updated_epoch) <= int(loaded_epoch):
                a3_failures.append(
                    f"rank {rank} scheduler epoch did not advance across resume: {loaded_epoch} -> {updated_epoch}"
                )

    required_versions = [0, 1, 2] + ([3] if phase == "final" else [])
    required_syncs = [
        (fresh_experiment, 0, "fresh_initial"),
        (fresh_experiment, 1, "post_actor_update"),
        (fresh_experiment, 2, "post_actor_update"),
    ]
    if phase == "final":
        required_syncs.extend([
            (resume_experiment, 2, "resume_loaded"),
            (resume_experiment, 3, "post_actor_update"),
        ])
    a4_ok, a4_failures, digests = audit_sync(
        execution_records,
        required_versions,
        manifest["weight_sync"]["required_worker_ranks"],
        required_syncs,
        manifest["weight_sync"]["parameter_names"],
        manifest["weight_sync"]["transfer_format"],
        manifest["weight_sync"]["expected_loaded_parameter_count"],
    )

    rollout_starts = [row for row in execution_records if row.get("record_type") == "rollout_start"]
    required_rollouts = [
        (fresh_experiment, 1, 0),
        (fresh_experiment, 2, 1),
    ] + ([(resume_experiment, 3, 2)] if phase == "final" else [])
    observed_rollouts = {
        (row.get("experiment_name"), int(row.get("global_step", -1)), int(row.get("actor_version", -1)))
        for row in rollout_starts
    }
    if observed_rollouts != set(required_rollouts):
        a4_failures.append(
            f"rollout_start coverage {sorted(observed_rollouts)} != {sorted(required_rollouts)}"
        )
    for experiment, step, actor_version in required_rollouts:
        matches = [
            row for row in rollout_starts
            if row.get("experiment_name") == experiment
            and int(row.get("global_step", -1)) == step
        ]
        if len(matches) != 1:
            a4_failures.append(
                f"expected exactly one rollout_start for {experiment} step {step}, found {len(matches)}"
            )
            continue
        record = matches[0]
        if int(record.get("actor_version", -1)) != actor_version:
            a4_failures.append(
                f"rollout step {step} used actor version {record.get('actor_version')} != {actor_version}"
            )
        if record.get("sampled_tensor_digest") != digests.get(actor_version):
            a4_failures.append(
                f"rollout step {step} digest is not bound to actor version {actor_version}"
            )
    a4_ok = not a4_failures

    signal_records = [
        row for row in execution_records if row.get("record_type") == "execution_signal"
    ]
    signal_steps = {int(row["global_step"]): row for row in signal_records}
    required_signal_steps = [1, 2] + ([3] if phase == "final" else [])
    a5_failures = []
    observed_signal_steps = {int(row.get("global_step", -1)) for row in signal_records}
    if observed_signal_steps != set(required_signal_steps):
        a5_failures.append(
            f"execution signal step coverage {sorted(observed_signal_steps)} "
            f"!= {required_signal_steps}"
        )
    signal_summary: dict[int, dict] = {}
    for step in required_signal_steps:
        step_records = [row for row in signal_records if int(row.get("global_step", -1)) == step]
        if len(step_records) != 1:
            a5_failures.append(
                f"expected exactly one execution signal for step {step}, found {len(step_records)}"
            )
            continue
        record = step_records[0]
        expected_experiment = fresh_experiment if step in (1, 2) else resume_experiment
        if (
            int(record.get("actor_version", -1)) != step
            or record.get("experiment_name") != expected_experiment
        ):
            a5_failures.append(
                f"step {step} execution signal identity drift: "
                f"experiment={record.get('experiment_name')}, actor_version={record.get('actor_version')}"
            )
        nonfinite_names = record.get("nonfinite_metric_names")
        if not isinstance(nonfinite_names, list):
            a5_failures.append(f"step {step} is missing explicit non-finite metric evidence")
        elif nonfinite_names:
            a5_failures.append(f"step {step} contains non-finite metrics {sorted(nonfinite_names)}")
        values = record.get("metrics", {}).values()
        if any(not math.isfinite(float(value)) for value in values):
            a5_failures.append(f"non-finite execution metric at step {step}")
        metrics = record.get("metrics", {})
        required_metric_fragments = ("grad_norm", "pg_loss", "rewards/", "advantages/")
        missing_fragments = [
            fragment for fragment in required_metric_fragments
            if not any(fragment in key for key in metrics)
        ]
        if missing_fragments:
            a5_failures.append(f"step {step} missing execution metric families {missing_fragments}")
        signal_summary[step] = {
            "grad_norm": metrics.get("actor/grad_norm"),
            "reward_mean": metrics.get("critic/rewards/mean"),
            "advantage_min": metrics.get("critic/advantages/min"),
            "advantage_max": metrics.get("critic/advantages/max"),
            "pg_loss": metrics.get("actor/pg_loss"),
        }
    fresh_signals = [signal_summary.get(step, {}) for step in (1, 2)]
    if not any(abs(float(row.get("grad_norm") or 0.0)) > 0 for row in fresh_signals):
        a5_failures.append("both fresh steps have zero or missing grad norm")
    if not any(
        abs(float(row.get("advantage_min") or 0.0)) > 0
        or abs(float(row.get("advantage_max") or 0.0)) > 0
        for row in fresh_signals
    ):
        a5_failures.append("both fresh steps have zero or missing advantages")
    fresh_delta = any(digests.get(left) != digests.get(right) for left, right in ((0, 1), (1, 2)))
    if not fresh_delta:
        a5_failures.append("fresh versions 0→1→2 contain no sampled parameter delta")
    if phase == "final" and digests.get(2) == digests.get(3):
        a5_failures.append("resume version 2→3 contains no sampled parameter delta")

    audits = {
        "A1": {"status": "PASS" if a1_ok else "FAIL", "failures": a1_failures},
        "A2": {"status": "PASS" if not a2_failures else "FAIL", "failures": a2_failures},
        "A3": {"status": "PASS" if not a3_failures else "FAIL", "failures": a3_failures, "applicable": phase == "final"},
        "A4": {"status": "PASS" if a4_ok else "FAIL", "failures": a4_failures, "version_digests": digests},
        "A5": {
            "status": "PASS" if not a5_failures else "FAIL",
            "failures": a5_failures,
            "signals": signal_summary,
            "zero_grad_fraction": (
                sum(abs(float(row.get("grad_norm") or 0.0)) == 0 for row in signal_summary.values())
                / len(signal_summary)
                if signal_summary else None
            ),
        },
    }
    applicable = ["A1", "A2", "A4", "A5"] + (["A3"] if phase == "final" else [])
    p0_path = Path(paths["certificate_root"]) / "p0_preflight.json"
    p0_certificate = json.loads(p0_path.read_text()) if p0_path.is_file() else {}
    p0_commit = p0_certificate.get("evidence", {}).get("git_commit")
    resolved_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    resolved_manifest_sha256 = hashlib.sha256(resolved_payload).hexdigest()
    p0_resolved_manifest_sha256 = p0_certificate.get("evidence", {}).get("resolved_manifest_sha256")
    p0_resolved_manifest_path = p0_certificate.get("evidence", {}).get("resolved_manifest_path")
    resolved_manifest_file_matches = False
    if p0_resolved_manifest_path and Path(p0_resolved_manifest_path).is_file():
        frozen_resolved_manifest = json.loads(Path(p0_resolved_manifest_path).read_text(encoding="utf-8"))
        resolved_manifest_file_matches = frozen_resolved_manifest == manifest
    ledger_commits = {row.get("git_commit") for row in execution_records}
    all_ledger_commits = {row.get("git_commit") for row in ledger}
    p0_run_id = p0_certificate.get("evidence", {}).get("run_id")
    ledger_run_ids = {row.get("run_id") for row in ledger}
    current_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    p0_pass = (
        p0_certificate.get("status") == "PASS"
        and p0_commit == current_commit
        and p0_resolved_manifest_sha256 == resolved_manifest_sha256
        and resolved_manifest_file_matches
        and ledger_commits == {current_commit}
        and all_ledger_commits == {current_commit}
        and ledger_run_ids == {p0_run_id}
        and not ledger_failures
    )
    p1_pass = not step2_missing and all(step in signal_steps for step in (1, 2))
    p2_pass = phase != "final" or (
        not step3_missing and 3 in signal_steps and not a3_failures and digests.get(2) != digests.get(3)
    )
    gates = {
        "P0": "PASS" if p0_pass else "FAIL",
        "P1": "PASS" if p1_pass else "FAIL",
        "P2": "PASS" if p2_pass else ("NOT_APPLICABLE" if phase != "final" else "FAIL"),
    }
    required_gates = ["P0", "P1"] + (["P2"] if phase == "final" else [])
    passed = all(gates[name] == "PASS" for name in required_gates) and all(
        audits[name]["status"] == "PASS" for name in applicable
    )
    failed_order = [name for name in required_gates if gates[name] != "PASS"] + [
        name for name in applicable if audits[name]["status"] == "FAIL"
    ]
    report = {
        "phase": phase,
        "status": "PASS" if passed else "FAIL",
        "decision": ("GATE_A_PASS" if phase == "final" else "P1_AUDIT_PASS") if passed else f"GATE_A_NO_GO:{failed_order[0]}",
        "gates": gates,
        "audits": audits,
        "ledger_sha256": sha256_file(ledger_path) if ledger_path.is_file() else None,
        "ledger_record_count": len(ledger),
        "ledger_tail_record_sha256": ledger[-1].get("record_sha256") if ledger else None,
        "ledger_failures": ledger_failures,
        "step2_inventory": step2_inventory,
        "step3_inventory": step3_inventory,
    }
    inventory_records = [
        {"global_step": 2, "inventory": step2_inventory},
        *([{"global_step": 3, "inventory": step3_inventory}] if phase == "final" else []),
    ]
    return report, inventory_records


def verify_resume_source(manifest: dict) -> dict:
    paths = manifest["paths"]
    failures: list[str] = []
    p1_path = Path(paths["certificate_root"]) / "p1_audit_report.json"
    if not p1_path.is_file():
        failures.append("missing P1 audit report")
        p1_report = {}
    else:
        p1_report = json.loads(p1_path.read_text(encoding="utf-8"))
        if p1_report.get("status") != "PASS" or p1_report.get("decision") != "P1_AUDIT_PASS":
            failures.append("P1 audit report is not an immutable PASS")
    current_inventory, missing = component_inventory(
        Path(paths["resume_source"]), int(manifest["gpu"]["world_size"])
    )
    failures.extend(f"resume source missing {item}" for item in missing)
    if p1_report.get("step2_inventory") != current_inventory:
        failures.append("resume source inventory differs from the P1-frozen step2 checkpoint")
    ledger_path = Path(paths["execution_ledger"])
    ledger = read_jsonl(ledger_path)
    failures.extend(validate_jsonl_chain(ledger))
    schema = json.loads((REPO_ROOT / manifest["ledger_schema"]).read_text(encoding="utf-8"))
    failures.extend(validate_ledger_schema(ledger, schema))
    p0_path = Path(paths["certificate_root"]) / "p0_preflight.json"
    p0 = json.loads(p0_path.read_text(encoding="utf-8")) if p0_path.is_file() else {}
    run_id = p0.get("evidence", {}).get("run_id")
    if not run_id or {row.get("run_id") for row in ledger} != {run_id}:
        failures.append("resume ledger is not bound to the P0 run ID")
    frozen_count = p1_report.get("ledger_record_count")
    if not isinstance(frozen_count, int) or frozen_count < 1 or frozen_count > len(ledger):
        failures.append(f"invalid P1 ledger prefix length: {frozen_count}")
    else:
        prefix = ledger[:frozen_count]
        if jsonl_records_sha256(prefix) != p1_report.get("ledger_sha256"):
            failures.append("execution ledger P1 prefix differs before resume")
        if prefix[-1].get("record_sha256") != p1_report.get("ledger_tail_record_sha256"):
            failures.append("execution ledger P1 tail differs before resume")
        suffix = ledger[frozen_count:]
        if [row.get("record_type") for row in suffix] != ["checkpoint_inventory", "audit_result"]:
            failures.append(
                "unexpected ledger records were appended between P1 evidence and resume: "
                f"{[row.get('record_type') for row in suffix]}"
            )
        elif (
            int(suffix[0].get("global_step", -1)) != 2
            or suffix[1].get("phase") != "p1"
            or suffix[1].get("status") != "PASS"
        ):
            failures.append("P1 audit suffix does not describe the frozen step2 PASS")
        else:
            inventory_record, audit_record = suffix
            expected_experiment = manifest["experiments"]["fresh"]
            expected_commit = p0.get("evidence", {}).get("git_commit")
            if (
                inventory_record.get("inventory") != current_inventory
                or inventory_record.get("inventory") != p1_report.get("step2_inventory")
            ):
                failures.append(
                    "current step2, hash-chained P1 inventory and P1 report inventory differ"
                )
            for record, label in ((inventory_record, "inventory"), (audit_record, "audit")):
                if (
                    record.get("experiment_name") != expected_experiment
                    or record.get("run_id") != run_id
                    or record.get("git_commit") != expected_commit
                ):
                    failures.append(f"P1 {label} ledger identity is not bound to this run/commit")
            if (
                inventory_record.get("record_index") != frozen_count
                or audit_record.get("record_index") != frozen_count + 1
            ):
                failures.append("P1 inventory/audit records are not at the certified chain positions")
    return {
        "gate": "P2_RESUME_PREFLIGHT",
        "status": "PASS" if not failures else "FAIL",
        "decision": "RESUME_SOURCE_PASS" if not failures else "GATE_A_NO_GO:P2",
        "failures": failures,
        "step2_inventory": current_inventory,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("p1", "final"), default="final")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--verify-resume-source", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_frozen_manifest(args.manifest)
    except ValueError as error:
        print(json.dumps({
            "phase": args.phase,
            "status": "FAIL",
            "decision": "GATE_A_NO_GO:P0",
            "failures": [str(error)],
        }, indent=2, sort_keys=True))
        return 1
    if args.verify_resume_source:
        if args.write_report or args.output:
            raise SystemExit("--verify-resume-source is read-only and cannot write a report")
        result = verify_resume_source(manifest)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    report, inventory_records = build_report(manifest, args.phase)
    if args.write_report:
        name = "p1_audit_report.json" if args.phase == "p1" else "gate_a_final_report.json"
        target = args.output or Path(manifest["paths"]["certificate_root"]) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise SystemExit(f"refusing to overwrite append-only report: {target}")
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        from recurrent.research.gate_a_execution import append_gate_a_record

        for inventory_record in inventory_records:
            append_gate_a_record("checkpoint_inventory", **inventory_record)
        append_gate_a_record(
            "audit_result",
            phase=args.phase,
            status=report["status"],
            decision=report["decision"],
            report=str(target),
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

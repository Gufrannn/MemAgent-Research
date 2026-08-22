#!/usr/bin/env python3
"""Fail-closed CORAL preflight. It never allocates or modifies a GPU."""
from __future__ import annotations
import argparse, copy, json, os, re, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from recurrent.research.cosi import canonical_sha256, sha256_file
from recurrent.research.coral_evidence import (
    validate_original_training_authority,
    validate_stable_s128_authority,
)

BRANCH="h20/qwen25-7b-cosi-t25-frozen-20260822"
EXPECTED_TRAINING = {
    "seed": 2026,
    "trajectory_seed_mode": "independent",
    "train_batch_size": 4,
    "rollout_n": 2,
    "ppo_mini_batch_size": 4,
    "chunk_size": 5000,
    "max_chunks": 8,
    "max_prompt_length": 8192,
    "max_response_length": 1024,
    "ppo_max_token_len_per_gpu": 16384,
    "log_prob_max_token_len_per_gpu": 32768,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 16,
    "gpu_memory_utilization": 0.55,
    "actor_learning_rate": 0.000001,
    "actor_lr_warmup_steps": 2,
    "clip_ratio_high": 0.2,
    "entropy_coefficient": 0.0,
    "kl_loss_coefficient": 0.001,
    "ppo_epochs": 1,
    "loss_aggregation": "token-mean",
    "anchors": [5, 10, 15, 20, 25],
}
EXPECTED_PROTOCOL = {
    "advantage_estimator": "grpo",
    "grpo_use_adv": False,
    "reward_manager": "naive",
    "reward_function": "recurrent/research/hotpotqa_dense_reward.py",
    "reward_function_sha256": "dfc62ce689305a93f2d58afad9ef81fd6ba219ef6c19b9171b68c0462e9549f9",
    "reward_function_name": "compute_score",
    "reward_f1_weight": 0.95,
    "reward_grounded_box_bonus": 0.05,
    "kl_loss_type": "low_var_kl",
    "weight_sync_samples_per_tensor": 256,
    "trajectory_candidates_per_prompt": 2,
}
EXPECTED_EVALUATION = {
    "eval_manifest_hash": "351d7e58d6e67a1dc91bc3275f2c9407fd329a470b4a92ed37cf65945d12d84a",
    "sample_count": 128,
    "interfaces": ["I", "Original5", "Original10", "Original15", "Original20", "Original25"],
    "metrics": ["normalized_em", "token_f1", "format"],
}
RESOLVED_CONFIG_WHITELIST = (
    "algorithm.coral",
    "custom_reward_function.path",
    "trainer.project_name", "trainer.experiment_name", "trainer.default_local_dir",
    "trainer.total_training_steps", "trainer.save_freq", "trainer.max_actor_ckpt_to_keep",
    "trainer.resume_mode", "trainer.resume_from_path",
)


def git(*args): return subprocess.check_output(["git","-C",str(ROOT),*args],text=True).strip()
def authenticated(path, decision):
    value=json.loads(path.read_text()); digest=value.get("report_sha256")
    unsigned={k:v for k,v in value.items() if k!="report_sha256"}
    if value.get("status")!="PASS" or value.get("decision")!=decision or digest!=canonical_sha256(unsigned):
        raise ValueError(f"COSI_NO_GO: invalid gate {path}")
    return value


def _sha_binding(env, path_key, hash_key, label):
    path = Path(env[path_key]).resolve()
    expected = env.get(hash_key, "")
    if not path.is_file() or re.fullmatch(r"[0-9a-f]{64}", expected) is None \
            or sha256_file(path) != expected:
        raise ValueError(f"COSI_NO_GO: {label} external SHA binding")
    return path


def _nested_parent(value, dotted):
    parent = value
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            raise ValueError(f"COSI_NO_GO: resolved config lacks whitelist path {dotted}")
        parent = parent[part]
    return parent, parts[-1]


def exhaustive_resolved_config_diff(original, method):
    """Require equality of the complete resolved Hydra tree outside whitelist."""
    left, right = copy.deepcopy(original), copy.deepcopy(method)
    observed = {}
    for dotted in RESOLVED_CONFIG_WHITELIST:
        left_parent, leaf = _nested_parent(left, dotted)
        right_parent, _ = _nested_parent(right, dotted)
        left_value = left_parent.get(leaf, "__ABSENT__")
        right_value = right_parent.get(leaf, "__ABSENT__")
        if left_value != right_value:
            observed[dotted] = {"original": left_value, "method": right_value}
        left_parent[leaf] = "__CORAL_WHITELIST__"
        right_parent[leaf] = "__CORAL_WHITELIST__"
    if left != right:
        raise ValueError("COSI_NO_GO: resolved Method config drift outside explicit whitelist")
    if "algorithm.coral" not in observed:
        raise ValueError("COSI_NO_GO: CORAL method field is not an observed whitelist difference")
    return observed


def emit_method_overrides(work, gpu_pair, stage):
    run_id = os.environ.get("MEMAGENT_COSI_RUN_ID", "coral_seed2026_primary_v1")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{7,79}", run_id) is None:
        raise ValueError("COSI_NO_GO: run ID for resolved Method config")
    experiment = f"qwen25_7b_coral_fresh_t25_seed2026_{run_id}"
    phase = "resume" if stage == "continue" else "fresh"
    environment = {
        **os.environ,
        "WORK_ROOT": str(work), "CODE": str(ROOT),
        "MODEL": str(work / "models/Qwen2.5-7B-Instruct"),
        "TRAIN": str(work / "datasets/hotpotqa/hotpotqa_train_32k.parquet"),
        "VAL": str(work / "datasets/hotpotqa/hotpotqa_dev.parquet"),
        "PYTHON": str(work / ".venv/bin/python"), "EMIT_TRAINER_OVERRIDES": "1",
        "CUDA_VISIBLE_DEVICES": ",".join(str(item) for item in gpu_pair),
        "PHASE": phase, "EXP": experiment,
        "RUN_SEED": "2026", "TRAIN_BATCH_SIZE": "4", "ROLLOUT_N": "2",
        "PPO_MINI_BATCH_SIZE": "4", "N_GPUS": "2", "FSDP_SIZE": "2",
        "FRESH_TOTAL_STEPS": "25", "SAVE_FREQ": "5", "MAX_ACTOR_CKPT_TO_KEEP": "30",
        "REWARD_MANAGER": "naive", "GPU_MEMORY_UTILIZATION": "0.55",
    }
    if phase == "resume":
        environment.update({
            "RESUME_FROM": str(work / "logs/memory_agent" / experiment / "global_step_5"),
            "RESUME_SOURCE_STEP": "5", "RESUME_TOTAL_STEPS": "25",
        })
    coral = [
        "+algorithm.coral.enabled=true", "+algorithm.coral.active_from_update=1",
        "+algorithm.coral.schedule=odd_writer_even_terminal_answer_v2",
        "+algorithm.coral.role_partition=nonfinal_memory_writer_vs_final_answer",
        "+algorithm.coral.require_recurrent=true", "+algorithm.coral.require_grpo=true",
        "+algorithm.coral.require_gate_a_sync=true", "trainer.project_name=memagent_coral",
    ]
    result = subprocess.run(
        ["bash", str(ROOT / "experiments/7b_gate_a/run_gate_a.sh"), *coral],
        cwd=ROOT, env=environment, text=True, capture_output=True,
    )
    if result.returncode:
        raise ValueError(f"COSI_NO_GO: real Method runner emit failed: {result.stderr}")
    try:
        overrides = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError("COSI_NO_GO: real Method runner emitted invalid argv") from error
    if not isinstance(overrides, list) or any(not isinstance(item, str) for item in overrides):
        raise ValueError("COSI_NO_GO: real Method runner argv schema")
    return overrides


def validate_resolved_original_copy(original_p0, method_overrides):
    evidence = original_p0.get("evidence", {})
    original_argv = evidence.get("trainer_override_argv")
    if original_p0.get("status") != "PASS" or original_p0.get("decision") != "T25_P0_PASS" \
            or not isinstance(original_argv, list) \
            or canonical_sha256(original_argv) != evidence.get("trainer_override_argv_sha256"):
        raise ValueError("COSI_NO_GO: accepted Original P0 trainer argv authentication")
    from tools.h20.preflight_qwen25_7b_original_t25 import compose_resolved_trainer_config
    original_config = compose_resolved_trainer_config(original_argv)
    if canonical_sha256(original_config) != evidence.get("resolved_trainer_config_sha256"):
        raise ValueError("COSI_NO_GO: accepted Original resolved config authentication")
    method_config = compose_resolved_trainer_config(method_overrides)
    from recurrent.research.coral import validate_config
    try:
        validate_config(method_config["algorithm"]["coral"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("COSI_NO_GO: resolved Method CORAL config") from error
    differences = exhaustive_resolved_config_diff(original_config, method_config)
    normalized_method = copy.deepcopy(method_config)
    for dotted in RESOLVED_CONFIG_WHITELIST:
        parent, leaf = _nested_parent(normalized_method, dotted)
        parent[leaf] = "__CORAL_WHITELIST__"
    return {
        "accepted_original_trainer_override_argv_sha256": canonical_sha256(original_argv),
        "accepted_original_resolved_config_sha256": canonical_sha256(original_config),
        "method_trainer_override_argv_sha256": canonical_sha256(method_overrides),
        "method_resolved_config_sha256": canonical_sha256(method_config),
        "method_nonwhitelist_config_sha256": canonical_sha256(normalized_method),
        "whitelist": list(RESOLVED_CONFIG_WHITELIST),
        "observed_whitelist_differences": differences,
    }


def validate_original_protocol(manifest, original):
    """Compare every frozen scientific/runtime leaf exposed by Original."""
    if manifest.get("training") != EXPECTED_TRAINING \
            or manifest.get("protocol") != EXPECTED_PROTOCOL \
            or manifest.get("evaluation") != EXPECTED_EVALUATION:
        raise ValueError("COSI_NO_GO: frozen Method protocol fields drifted")
    original_model = original.get("model", {})
    original_files = original_model.get("files")
    if not isinstance(original_files, list) or len(original_files) < 5:
        raise ValueError("COSI_NO_GO: Original model/tokenizer inventory missing")
    file_map = {}
    for item in original_files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"} \
                or not isinstance(item["path"], str) or not item["path"] \
                or Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts \
                or item["path"] in file_map:
            raise ValueError("COSI_NO_GO: Original model/tokenizer inventory malformed")
        file_map[item["path"]] = {"size": item["size"], "sha256": item["sha256"]}
    required_model_files = {
        "config.json", "generation_config.json", "merges.txt", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "model.safetensors.index.json",
    }
    if not required_model_files.issubset(file_map) \
            or any(re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) is None
                   or type(row["size"]) is not int or row["size"] < 1
                   for row in file_map.values()):
        raise ValueError("COSI_NO_GO: Original model/tokenizer inventory malformed")
    original_training = original.get("training", {})
    original_data = original.get("data", {})
    original_backend = original.get("backend", {})
    original_weight_sync = original.get("weight_sync", {})
    comparisons = {
        "model.id": (manifest["model"]["id"], original_model.get("id")),
        "model.revision": (manifest["model"]["revision"], original_model.get("revision")),
        "model.config_sha256": (
            manifest["model"]["config_sha256"], file_map.get("config.json", {}).get("sha256"),
        ),
        "data.train_sha256": (manifest["data"]["train_sha256"], original_data.get("train_sha256")),
        "data.validation_sha256": (manifest["data"]["validation_sha256"], original_data.get("validation_sha256")),
        "data.shuffle": (manifest["data"]["shuffle"], original_data.get("shuffle")),
        "data.dataloader_num_workers": (
            manifest["data"]["dataloader_num_workers"], original_data.get("dataloader_num_workers"),
        ),
        "data.filter_overlong_prompts": (True, original_data.get("filter_overlong_prompts")),
        "training.seed": (EXPECTED_TRAINING["seed"], original_training.get("seed")),
        "training.trajectory_seed_mode": (
            EXPECTED_TRAINING["trajectory_seed_mode"], original_training.get("trajectory_seed_mode"),
        ),
        "training.train_batch_size": (
            EXPECTED_TRAINING["train_batch_size"], original_training.get("train_batch_size"),
        ),
        "training.rollout_n": (EXPECTED_TRAINING["rollout_n"], original_training.get("rollout_n")),
        "training.ppo_mini_batch_size": (
            EXPECTED_TRAINING["ppo_mini_batch_size"], original_training.get("ppo_mini_batch_size"),
        ),
        "training.chunk_size": (EXPECTED_TRAINING["chunk_size"], original_training.get("chunk_size")),
        "training.max_chunks": (EXPECTED_TRAINING["max_chunks"], original_training.get("max_chunks")),
        "training.max_prompt_length": (
            EXPECTED_TRAINING["max_prompt_length"], original_training.get("max_prompt_length"),
        ),
        "training.max_response_length": (
            EXPECTED_TRAINING["max_response_length"], original_training.get("max_response_length"),
        ),
        "training.ppo_max_token_len_per_gpu": (
            EXPECTED_TRAINING["ppo_max_token_len_per_gpu"], original_training.get("ppo_max_token_len_per_gpu"),
        ),
        "training.log_prob_max_token_len_per_gpu": (
            EXPECTED_TRAINING["log_prob_max_token_len_per_gpu"], original_training.get("log_prob_max_token_len_per_gpu"),
        ),
        "training.max_num_batched_tokens": (
            EXPECTED_TRAINING["max_num_batched_tokens"], original_training.get("max_num_batched_tokens"),
        ),
        "training.max_num_seqs": (
            EXPECTED_TRAINING["max_num_seqs"], original_training.get("max_num_seqs"),
        ),
        "training.gpu_memory_utilization": (
            EXPECTED_TRAINING["gpu_memory_utilization"], original_training.get("gpu_memory_utilization"),
        ),
        "training.actor_learning_rate": (
            EXPECTED_TRAINING["actor_learning_rate"], original_training.get("actor_learning_rate"),
        ),
        "training.actor_lr_warmup_steps": (
            EXPECTED_TRAINING["actor_lr_warmup_steps"], original_training.get("actor_lr_warmup_steps"),
        ),
        "training.clip_ratio_high": (
            EXPECTED_TRAINING["clip_ratio_high"], original_training.get("clip_ratio_high"),
        ),
        "training.entropy_coefficient": (
            EXPECTED_TRAINING["entropy_coefficient"], original_training.get("entropy_coefficient"),
        ),
        "training.kl_loss_coefficient": (
            EXPECTED_TRAINING["kl_loss_coefficient"], original_training.get("kl_loss_coefficient"),
        ),
        "backend.rollout": (manifest["backend"]["rollout"], original_backend.get("rollout")),
        "backend.allow_hf_fallback": (
            manifest["backend"]["allow_hf_fallback"], original_backend.get("allow_hf_fallback"),
        ),
        "backend.reward_manager": (EXPECTED_PROTOCOL["reward_manager"], original_backend.get("reward_manager")),
        "weight_sync.samples_per_tensor": (
            EXPECTED_PROTOCOL["weight_sync_samples_per_tensor"], original_weight_sync.get("samples_per_tensor"),
        ),
        "weight_sync.required_worker_ranks": ([0, 1], original_weight_sync.get("required_worker_ranks")),
    }
    drift = {
        key: {"method": left, "original": right}
        for key, (left, right) in comparisons.items() if left != right
    }
    if drift:
        raise ValueError(f"COSI_NO_GO: non-whitelisted Original protocol drift {drift}")
    return {
        "compared_leaves": comparisons,
        "original_model_file_inventory": file_map,
    }


def validate_local_artifacts(work, manifest, protocol_evidence):
    model_root = work / "models/Qwen2.5-7B-Instruct"
    inventory = protocol_evidence["original_model_file_inventory"]
    verified = []
    for relative, expected in sorted(inventory.items()):
        path = (model_root / relative).resolve()
        if model_root.resolve() not in path.parents or not path.is_file() \
                or path.stat().st_size != expected["size"] \
                or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"COSI_NO_GO: fresh-base model/tokenizer drift {relative}")
        verified.append({"path": relative, **expected})
    train = work / "datasets/hotpotqa/hotpotqa_train_32k.parquet"
    validation = work / "datasets/hotpotqa/hotpotqa_dev.parquet"
    if sha256_file(train) != manifest["data"]["train_sha256"] \
            or sha256_file(validation) != manifest["data"]["validation_sha256"]:
        raise ValueError("COSI_NO_GO: local train/validation data drift")
    reward_path = ROOT / EXPECTED_PROTOCOL["reward_function"]
    if sha256_file(reward_path) != EXPECTED_PROTOCOL["reward_function_sha256"]:
        raise ValueError("COSI_NO_GO: reward implementation drift")
    return canonical_sha256(verified)


def validate_continuation_binding(t5_preflight, *, expected_commit,
                                  manifest_sha256, original_manifest_sha256,
                                  original_p0_certificate_sha256,
                                  s128_manifest_sha256, model_inventory_sha256,
                                  protocol_comparison_sha256,
                                  method_nonwhitelist_config_sha256,
                                  evidence_authority_sha256, gpu_pair):
    expected = {
        "stage": "t5",
        "git_commit": expected_commit,
        "manifest_sha256": manifest_sha256,
        "original_resolved_manifest_sha256": original_manifest_sha256,
        "original_p0_certificate_sha256": original_p0_certificate_sha256,
        "s128_resolved_manifest_sha256": s128_manifest_sha256,
        "fresh_base_model_tokenizer_inventory_sha256": model_inventory_sha256,
        "original_protocol_comparison_sha256": protocol_comparison_sha256,
        "method_nonwhitelist_config_sha256": method_nonwhitelist_config_sha256,
        "evidence_authority_sha256": evidence_authority_sha256,
        "gpu_pair": gpu_pair,
    }
    if not isinstance(t5_preflight, dict) or any(
        t5_preflight.get(field) != value for field, value in expected.items()
    ):
        raise ValueError("COSI_NO_GO: T5/continuation binding drift")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--stage",choices=["research","t5","continue"],required=True); p.add_argument("--write-certificate",action="store_true"); a=p.parse_args()
    env=os.environ; required=[
        "MEMAGENT_COSI_WORK_ROOT", "MEMAGENT_COSI_REPO_DIR",
        "MEMAGENT_COSI_EXPECTED_COMMIT", "MEMAGENT_COSI_GPU_PAIR",
        "MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST",
        "MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST_SHA256",
        "MEMAGENT_COSI_S128_RESOLVED_MANIFEST",
        "MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256",
    ]
    missing=[x for x in required if not env.get(x)];
    if missing: raise ValueError(f"COSI_NO_GO: missing explicit bindings {missing}")
    repo=Path(env["MEMAGENT_COSI_REPO_DIR"]).resolve(); work=Path(env["MEMAGENT_COSI_WORK_ROOT"]).resolve()
    if repo!=ROOT.resolve() or not work.is_absolute(): raise ValueError("COSI_NO_GO: repository/work-root binding")
    expected=env["MEMAGENT_COSI_EXPECTED_COMMIT"]
    if not re.fullmatch(r"[0-9a-f]{40}",expected) or git("rev-parse","HEAD")!=expected: raise ValueError("COSI_NO_GO: exact commit mismatch")
    if git("branch","--show-current")!=BRANCH or git("status","--porcelain"): raise ValueError("COSI_NO_GO: wrong branch or dirty worktree")
    pair=env["MEMAGENT_COSI_GPU_PAIR"]
    if not re.fullmatch(r"[0-9]+,[0-9]+",pair): raise ValueError("COSI_NO_GO: GPU pair syntax")
    gpu=[int(v) for v in pair.split(",")]
    if gpu!=sorted(set(gpu)) or len(gpu)!=2: raise ValueError("COSI_NO_GO: GPU pair must be distinct canonical ascending")
    manifest_path=Path(a.manifest).resolve(); manifest=json.loads(manifest_path.read_text())
    if manifest.get("schema")!="memagent.cosi.run-manifest.v1" or not manifest.get("fresh_base_only") or manifest.get("method_active_from_update")!=1: raise ValueError("COSI_NO_GO: manifest method/fresh-base drift")
    if manifest.get("training") != EXPECTED_TRAINING \
            or manifest.get("protocol") != EXPECTED_PROTOCOL \
            or manifest.get("evaluation") != EXPECTED_EVALUATION:
        raise ValueError("COSI_NO_GO: frozen training/protocol fields drifted")
    from recurrent.research.coral import validate_config
    validate_config(manifest.get("coral", {}))
    original_path = _sha_binding(
        env, "MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST",
        "MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST_SHA256", "Original resolved manifest",
    )
    s128_resolved_path = _sha_binding(
        env, "MEMAGENT_COSI_S128_RESOLVED_MANIFEST",
        "MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256", "S128 resolved manifest",
    )
    original=json.loads(original_path.read_text())
    s128_resolved=json.loads(s128_resolved_path.read_text())
    authority = manifest.get("evidence_authority")
    if not isinstance(authority, dict):
        raise ValueError("COSI_NO_GO: shared evidence authority missing")
    original_authority = validate_original_training_authority(
        authority.get("original_training", {}), resolved_path=original_path,
        expected_resolved_sha256=env["MEMAGENT_COSI_ORIGINAL_RESOLVED_MANIFEST_SHA256"],
    )
    stable_authority = validate_stable_s128_authority(
        authority.get("stable_s128", {}), resolved_path=s128_resolved_path,
        expected_resolved_sha256=env["MEMAGENT_COSI_S128_RESOLVED_MANIFEST_SHA256"],
    )
    s128_resolved = stable_authority["resolved"]
    if s128_resolved.get("eval_manifest_hash") != EXPECTED_EVALUATION["eval_manifest_hash"]:
        raise ValueError("COSI_NO_GO: authenticated S128 eval protocol drift")
    protocol_evidence = validate_original_protocol(manifest, original)
    model_inventory_sha = validate_local_artifacts(work, manifest, protocol_evidence)
    method_overrides = emit_method_overrides(work, gpu, a.stage)
    resolved_config_evidence = validate_resolved_original_copy(
        original_authority["p0"], method_overrides,
    )
    resolved_config_comparison_sha = canonical_sha256(resolved_config_evidence)
    cert=work/"logs"/"cosi_preflight"/"certificates"; gates={}
    if a.stage in ("t5","continue"):
        # The repository never self-authorizes.  T5 becomes runnable only
        # after all externally issued, content-addressed gates below pass.
        if manifest.get("training_authorized") is not False \
                or manifest.get("authorization_mode") \
                != "external_authenticated_gates_only":
            raise ValueError("COSI_NO_GO: repository attempted self-authorization")
        expected_gate_hashes={
            "paper":"MEMAGENT_COSI_PAPER_REVIEW_SHA256",
            "e0":"MEMAGENT_COSI_E0_REPORT_SHA256",
            "e1":"MEMAGENT_COSI_E1_REPORT_SHA256",
            "baseline":"MEMAGENT_COSI_BASELINE_REPORT_SHA256",
        }
        for gate,variable in expected_gate_hashes.items():
            value=env.get(variable,"")
            if not re.fullmatch(r"[0-9a-f]{64}",value):
                raise ValueError(f"COSI_NO_GO: missing external expected hash {variable}")
        gates["paper"]=authenticated(cert/"paper_framing_review.json","CORAL_PAPER_FRAMING_GO")
        gates["e0"]=authenticated(cert/"coral_e0.json","CORAL_E0_PASS")
        gates["e1"]=authenticated(cert/"coral_e1_final_report.json","CORAL_E1_PASS")
        gates["baseline"]=authenticated(cert/"baseline_import.json","COSI_BASELINE_IMPORT_PASS")
        gate_paths={
            "paper":cert/"paper_framing_review.json",
            "e0":cert/"coral_e0.json",
            "e1":cert/"coral_e1_final_report.json",
            "baseline":cert/"baseline_import.json",
        }
        for gate,path in gate_paths.items():
            if sha256_file(path)!=env[expected_gate_hashes[gate]]:
                raise ValueError(f"COSI_NO_GO: external expected hash mismatch for {gate}")
    manifest_sha = sha256_file(manifest_path)
    original_manifest_sha = sha256_file(original_path)
    s128_resolved_manifest_sha = sha256_file(s128_resolved_path)
    if a.stage=="continue":
        run_id = os.environ.get("MEMAGENT_COSI_RUN_ID", "coral_seed2026_primary_v1")
        gates["t5"]=authenticated(
            work/"logs"/"coral"/run_id/"certificates"/"t5_health.json",
            "COSI_T5_HEALTH_PASS",
        )
        t5_preflight=authenticated(cert/"p0_t5.json","COSI_T5_P0_PASS")
        comparison_receipt = {
            key: {"method": left, "original": right}
            for key, (left, right) in sorted(protocol_evidence["compared_leaves"].items())
        }
        validate_continuation_binding(
            t5_preflight, expected_commit=expected,
            manifest_sha256=manifest_sha,
            original_manifest_sha256=original_manifest_sha,
            original_p0_certificate_sha256=original_authority["p0_sha256"],
            s128_manifest_sha256=s128_resolved_manifest_sha,
            model_inventory_sha256=model_inventory_sha,
            protocol_comparison_sha256=canonical_sha256(comparison_receipt),
            method_nonwhitelist_config_sha256=(
                resolved_config_evidence["method_nonwhitelist_config_sha256"]
            ),
            evidence_authority_sha256=canonical_sha256(authority),
            gpu_pair=gpu,
        )
        gates["t5_preflight"]=t5_preflight
    comparison_receipt = {
        key: {"method": left, "original": right}
        for key, (left, right) in sorted(protocol_evidence["compared_leaves"].items())
    }
    report={"schema":"memagent.cosi.preflight.v4","status":"PASS","decision":f"COSI_{a.stage.upper()}_P0_PASS","stage":a.stage,"git_commit":expected,"manifest_sha256":manifest_sha,"original_resolved_manifest_sha256":original_manifest_sha,"original_p0_certificate_sha256":original_authority["p0_sha256"],"original_training_final_sha256":original_authority["final_sha256"],"original_training_ledger_sha256":original_authority["ledger_sha256"],"s128_resolved_manifest_sha256":s128_resolved_manifest_sha,"s128_final_sha256":stable_authority["final_sha256"],"s128_ledger_sha256":stable_authority["ledger_sha256"],"evidence_authority_sha256":canonical_sha256(authority),"fresh_base_model_tokenizer_inventory_sha256":model_inventory_sha,"original_protocol_comparison_sha256":canonical_sha256(comparison_receipt),"original_protocol_compared_leaves":sorted(comparison_receipt),"resolved_config_comparison_sha256":resolved_config_comparison_sha,"method_nonwhitelist_config_sha256":resolved_config_evidence["method_nonwhitelist_config_sha256"],"resolved_config_comparison":resolved_config_evidence,"gpu_pair":gpu,"gate_hashes":{k:v["report_sha256"] for k,v in gates.items()}}
    report["report_sha256"]=canonical_sha256(report)
    if a.write_certificate:
        cert.mkdir(parents=True,exist_ok=True); out=cert/f"p0_{a.stage}.json"
        with out.open("x") as f: json.dump(report,f,indent=2,sort_keys=True); f.write("\n")
    print(json.dumps(report,sort_keys=True))
if __name__=="__main__": main()

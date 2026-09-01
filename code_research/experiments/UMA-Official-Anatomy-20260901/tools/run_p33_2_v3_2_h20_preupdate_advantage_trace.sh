#!/usr/bin/env bash
# P33.2-v3.2 H20 official-UMA pre-update advantage tracing runner.
#
# This script is intentionally fail-closed:
# - clones/checks out only the official UMA source at exact commit 768f962...
# - prepares a fresh instrumented worktree using the reviewed exact-anchor patch
# - never downloads models or datasets
# - never installs dependencies
# - runs only when explicit local model/data paths are provided
# - exits immediately after true GRPO advantages are traced, before optimizer steps

set -euo pipefail

OFFICIAL_UMA_REPO_URL="${OFFICIAL_UMA_REPO_URL:-https://github.com/ictnlp/unified-memory-agent.git}"
OFFICIAL_UMA_COMMIT="${OFFICIAL_UMA_COMMIT:-768f9620231bae11264771f59e43a4839506cf94}"
WORK_ROOT="${WORK_ROOT:-/data/cw/memagent_work}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_MODE="${RUN_MODE:-precheck}"
RUN_TAG="${RUN_TAG:-p33_2_v3_2_$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PREPARE_SCRIPT="$SCRIPT_DIR/prepare_p33_2_uma_credit_instrumentation.py"
SUMMARIZER_SCRIPT="$SCRIPT_DIR/summarize_p33_2_credit_trace.py"

CODE_ROOT="$WORK_ROOT/code"
OFFICIAL_SRC_DIR="${OFFICIAL_SRC_DIR:-$CODE_ROOT/UMA-official-$OFFICIAL_UMA_COMMIT}"
INSTRUMENTED_DIR="${INSTRUMENTED_DIR:-$CODE_ROOT/UMA-P33_2-v3_2-instrumented-$OFFICIAL_UMA_COMMIT-$RUN_TAG}"
TRACE_DIR="${TRACE_DIR:-$WORK_ROOT/runs/$RUN_TAG/uma_credit_trace}"
SUMMARY_DIR="${SUMMARY_DIR:-$WORK_ROOT/runs/$RUN_TAG/summary}"

UMA_MODEL_PATH="${UMA_MODEL_PATH:?Set UMA_MODEL_PATH to a local official 4B model directory. No HF download/substitution is allowed.}"
UMA_TRAIN_FILES_LITERAL="${UMA_TRAIN_FILES_LITERAL:?Set UMA_TRAIN_FILES_LITERAL to a Python-list literal of local train parquet paths.}"
UMA_VAL_FILES_LITERAL="${UMA_VAL_FILES_LITERAL:?Set UMA_VAL_FILES_LITERAL to a Python-list literal of local validation parquet paths.}"
EMBEDDING_SERVICE_ENDPOINT="${EMBEDDING_SERVICE_ENDPOINT:?Set EMBEDDING_SERVICE_ENDPOINT to the local embedding service used by official UMA.}"

PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-prompt_template.yaml}"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-external/verl/memagent/tool_config.yaml}"
REWARD_PATH="${REWARD_PATH:-external/verl/memagent/hotpotqa.py}"

UMA_TRAIN_BATCH_SIZE="${UMA_TRAIN_BATCH_SIZE:-8}"
UMA_ROLLOUT_N="${UMA_ROLLOUT_N:-4}"
UMA_N_GPUS="${UMA_N_GPUS:-8}"
UMA_MAX_PROMPT_LENGTH="${UMA_MAX_PROMPT_LENGTH:-8192}"
UMA_MAX_RESPONSE_LENGTH="${UMA_MAX_RESPONSE_LENGTH:-8192}"
UMA_PPO_MINI_BATCH_SIZE="${UMA_PPO_MINI_BATCH_SIZE:-4}"
UMA_ACTOR_MAX_TOKEN_LEN_PER_GPU="${UMA_ACTOR_MAX_TOKEN_LEN_PER_GPU:-16384}"
UMA_LOGPROB_MAX_TOKEN_LEN_PER_GPU="${UMA_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-32768}"
UMA_GPU_MEMORY_UTILIZATION="${UMA_GPU_MEMORY_UTILIZATION:-0.80}"

case "$RUN_MODE" in
  precheck|one_step)
    ;;
  *)
    echo "Invalid RUN_MODE=$RUN_MODE; expected precheck or one_step" >&2
    exit 2
    ;;
esac

mkdir -p "$CODE_ROOT" "$TRACE_DIR" "$SUMMARY_DIR"

echo "P33_2_V3_2_RUNNER_START"
echo "RUN_MODE=$RUN_MODE"
echo "OFFICIAL_UMA_REPO_URL=$OFFICIAL_UMA_REPO_URL"
echo "OFFICIAL_UMA_COMMIT=$OFFICIAL_UMA_COMMIT"
echo "OFFICIAL_SRC_DIR=$OFFICIAL_SRC_DIR"
echo "INSTRUMENTED_DIR=$INSTRUMENTED_DIR"
echo "TRACE_DIR=$TRACE_DIR"
echo "SUMMARY_DIR=$SUMMARY_DIR"
echo "UMA_MODEL_PATH=$UMA_MODEL_PATH"

if [[ ! -f "$PREPARE_SCRIPT" ]]; then
  echo "Missing prepare script: $PREPARE_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$SUMMARIZER_SCRIPT" ]]; then
  echo "Missing summarizer script: $SUMMARIZER_SCRIPT" >&2
  exit 2
fi

if [[ -e "$INSTRUMENTED_DIR" ]]; then
  echo "Instrumented destination already exists; refusing to overwrite: $INSTRUMENTED_DIR" >&2
  exit 2
fi

if [[ ! -d "$UMA_MODEL_PATH" || ! -f "$UMA_MODEL_PATH/config.json" ]]; then
  echo "UMA_MODEL_PATH is not a readable local model directory with config.json: $UMA_MODEL_PATH" >&2
  exit 2
fi

"$PYTHON_BIN" -m py_compile "$PREPARE_SCRIPT" "$SUMMARIZER_SCRIPT"

if [[ ! -d "$OFFICIAL_SRC_DIR/.git" ]]; then
  git clone "$OFFICIAL_UMA_REPO_URL" "$OFFICIAL_SRC_DIR"
fi

if [[ -n "$(git -C "$OFFICIAL_SRC_DIR" status --porcelain)" ]]; then
  echo "Official UMA source worktree is dirty; refusing checkout: $OFFICIAL_SRC_DIR" >&2
  git -C "$OFFICIAL_SRC_DIR" status --short >&2
  exit 2
fi

git -C "$OFFICIAL_SRC_DIR" fetch origin "$OFFICIAL_UMA_COMMIT"
git -C "$OFFICIAL_SRC_DIR" checkout --detach "$OFFICIAL_UMA_COMMIT"

ACTUAL_COMMIT="$(git -C "$OFFICIAL_SRC_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$OFFICIAL_UMA_COMMIT" ]]; then
  echo "Official UMA commit mismatch: expected $OFFICIAL_UMA_COMMIT got $ACTUAL_COMMIT" >&2
  exit 2
fi

for rel in \
  "external/verl/verl/experimental/agent_loop/agent_loop.py" \
  "external/verl/verl/experimental/agent_loop/tool_mem_agent_loop.py" \
  "external/verl/verl/trainer/ppo/ray_trainer.py" \
  "external/verl/run_qwen3-4b_memagent.sh" \
  "external/verl/memagent/tool_config.yaml" \
  "external/verl/memagent/hotpotqa.py" \
  "prompt_template.yaml"
do
  if [[ ! -f "$OFFICIAL_SRC_DIR/$rel" ]]; then
    echo "Missing official UMA target file: $rel" >&2
    exit 2
  fi
done

"$PYTHON_BIN" "$PREPARE_SCRIPT" \
  --src-repo "$OFFICIAL_SRC_DIR" \
  --dst-repo "$INSTRUMENTED_DIR" \
  --expected-commit "$OFFICIAL_UMA_COMMIT" \
  > "$SUMMARY_DIR/p33_2_v3_2_prepare_manifest.stdout.json"

for rel in \
  "external/verl/verl/experimental/agent_loop/agent_loop.py" \
  "external/verl/verl/experimental/agent_loop/tool_mem_agent_loop.py" \
  "external/verl/verl/trainer/ppo/ray_trainer.py" \
  "external/verl/verl/utils/uma_credit_trace.py"
do
  "$PYTHON_BIN" -m py_compile "$INSTRUMENTED_DIR/$rel"
done

"$PYTHON_BIN" - <<'PY' "$UMA_TRAIN_FILES_LITERAL" "$UMA_VAL_FILES_LITERAL"
import ast
import pathlib
import sys

for label, literal in [("train", sys.argv[1]), ("val", sys.argv[2])]:
    value = ast.literal_eval(literal)
    if not isinstance(value, list) or not value:
        raise SystemExit(f"{label} files literal must be a non-empty list")
    missing = [str(p) for p in value if not pathlib.Path(str(p)).is_file()]
    if missing:
        raise SystemExit(f"missing {label} files: {missing}")
print("DATA_FILE_PRECHECK_OK")
PY

"$PYTHON_BIN" - <<'PY' "$PYTHON_BIN" "$SUMMARY_DIR/p33_2_v3_2_env_inventory.json"
import importlib.util
import json
import shutil
import sys

modules = [
    "torch",
    "ray",
    "omegaconf",
    "transformers",
    "accelerate",
    "deepspeed",
    "sglang",
    "verl",
]
rows = {}
for name in modules:
    spec = importlib.util.find_spec(name)
    rows[name] = bool(spec)
payload = {
    "python": sys.executable,
    "python_arg": sys.argv[1],
    "modules": rows,
    "nvidia_smi": shutil.which("nvidia-smi"),
}
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
print(json.dumps(payload, indent=2, sort_keys=True))
required = ["torch", "ray", "omegaconf", "transformers"]
missing_required = [name for name in required if not rows.get(name)]
if missing_required:
    raise SystemExit(f"missing required modules: {missing_required}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$SUMMARY_DIR/p33_2_v3_2_nvidia_smi.txt"
else
  echo "nvidia-smi not found" > "$SUMMARY_DIR/p33_2_v3_2_nvidia_smi.txt"
fi

echo "P33_2_V3_2_PRECHECK_OK"

if [[ "$RUN_MODE" == "precheck" ]]; then
  echo "RUN_MODE_PRECHECK_COMPLETE_NO_ROLLOUT_NO_TRAINING_NO_OPTIMIZER"
  exit 0
fi

cd "$INSTRUMENTED_DIR"

if [[ ! -f "$PROMPT_TEMPLATE_PATH" ]]; then
  echo "Prompt template missing relative to instrumented worktree: $PROMPT_TEMPLATE_PATH" >&2
  exit 2
fi
if [[ ! -f "$TOOL_CONFIG_PATH" ]]; then
  echo "Tool config missing relative to instrumented worktree: $TOOL_CONFIG_PATH" >&2
  exit 2
fi
if [[ ! -f "$REWARD_PATH" ]]; then
  echo "Reward file missing relative to instrumented worktree: $REWARD_PATH" >&2
  exit 2
fi

export PYTHONPATH="$INSTRUMENTED_DIR/external/verl:${PYTHONPATH:-}"
export VERL_LOGGING_LEVEL=DEBUG
export HF_HUB_OFFLINE=1
export UMA_CREDIT_TRACE=1
export UMA_CREDIT_TRACE_PATH="$TRACE_DIR"
export UMA_CREDIT_TRACE_EXIT_AFTER_ADVANTAGE=1
export PROMPT_TEMPLATE_PATH="$PROMPT_TEMPLATE_PATH"
export EMBEDDING_SERVICE_ENDPOINT="$EMBEDDING_SERVICE_ENDPOINT"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init.runtime_env.working_dir="$INSTRUMENTED_DIR" \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="1" \
  +ray_kwargs.ray_init.runtime_env.env_vars.UMA_CREDIT_TRACE="1" \
  +ray_kwargs.ray_init.runtime_env.env_vars.UMA_CREDIT_TRACE_PATH="$TRACE_DIR" \
  +ray_kwargs.ray_init.runtime_env.env_vars.UMA_CREDIT_TRACE_EXIT_AFTER_ADVANTAGE="1" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PROMPT_TEMPLATE_PATH="$PROMPT_TEMPLATE_PATH" \
  +ray_kwargs.ray_init.runtime_env.env_vars.EMBEDDING_SERVICE_ENDPOINT="$EMBEDDING_SERVICE_ENDPOINT" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=True \
  algorithm.norm_adv_by_std_in_grpo=True \
  data.train_batch_size="$UMA_TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$UMA_MAX_PROMPT_LENGTH" \
  data.max_response_length="$UMA_MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.train_files="$UMA_TRAIN_FILES_LITERAL" \
  data.val_files="$UMA_VAL_FILES_LITERAL" \
  actor_rollout_ref.model.path="$UMA_MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$UMA_PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$UMA_ACTOR_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$UMA_LOGPROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.gpu_memory_utilization="$UMA_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.n="$UMA_ROLLOUT_N" \
  actor_rollout_ref.rollout.over_sample_rate=0.1 \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=10 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=3000 \
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$UMA_LOGPROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=p33_2_credit_trace \
  trainer.experiment_name="$RUN_TAG" \
  trainer.n_gpus_per_node="$UMA_N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  custom_reward_function.path="$REWARD_PATH" \
  custom_reward_function.name=reward_func

"$PYTHON_BIN" "$SUMMARIZER_SCRIPT" \
  --trace-glob "$TRACE_DIR/*.jsonl" \
  --output-prefix "$SUMMARY_DIR/p33_2_v3_2_credit_trace_summary"

echo "P33_2_V3_2_ONE_STEP_COMPLETE_EXITED_BEFORE_OPTIMIZER"

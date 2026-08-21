#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/hdr_memrl_common.sh"
: "${HDR_TARGET_STEP:?set HDR_TARGET_STEP to 5,10,15,20,25}"
[[ $HDR_TARGET_STEP =~ ^(5|10|15|20|25)$ ]] || { echo HDR_NO_GO:invalid_target >&2; exit 73; }
hdr_require_checkout; hdr_acquire_gpu_locks; hdr_require_gates; hdr_require_idle
mkdir -p "$HDR_ROOT" "$HDR_CERT"
if [[ $HDR_TARGET_STEP -eq 5 ]]; then
  [[ ! -e $HDR_OUTPUT ]] || { echo HDR_NO_GO:fresh_output_exists >&2; exit 74; }
  PHASE=fresh; SOURCE_STEP=0
else
  case "$HDR_TARGET_STEP" in 10) SOURCE_STEP=5;; 15) SOURCE_STEP=10;; 20) SOURCE_STEP=15;; 25) SOURCE_STEP=20;; esac
  "$HDR_PYTHON" - "$HDR_CERT/t${SOURCE_STEP}_health.json" "$SOURCE_STEP" <<'PY'
import json,os,sys
r=json.load(open(sys.argv[1])); prefix="HDR" if os.environ.get("HDR_VARIANT","dro")=="dro" else "UNIFORM"; expected=f"{prefix}_T{sys.argv[2]}_HEALTH_PASS"
raise SystemExit(0 if r.get("status")=="PASS" and r.get("decision")==expected else "HDR_NO_GO:previous_anchor_health")
PY
  [[ -d $HDR_OUTPUT/global_step_$SOURCE_STEP/actor && -f $HDR_OUTPUT/global_step_$SOURCE_STEP/data.pt && -f $HDR_OUTPUT/global_step_$SOURCE_STEP/hdr_dro_state.json && -f $HDR_OUTPUT/global_step_$SOURCE_STEP/hdr_checkpoint_binding.json ]] || { echo HDR_NO_GO:incomplete_resume_checkpoint >&2; exit 75; }
  "$HDR_PYTHON" - "$HDR_OUTPUT/global_step_$SOURCE_STEP" "$MEMAGENT_HDR_EXPECTED_COMMIT" "$HDR_RUN_ID" "$GPU_PAIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); b=json.load(open(root/'hdr_checkpoint_binding.json'))
if (b.get('git_commit'),b.get('run_id'),b.get('gpu_pair')) != tuple(sys.argv[2:5]): raise SystemExit('HDR_NO_GO:resume_binding_drift')
for item in b.get('inventory',[]):
 p=root/item['path']; h=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
 if h!=item['sha256'] or p.stat().st_size!=item['size']: raise SystemExit('HDR_NO_GO:checkpoint_inventory_tamper')
PY
  [[ ! -e $HDR_OUTPUT/global_step_$HDR_TARGET_STEP ]] || { echo HDR_NO_GO:target_would_overwrite >&2; exit 76; }
  PHASE=resume
fi
export CUDA_VISIBLE_DEVICES=$GPU_PAIR
export HDR_MANIFEST_SHA256
HDR_MANIFEST_SHA256=$(shasum -a 256 "$HDR_MANIFEST" | awk '{print $1}')
export GATE_A_FROZEN_AUDIT=1 GATE_A_EXECUTION_LEDGER=$HDR_ROOT/hdr_weight_sync_ledger.jsonl
export GATE_A_EXPERIMENT_NAME=$HDR_EXP GATE_A_GIT_COMMIT=$MEMAGENT_HDR_EXPECTED_COMMIT GATE_A_RUN_ID=$HDR_RUN_ID
export GATE_A_WEIGHT_DIGEST_SAMPLES=256
export GATE_A_WEIGHT_DIGEST_PARAMETERS=model.embed_tokens.weight,model.layers.0.input_layernorm.weight,model.layers.0.self_attn.o_proj.weight,model.layers.0.mlp.down_proj.weight,model.layers.27.input_layernorm.weight,model.layers.27.self_attn.o_proj.weight,model.layers.27.mlp.down_proj.weight,model.norm.weight
export HDR_ENABLE=1 HDR_DATASET_SHA256=798b7a2a9ece4f40884e2a9d02d165d7352df7763d1569ceaf402b45f76896f8
export HDR_HORIZONS='[8,12,16,32]' HDR_MIN_HORIZON=8 HDR_SCHEDULER_SEED=2026 HDR_ETA=0.1 HDR_RHO=0.2
if [[ $HDR_VARIANT == dro ]]; then export HDR_DRO_ENABLED=true; else export HDR_DRO_ENABLED=false; fi
export WORK_ROOT=$MEMAGENT_HDR_WORK_ROOT CODE=$HDR_REPO PYTHON=$HDR_PYTHON EXP=$HDR_EXP RUN_SEED=2026
export TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=4 N_GPUS=2 FSDP_SIZE=2 REWARD_MANAGER=naive
export GPU_MEMORY_UTILIZATION=0.55 PHASE SAVE_FREQ=5 MAX_ACTOR_CKPT_TO_KEEP=5
if [[ $PHASE == fresh ]]; then export FRESH_TOTAL_STEPS=5; else export RESUME_TOTAL_STEPS=$HDR_TARGET_STEP RESUME_SOURCE_STEP=$SOURCE_STEP RESUME_FROM=$HDR_OUTPUT/global_step_$SOURCE_STEP; fi
OVERRIDES_JSON=$(EMIT_TRAINER_OVERRIDES=1 bash "$HDR_REPO/experiments/7b_gate_a/run_gate_a.sh" | tail -n 1)
export OVERRIDES_JSON
HDR_RUNTIME_OVERRIDE_SHA256=$("$HDR_PYTHON" - "$HDR_ROOT/runtime_overrides_t${HDR_TARGET_STEP}.json" "$HDR_TARGET_STEP" "$SOURCE_STEP" <<'PY'
import hashlib,json,os,sys
raw=json.loads(os.environ['OVERRIDES_JSON'])
values={x.split('=',1)[0].lstrip('+'):x.split('=',1)[1] for x in raw if '=' in x}
expected={'data.train_batch_size':'4','actor_rollout_ref.rollout.n':'2','actor_rollout_ref.actor.ppo_mini_batch_size':'4','actor_rollout_ref.actor.optim.lr':'1e-6','algorithm.hdr_memrl.enabled':'true','algorithm.hdr_memrl.horizons':'[8,12,16,32]','recurrent.memory.config.hdr_enable':'true','trainer.total_training_steps':sys.argv[2]}
expected['algorithm.hdr_memrl.dro_enabled']=os.environ['HDR_DRO_ENABLED']
expected['actor_rollout_ref.rollout.seed']='2026'
expected['reward_model.reward_manager']='naive'
expected['actor_rollout_ref.rollout.name']='vllm'
expected['data.train_files']=os.environ['WORK_ROOT']+'/datasets/hotpotqa/hotpotqa_train_32k.parquet'
expected['data.val_files']=os.environ['WORK_ROOT']+'/datasets/hotpotqa/hotpotqa_dev.parquet'
expected['actor_rollout_ref.model.path']=os.environ['WORK_ROOT']+'/models/Qwen2.5-7B-Instruct'
for key,value in expected.items():
    if values.get(key)!=value: raise SystemExit(f'HDR_NO_GO:runtime_manifest_drift:{key}')
if int(sys.argv[3])==0 and values.get('trainer.resume_mode')!='disable': raise SystemExit('HDR_NO_GO:fresh_runtime_not_disable')
if int(sys.argv[3])>0 and values.get('trainer.resume_mode')!='resume_path': raise SystemExit('HDR_NO_GO:resume_runtime_not_path')
payload=json.dumps(raw,separators=(',',':'))
open(sys.argv[1],'x').write(payload+'\n')
print(hashlib.sha256(payload.encode()).hexdigest())
PY
)
"$HDR_PYTHON" - "$HDR_REPO" "$HDR_LEDGER" "$HDR_TARGET_STEP" "$SOURCE_STEP" "$MEMAGENT_HDR_EXPECTED_COMMIT" "$GPU_PAIR" "$HDR_RUN_ID" "$HDR_RUNTIME_OVERRIDE_SHA256" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
from recurrent.research.gate_a_execution import append_jsonl
append_jsonl(sys.argv[2],{"record_type":"launch","target_step":int(sys.argv[3]),"source_step":int(sys.argv[4]),"git_commit":sys.argv[5],"gpu_pair":sys.argv[6],"run_id":sys.argv[7],"runtime_override_sha256":sys.argv[8]})
PY
bash "$HDR_REPO/experiments/7b_gate_a/run_gate_a.sh" 2>&1 | tee -a "$HDR_ROOT/train_to_t${HDR_TARGET_STEP}.log"
[[ -f $HDR_OUTPUT/global_step_$HDR_TARGET_STEP/hdr_dro_state.json ]] || { echo HDR_NO_GO:missing_target_dual_checkpoint >&2; exit 77; }

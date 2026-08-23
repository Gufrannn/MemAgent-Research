#!/usr/bin/env bash
set -euo pipefail

[[ -n ${RWWPO_REPO_DIR:-} && $RWWPO_REPO_DIR == /* ]] || {
  echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:set absolute RWWPO_REPO_DIR' >&2; exit 60;
}
[[ -n ${RWWPO_WORK_ROOT:-} && $RWWPO_WORK_ROOT == /* ]] || {
  echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:set absolute RWWPO_WORK_ROOT' >&2; exit 61;
}
[[ ${RWWPO_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:set exact commit' >&2; exit 62;
}
[[ ${GPU_PAIR:-} =~ ^[0-9]+,[0-9]+$ ]] || {
  echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:set GPU_PAIR=N,M' >&2; exit 63;
}
IFS=, read -r GPU_A GPU_B <<< "$GPU_PAIR"
(( GPU_A < GPU_B )) || {
  echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:canonical ascending GPU pair' >&2; exit 64;
}
[[ -n ${RWWPO_NUMERIC_ORACLE_ROOT:-} && $RWWPO_NUMERIC_ORACLE_ROOT == /* ]] || {
  echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:set absolute one-use output root' >&2; exit 65;
}
[[ -f ${RWWPO_RELEASE_TEST_RECEIPT:-} \
   && ${RWWPO_RELEASE_TEST_RECEIPT_SHA256:-} =~ ^[0-9a-f]{64}$ \
   && -f ${RWWPO_MANIFEST:-} \
   && ${RWWPO_MANIFEST_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || {
  echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:bind release-test and manifest receipts' >&2; exit 71;
}
[[ ! -e $RWWPO_NUMERIC_ORACLE_ROOT ]] || {
  echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:output root already consumed' >&2; exit 66;
}
[[ $(cd "$RWWPO_REPO_DIR" && git rev-parse HEAD) == "$RWWPO_EXPECTED_COMMIT" \
   && $(cd "$RWWPO_REPO_DIR" && git branch --show-current) == \
      h20/qwen25-7b-tf-rwwpo-t25-frozen-20260822 \
   && -z $(cd "$RWWPO_REPO_DIR" && git status --porcelain) ]] || {
  echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:checkout' >&2; exit 67;
}
[[ -x $RWWPO_WORK_ROOT/.venv/bin/torchrun \
   && -f $RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct/config.json ]] || {
  echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:runtime/model' >&2; exit 68;
}

"$RWWPO_WORK_ROOT/.venv/bin/python" \
  "$RWWPO_REPO_DIR/tools/h20/verify_rwwpo2_release_tests.py" \
  --receipt "$RWWPO_RELEASE_TEST_RECEIPT" \
  --receipt-sha256 "$RWWPO_RELEASE_TEST_RECEIPT_SHA256" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --manifest "$RWWPO_MANIFEST" \
  --manifest-sha256 "$RWWPO_MANIFEST_SHA256" \
  --work-root "$RWWPO_WORK_ROOT"

mkdir -p "$RWWPO_WORK_ROOT/locks"
exec 8>"$RWWPO_WORK_ROOT/locks/memagent_h20_gpu_${GPU_A}.lock"
flock -n 8 || { echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:first lock conflict' >&2; exit 69; }
exec 9>"$RWWPO_WORK_ROOT/locks/memagent_h20_gpu_${GPU_B}.lock"
flock -n 9 || { echo 'RWWPO2_NUMERIC_ORACLE_NO_GO:second lock conflict' >&2; exit 70; }
for pass in 1 2; do
  active=$(nvidia-smi -i "$GPU_PAIR" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${active//[[:space:]]/} ]] || {
    echo "RWWPO2_NUMERIC_ORACLE_NO_GO:GPU occupied on check $pass; no process changed" >&2
    exit 79
  }
done

cd "$RWWPO_REPO_DIR"
export CUDA_VISIBLE_DEVICES=$GPU_PAIR GPU_PAIR
"$RWWPO_WORK_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node=2 \
  tools/h20/calibrate_rwwpo2_numeric_oracle.py \
  --model "$RWWPO_WORK_ROOT/models/Qwen2.5-7B-Instruct" \
  --expected-commit "$RWWPO_EXPECTED_COMMIT" \
  --output-root "$RWWPO_NUMERIC_ORACLE_ROOT"

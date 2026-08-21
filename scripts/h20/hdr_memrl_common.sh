#!/usr/bin/env bash
set -euo pipefail
HDR_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HDR_REPO=$(cd -- "$HDR_SCRIPT_DIR/../.." && pwd -P)
: "${MEMAGENT_HDR_WORK_ROOT:?set task-scoped MEMAGENT_HDR_WORK_ROOT}"
: "${MEMAGENT_HDR_EXPECTED_COMMIT:?set exact 40-char MEMAGENT_HDR_EXPECTED_COMMIT}"
: "${GPU_PAIR:?set explicit canonical GPU_PAIR=a,b}"
: "${HDR_RUN_ID:?set unique HDR_RUN_ID (letters/digits/dash only)}"
: "${MEMAGENT_HDR_REVIEW_SHA256:?set controller-authorized independent review SHA256}"
: "${MEMAGENT_HDR_BASELINE_BUNDLE_SHA256:?set controller-authorized baseline bundle SHA256}"
[[ $MEMAGENT_HDR_WORK_ROOT == /* && $MEMAGENT_HDR_EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ && $HDR_RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9_-]{5,63}$ ]] || { echo HDR_NO_GO:invalid_environment >&2; exit 64; }
IFS=, read -r HDR_GPU0 HDR_GPU1 <<< "$GPU_PAIR"
[[ $HDR_GPU0 =~ ^[0-9]+$ && $HDR_GPU1 =~ ^[0-9]+$ && $HDR_GPU0 -lt $HDR_GPU1 ]] || { echo HDR_NO_GO:GPU_PAIR_not_canonical >&2; exit 65; }
readonly HDR_ROOT=$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/$HDR_RUN_ID
readonly HDR_CERT=$HDR_ROOT/certificates
readonly HDR_LEDGER=$HDR_ROOT/hdr_execution_ledger.jsonl
: "${HDR_VARIANT:=dro}"
[[ $HDR_VARIANT =~ ^(dro|uniform)$ ]] || { echo HDR_NO_GO:invalid_variant >&2; exit 64; }
if [[ $HDR_VARIANT == dro ]]; then
  readonly HDR_MANIFEST=$HDR_REPO/manifests/h20/qwen25_7b_hdr_memrl_seed2026.json
else
  readonly HDR_MANIFEST=$HDR_REPO/manifests/h20/qwen25_7b_hdr_uniform_erm_seed2026.json
  : "${HDR_AUTHORITY_RUN_ID:?uniform variant requires the authorized DRO HDR_AUTHORITY_RUN_ID}"
fi
readonly HDR_AUTHORITY_ROOT=$MEMAGENT_HDR_WORK_ROOT/logs/hdr_memrl/${HDR_AUTHORITY_RUN_ID:-$HDR_RUN_ID}
readonly HDR_AUTHORITY_CERT=$HDR_AUTHORITY_ROOT/certificates
readonly HDR_EXP=qwen25_7b_hdr_memrl_${HDR_VARIANT}_seed2026_$HDR_RUN_ID
readonly HDR_OUTPUT=$MEMAGENT_HDR_WORK_ROOT/logs/memory_agent/$HDR_EXP
readonly HDR_PYTHON=$MEMAGENT_HDR_WORK_ROOT/.venv/bin/python

hdr_require_checkout() {
  [[ $(git -C "$HDR_REPO" branch --show-current) == h20/qwen25-7b-hdr-memrl-t25-frozen-20260822 ]] || { echo HDR_NO_GO:wrong_branch >&2; exit 66; }
  [[ $(git -C "$HDR_REPO" rev-parse HEAD) == "$MEMAGENT_HDR_EXPECTED_COMMIT" ]] || { echo HDR_NO_GO:wrong_commit >&2; exit 67; }
  [[ -z $(git -C "$HDR_REPO" status --porcelain) ]] || { echo HDR_NO_GO:dirty_tree >&2; exit 68; }
}
hdr_acquire_gpu_locks() {
  command -v flock >/dev/null || { echo HDR_NO_GO:flock_missing >&2; exit 69; }
  mkdir -p "$MEMAGENT_HDR_WORK_ROOT/locks"
  exec 8>"$MEMAGENT_HDR_WORK_ROOT/locks/memagent_h20_gpu_${HDR_GPU0}.lock"
  exec 9>"$MEMAGENT_HDR_WORK_ROOT/locks/memagent_h20_gpu_${HDR_GPU1}.lock"
  flock -n 8 && flock -n 9 || { echo HDR_NO_GO:gpu_lock_conflict >&2; exit 70; }
}
hdr_require_idle() {
  command -v nvidia-smi >/dev/null || { echo HDR_NO_GO:nvidia_smi_missing >&2; exit 71; }
  local apps
  apps=$(nvidia-smi -i "$GPU_PAIR" --query-compute-apps=pid --format=csv,noheader,nounits)
  [[ -z ${apps//[[:space:]]/} ]] || { echo HDR_NO_GO:gpu_occupied_no_process_killed >&2; exit 72; }
}
hdr_require_gates() {
  "$HDR_PYTHON" - "$HDR_CERT" "$HDR_AUTHORITY_CERT" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); authority=Path(sys.argv[2]); expected={"p0.json":"HDR_P0_PASS","e0.json":"HDR_E0_PASS","e1.json":"HDR_E1_PASS","baseline_import.json":"ORIGINAL_BASELINE_IMPORT_PASS","paper_review.json":"PAPER_FRAMING_GO"}
for name,decision in expected.items():
 source=p if name=="p0.json" else authority
 raw=(source/name).read_bytes(); r=json.loads(raw)
 if r.get("status")!="PASS" or r.get("decision")!=decision: raise SystemExit(f"HDR_NO_GO:gate:{name}")
 if name=="baseline_import.json" and r.get("bundle_sha256")!=os.environ["MEMAGENT_HDR_BASELINE_BUNDLE_SHA256"]: raise SystemExit("HDR_NO_GO:baseline_authority_drift")
 if name=="paper_review.json" and hashlib.sha256(raw).hexdigest()!=os.environ["MEMAGENT_HDR_REVIEW_SHA256"]: raise SystemExit("HDR_NO_GO:review_authority_sha_drift")
 if name=="paper_review.json" and (r.get("reviewed_commit")!=os.environ["MEMAGENT_HDR_EXPECTED_COMMIT"] or r.get("reviewer_task")!="independent_adversarial_reviewer" or r.get("paper_framing")!="GO" or r.get("code_release")!="GO" or r.get("blockers")!=[]): raise SystemExit("HDR_NO_GO:unbound_review")
PY
}

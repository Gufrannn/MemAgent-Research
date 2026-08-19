#!/usr/bin/env python3
"""Validate the pre-unblinding checkpoint continuation/stop freeze."""
import argparse,json
from pathlib import Path

ALLOWED=["NaN","unrecoverable_OOM","identity_or_data_corruption","resource_outside_existing_authorization"]
EXPECTED={
  "schema_version":"adaptive-stop-v4","frozen_before_any_anchor_unblinding":True,
  "anchors":[2,25,50,100,200],"terminal_step":200,"post_200_action":"stop",
  "t100_and_intermediate_anchor_use":"post_freeze_screening_only",
  "controls_continuation":False,"metrics_control_continuation":False,
  "allowed_early_stop_reasons":ALLOWED,"select_best_checkpoint_for_confirmatory":False,
  "step_400_automatic":False,"step_400_requires_separate_prefreeze":True,
  "authorizes_new_training":False,
}

def validate(value):
    wrong={key:(value.get(key),expected) for key,expected in EXPECTED.items() if value.get(key)!=expected}
    extra=set(value.get("allowed_early_stop_reasons",[]))-set(ALLOWED)
    if wrong or extra:raise ValueError(f"STOP_RULE_V4_FAIL_CLOSED: wrong={wrong}, extra_stop_reasons={sorted(extra)}")
    return {"status":"STOP_RULE_V4_FROZEN","terminal_step":200,"controls_continuation":False,
      "training_authorized":False,"step_400_authorized":False,"best_checkpoint_selection":False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",type=Path,required=True);args=parser.parse_args()
    print(json.dumps(validate(json.loads(args.manifest.read_text())),indent=2,sort_keys=True))
if __name__=="__main__":main()

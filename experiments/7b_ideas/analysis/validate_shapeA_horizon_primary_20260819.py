#!/usr/bin/env python3
"""Outcome-blind freeze for the sole Shape A horizon/endpoint primary."""
import argparse,json
from pathlib import Path

B0=["checkpoint","turn_type","support_position","length","horizon","truncation_validity","prefrozen_difficulty"]
SECONDARY=["E0_immediate_normalized_f1","E0_immediate_probe","terminal_EM","grounding",
           "distribution_divergence","terminal_minus_immediate"]
EXPECTED={
  "schema_version":"shapeA-horizon-primary-v1","outcomes_accessed":False,"estimand_mode":"EH",
  "endpoint":"terminal_continuous_normalized_f1","harm_definition":"F1_retain_minus_F1_commit",
  "primary_test":"heldout_B0_plus_D_star_vs_B0_continuous_prediction_increment",
  "b0_covariates":B0,"b0_candidate_descendants":False,"b0_post_outcome_fields":False,
  "secondary_endpoints":SECONDARY,"EF_role":"construct_only","VG_role":"independent_closed_loop_study",
  "independent_unit":"stable_example_id","turns_or_writes_increase_n":False,
  "reader_repeats_increase_n":False,"horizon_rows_increase_n":False,
  "crossfit_cluster":"stable_example_id","bootstrap_cluster":"stable_example_id",
  "analysis_row_key":["stable_example_id","write_id"],"write_id_required":True,
  "write_weight":"1/m_i","stable_example_total_weight":1.0,
  "stacked_arm_coding":{"commit":0,"retain":1},"stacked_fixed_effect":"write_pair",
  "stacked_second_evidence":False,"interval_cluster":"stable_example_id",
  "oof_loss_aggregation":"within_example_1_over_m_then_across_examples","row_level_hc3":False,
  "endpoint_estimand_subset_selection_after_results":False,"delete_negative_or_invalid_rows_after_results":False,
  "training_authorized":False,"scientific_outcome_read_authorized":False,
}
FORBIDDEN_OUTCOME_KEYS={"f1_commit","f1_retain","tau","harm","result","p_value","selected_horizon","selected_turn_subset"}

def validate(value):
    wrong={key:(value.get(key),expected) for key,expected in EXPECTED.items() if value.get(key)!=expected}
    leaked=sorted(FORBIDDEN_OUTCOME_KEYS&set(value))
    if wrong or leaked:
        raise ValueError(f"HORIZON_ENDPOINT_SELECTION_INVALID: wrong={wrong}, outcome_fields={leaked}")
    return {"status":"SHAPEA_EH_PRIMARY_FROZEN","estimand":"EH","endpoint":"terminal_continuous_normalized_f1",
      "independent_unit":"stable_example_id","outcomes_read":False,"training_authorized":False,
      "scientific_outcome_read_authorized":False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",type=Path,required=True);args=parser.parse_args()
    print(json.dumps(validate(json.loads(args.manifest.read_text())),indent=2,sort_keys=True))
if __name__=="__main__":main()

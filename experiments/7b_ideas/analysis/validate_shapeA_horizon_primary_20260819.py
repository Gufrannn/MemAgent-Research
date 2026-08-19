#!/usr/bin/env python3
"""Outcome-blind freeze for the sole Shape A horizon/endpoint primary."""
import argparse,json
from pathlib import Path

B0=["checkpoint","turn","support_position","pre_state_length","chunk_length","old_memory_length","horizon","pre_action_difficulty"]
FORBIDDEN_B0=["candidate_length","compression_edit","candidate_validity","factual_commit_complete",
 "noop_retain_complete","pair_qualified","arm_truncation","failure_reason"]
ATTRITION=["stable_example_id","checkpoint_hash","write_id","target_write_count","measurement_available",
 "measurement_exclusion_reason","eligible_write_count","eligible","attempted",
 "factual_commit_complete","noop_retain_complete","pair_qualified","failure_reason","fixed_paired_weight"]
SECONDARY=["E0_immediate_normalized_f1","E0_immediate_probe","terminal_EM","grounding",
           "distribution_divergence","terminal_minus_immediate"]
EXPECTED={
  "schema_version":"shapeA-horizon-primary-v3","outcomes_accessed":False,"estimand_mode":"EH",
  "endpoint":"terminal_continuous_normalized_f1","harm_definition":"F1_retain_minus_F1_commit",
  "primary_test":"heldout_B0_plus_D_star_vs_B0_continuous_prediction_increment",
  "b0_covariates":B0,"b0_candidate_descendants":False,"b0_post_outcome_fields":False,
  "b0_decision_time":"T0_pre_candidate_pre_branch","b0_forbidden_postcandidate_fields":FORBIDDEN_B0,
  "secondary_endpoints":SECONDARY,"EF_role":"construct_only","VG_role":"independent_closed_loop_study",
  "independent_unit":"stable_example_id","turns_or_writes_increase_n":False,
  "reader_repeats_increase_n":False,"horizon_rows_increase_n":False,
  "crossfit_cluster":"stable_example_id","bootstrap_cluster":"stable_example_id",
  "analysis_row_key":["stable_example_id","checkpoint_hash","write_id"],"write_id_required":True,
  "write_weight":"1/m_i^elig","stable_example_total_weight":1.0,
  "stacked_arm_coding":{"commit":0,"retain":1},"stacked_fixed_effect":"write_pair",
  "stacked_second_evidence":False,"interval_cluster":"stable_example_id",
  "oof_loss_aggregation":"within_example_1_over_m_then_across_examples","row_level_hc3":False,
  "eligible_write_count_source":"prebranch_manifest","eligible_write_weight":"1/m_i^elig",
  "target_write_count_source":"initial_outcome_blind_turn_selection_manifest",
  "measurement_availability_report":True,"r0_role":"prebranch_measurement_availability_selection_not_arm_failure",
  "paired_denominator_population":"R1_measurable_prebranch_events","paired_closure_relative_to":"m_i^elig",
  "report_separate_target_and_paired_denominators":True,"combined_coverage_only":False,
  "qualified_count_replaces_eligible_count":False,
  "eligible_weight_coverage_required":{"pilot4":1.0,"audit32":1.0,"B128":1.0},
  "postbranch_missing_action":"construction_diagnostic_only","missing_weight_reallocation":False,
  "complete_case_primary":False,"ipw_primary":False,"imputation_primary":False,
  "attrition_ledger_fields":ATTRITION,"rerun_eligibility_manifest":"same_frozen_manifest",
  "rerun_experiment_name":"new_unique","retain_old_ledger":True,
  "primary_outcome_symbol":"H_H","primary_outcome_direction":"Y_retain_minus_Y_commit",
  "primary_higher_means":"more_harmful","dstar_higher_means":"worse",
  "tau_H_role":"secondary_opposite_direction_alias_only","tau_H_equals":"minus_H_H",
  "mixed_tau_H_and_H_H_primary":False,"legacy_valid_n_96_rescue":False,
  "dstar_calibration":"frozen_turn_type_x_required_component_pattern_joint_null_median_mad",
  "dstar_clip":[-5,5],"dstar_truncate_at_zero":False,"legacy_max0_w_minus_q95":False,
  "q95_role":"secondary_anomaly_flag_only","missing_required_component_policy":"row_invalid_no_zero_fill",
  "insufficient_joint_null_yield_policy":"no_cross_turn_scale_borrowing",
  "endpoint_estimand_subset_selection_after_results":False,"delete_negative_or_invalid_rows_after_results":False,
  "training_authorized":False,"scientific_outcome_read_authorized":False,
}
FORBIDDEN_OUTCOME_KEYS={"f1_commit","f1_retain","tau","harm","result","p_value","selected_horizon","selected_turn_subset"}

def validate(value):
    wrong={key:(value.get(key),expected) for key,expected in EXPECTED.items() if value.get(key)!=expected}
    leaked=sorted(FORBIDDEN_OUTCOME_KEYS&set(value));b0_leak=sorted(set(value.get("b0_covariates",[]))&set(FORBIDDEN_B0))
    if wrong or leaked or b0_leak:
        raise ValueError(f"HORIZON_ENDPOINT_SELECTION_INVALID: wrong={wrong}, outcome_fields={leaked}, b0_postbranch={b0_leak}")
    return {"status":"SHAPEA_EH_PRIMARY_FROZEN","estimand":"EH","endpoint":"terminal_continuous_normalized_f1",
      "independent_unit":"stable_example_id","outcomes_read":False,"training_authorized":False,
      "scientific_outcome_read_authorized":False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",type=Path,required=True);args=parser.parse_args()
    print(json.dumps(validate(json.loads(args.manifest.read_text())),indent=2,sort_keys=True))
if __name__=="__main__":main()

#!/usr/bin/env python
"""
Tree-structure robustness runs for delayed-action analysis.

This script follows the same SGE array-job architecture as
main_ensemble_delayed_cluster.py. Each task maps to one
(delay_year, tree_spec) combination at research_runs.csv row 0 and writes raw CSV outputs
only. Plotting and summary tables are handled later in the notebook.
"""

import os
import sys

import numpy as np

from _project_paths import configure_paths
configure_paths()

import main_ensemble_delayed_cluster as ensemble
from src.config import GAUSSIAN_PRIOR_SET_NAME, PARAMETER_PRIOR_DIMS, RUN0_PARAMETER_VALUES


output_folder = "tree-robustness-analysis"

# Default robustness grid:
# row-0 parameter vector * 3 delay years * 5 tree specs = 15 tasks.
DEFAULT_TREE_SPEC_NAMES = [
    'default',
    'front_loaded_decisions',
    'back_loaded_decisions',
    'higher_fragility_weight',
    'lower_fragility_weight',
]

TREE_SPECS = {
    'default': {
        'decision_times': ensemble.DEFAULT_DECISION_TIMES.copy(),
        'prob_scale': 1.0,
    },
    'standard_10yr_second_decision': {
        'decision_times': [0, 10, 35, 75, 125, 175, 225],
        'prob_scale': 1.0,
    },
    'front_loaded_decisions': {
        'decision_times': [0, 5, 20, 50, 100, 160, 225],
        'prob_scale': 1.0,
    },
    'back_loaded_decisions': {
        'decision_times': [0, 5, 50, 100, 150, 200, 225],
        'prob_scale': 1.0,
    },
    'higher_fragility_weight': {
        'decision_times': ensemble.DEFAULT_DECISION_TIMES.copy(),
        'prob_scale': 0.75,
    },
    'lower_fragility_weight': {
        'decision_times': ensemble.DEFAULT_DECISION_TIMES.copy(),
        'prob_scale': 1.5,
    },
}


def selected_tree_spec_names():
    env_value = os.environ.get('ROBUSTNESS_TREE_SPECS')
    if env_value:
        names = [name.strip() for name in env_value.split(',') if name.strip()]
    else:
        names = DEFAULT_TREE_SPEC_NAMES

    unknown = [name for name in names if name not in TREE_SPECS]
    if unknown:
        raise ValueError(f"Unknown tree specs: {unknown}. Valid specs: {list(TREE_SPECS.keys())}")

    return names


def get_cluster_config():
    sge_task_id = os.environ.get('SGE_TASK_ID')
    if sge_task_id is None:
        print("ERROR: SGE_TASK_ID environment variable not found!")
        print("This script is designed to run as part of an SGE array job.")
        sys.exit(1)

    try:
        task_id = int(sge_task_id)
    except ValueError:
        print(f"ERROR: Invalid SGE_TASK_ID: {sge_task_id}")
        sys.exit(1)

    tree_names = selected_tree_spec_names()
    num_delays = len(ensemble.delay_years)
    num_trees = len(tree_names)
    total_combinations = num_delays * num_trees
    task_index = task_id - 1

    if task_index >= total_combinations:
        print(f"Error: Task ID {task_id} exceeds total combinations ({total_combinations})")
        print(f"delay_years = {ensemble.delay_years}")
        print(f"tree_specs = {tree_names}")
        print(f"Expected task range: 1-{total_combinations}")
        sys.exit(1)

    delay_idx = task_index // num_trees
    tree_idx = task_index % num_trees

    delay_year = ensemble.delay_years[delay_idx]
    tree_spec = tree_names[tree_idx]
    out_folder = os.environ.get('OUTPUT_FOLDER', output_folder)
    baseline = int(os.environ.get('BASELINE_NUM', ensemble.baseline_num))
    job_id = os.environ.get('JOB_ID', 'Unknown')

    print("\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {total_combinations}")
    print(f"  Mapping: run=0, delay_year={delay_year}, tree_spec={tree_spec}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")
    print(f"  Output folder: {out_folder}")
    print(f"  Baseline (SSP): {baseline}")

    return 0, task_id, delay_year, tree_spec, out_folder, baseline


def get_tree_decision_times(tree_spec, delay_year):
    spec = TREE_SPECS[tree_spec]
    decision_times_delay = list(spec['decision_times'])
    if delay_year > 0:
        decision_times_delay[1] = delay_year
    decision_times_baseline = decision_times_delay.copy()
    return decision_times_baseline, decision_times_delay


def main():
    print("\nEZClimate TREE ROBUSTNESS DELAYED ACTION ANALYSIS - CLUSTER ARRAY JOB\n")

    sample_index, task_id, delay_year, tree_spec, out_folder, baseline = get_cluster_config()
    ensemble.setup_cluster_directories(out_folder)

    param_vals = np.atleast_2d(RUN0_PARAMETER_VALUES)

    if task_id == 1:
        samples_copy = os.path.join(
            ensemble.DATA_DIR, out_folder, 'samples',
            f'tree_robustness_run0_params_DIMS{PARAMETER_PRIOR_DIMS}_{GAUSSIAN_PRIOR_SET_NAME}.csv'
        )
        np.savetxt(samples_copy, param_vals, delimiter=',')
        print(f"\nSaved row-0 parameter values to: {samples_copy}")

    decision_times_baseline, decision_times_delay = get_tree_decision_times(tree_spec, delay_year)
    prob_scale = TREE_SPECS[tree_spec]['prob_scale']

    print(f"\nExecution Configuration:")
    print(f"  Test mode:       {ensemble.test_mode}")
    print(f"  Import damages:  {ensemble.import_damages}")
    print(f"  Baseline (SSP):  {baseline}")
    print(f"  Tree spec:       {tree_spec}")
    print(f"  Baseline times:  {decision_times_baseline}")
    print(f"  Delayed times:   {decision_times_delay}")
    print(f"  Probability scale: {prob_scale}")

    try:
        ensemble.run_ensemble_delayed_analysis(
            sample_index=sample_index,
            delay_year=delay_year,
            param_vals=param_vals,
            out_folder=out_folder,
            baseline=baseline,
            test_mode=ensemble.test_mode,
            import_damages=ensemble.import_damages,
            run_type='tree_robustness',
            tree_spec=tree_spec,
            decision_times_baseline=decision_times_baseline,
            decision_times_delay=decision_times_delay,
            prob_scale_baseline=prob_scale,
            prob_scale_delay=prob_scale,
        )
    except Exception as e:
        print(f"ERROR running sample {sample_index}, delay {delay_year}, tree {tree_spec}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nTASK COMPLETE: Run 0 (delay {delay_year}, tree {tree_spec})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

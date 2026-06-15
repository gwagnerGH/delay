#!/usr/bin/env python
"""
Run delayed-action scenarios at the research_runs.csv row-0 parameter vector.

This script follows the same SGE array-job architecture as
main_ensemble_delayed_cluster.py, but maps each task only to a delay year.
It writes raw machine-readable outputs only; figures and summary tables are
made later in notebooks/professor_response_analysis.ipynb.
"""

import os
import sys
import numpy as np

from _project_paths import configure_paths
configure_paths()

import main_ensemble_delayed_cluster as ensemble
from src.config import GAUSSIAN_PRIOR_SET_NAME, PARAMETER_PRIOR_DIMS, RUN0_PARAMETER_VALUES


output_folder = "mean-parameter-analysis"


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

    total_combinations = len(ensemble.delay_years)
    task_index = task_id - 1

    if task_index >= total_combinations:
        print(f"Error: Task ID {task_id} exceeds total combinations ({total_combinations})")
        print(f"delay_years = {ensemble.delay_years}")
        print(f"Expected task range: 1-{total_combinations}")
        sys.exit(1)

    delay_year = ensemble.delay_years[task_index]
    out_folder = os.environ.get('OUTPUT_FOLDER', output_folder)
    baseline = int(os.environ.get('BASELINE_NUM', ensemble.baseline_num))
    job_id = os.environ.get('JOB_ID', 'Unknown')

    print("\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {total_combinations}")
    print(f"  Mapping: mean parameter vector, delay_year={delay_year}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")
    print(f"  Output folder: {out_folder}")
    print(f"  Baseline (SSP): {baseline}")

    return task_id, delay_year, out_folder, baseline


def mean_parameter_values():
    return np.atleast_2d(RUN0_PARAMETER_VALUES)


def main():
    print("\nEZClimate MEAN-PARAMETER DELAYED ACTION ANALYSIS - CLUSTER ARRAY JOB\n")

    task_id, delay_year, out_folder, baseline = get_cluster_config()
    ensemble.setup_cluster_directories(out_folder)

    param_vals = mean_parameter_values()

    if task_id == 1:
        samples_copy = os.path.join(
            ensemble.DATA_DIR, out_folder, 'samples',
            f'Gaussian_prior_means_DIMS{PARAMETER_PRIOR_DIMS}_{GAUSSIAN_PRIOR_SET_NAME}.csv'
        )
        np.savetxt(samples_copy, param_vals, delimiter=',')
        print(f"\nSaved mean parameter vector to: {samples_copy}")

    print(f"\nExecution Configuration:")
    print(f"  Test mode:       {ensemble.test_mode}")
    print(f"  Import damages:  {ensemble.import_damages}")
    print(f"  Baseline (SSP):  {baseline}")

    try:
        ensemble.run_ensemble_delayed_analysis(
            sample_index=0,
            delay_year=delay_year,
            param_vals=param_vals,
            out_folder=out_folder,
            baseline=baseline,
            test_mode=ensemble.test_mode,
            import_damages=ensemble.import_damages,
            run_type='mean_parameter_vector',
            tree_spec='default',
            sample_label='mean_params',
        )
    except Exception as e:
        print(f"ERROR running mean-parameter scenario with delay {delay_year}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nTASK COMPLETE: Mean parameter vector (delay {delay_year})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

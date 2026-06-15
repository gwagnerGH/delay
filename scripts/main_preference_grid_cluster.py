#!/usr/bin/env python
"""
Deterministic preference-grid robustness runs for delayed-action analysis.

Each SGE task maps to one (preference-grid point, delay_year) combination.
The grid varies RA, EIS, and PRTP while holding technology, backstop premium,
and growth at the research_runs.csv row-0 values. Outputs use the same consolidated
schema as main_ensemble_delayed_cluster.py.
"""

import os
import sys

import numpy as np

from _project_paths import configure_paths
configure_paths()

import main_ensemble_delayed_cluster as ensemble
from src.config import (
    GAUSSIAN_PRIOR_SET_NAME,
    PARAMETER_PRIOR_INDEX,
    PARAMETER_PRIOR_LOWER_BOUNDS,
    PARAMETER_PRIOR_MEANS,
    PARAMETER_PRIOR_UPPER_BOUNDS,
    RUN0_PARAMETER_VALUES,
)


output_folder = "preference-grid-analysis"


def parse_levels(env_name, default_values):
    env_value = os.environ.get(env_name)
    if not env_value:
        return list(default_values)
    return [float(value.strip()) for value in env_value.split(',') if value.strip()]


def default_levels(name):
    index = PARAMETER_PRIOR_INDEX[name]
    return [
        PARAMETER_PRIOR_LOWER_BOUNDS[index],
        PARAMETER_PRIOR_MEANS[index],
        PARAMETER_PRIOR_UPPER_BOUNDS[index],
    ]


def preference_grid_values():
    ra_levels = parse_levels('PREFERENCE_RA_LEVELS', default_levels('RA'))
    eis_levels = parse_levels('PREFERENCE_EIS_LEVELS', default_levels('EIS'))
    prtp_levels = parse_levels('PREFERENCE_PRTP_LEVELS', default_levels('PRTP'))

    values = []
    labels = []
    for ra in ra_levels:
        for eis in eis_levels:
            for prtp in prtp_levels:
                row = RUN0_PARAMETER_VALUES.copy()
                row[PARAMETER_PRIOR_INDEX['RA']] = ra
                row[PARAMETER_PRIOR_INDEX['EIS']] = eis
                row[PARAMETER_PRIOR_INDEX['PRTP']] = prtp
                values.append(row)
                labels.append(f"RA{ra:g}_EIS{eis:g}_PRTP{prtp:g}")

    return np.asarray(values, dtype=float), labels, {
        'RA': ra_levels,
        'EIS': eis_levels,
        'PRTP': prtp_levels,
    }


def get_cluster_config(param_vals, labels, levels):
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

    num_delays = len(ensemble.delay_years)
    total_combinations = len(param_vals) * num_delays
    task_index = task_id - 1

    if task_index >= total_combinations:
        print(f"ERROR: Task ID {task_id} exceeds total combinations ({total_combinations})")
        print(f"Preference levels = {levels}")
        print(f"delay_years = {ensemble.delay_years}")
        print(f"Expected task range: 1-{total_combinations}")
        sys.exit(1)

    grid_index = task_index // num_delays
    delay_idx = task_index % num_delays
    delay_year = ensemble.delay_years[delay_idx]
    out_folder = os.environ.get('OUTPUT_FOLDER', output_folder)
    baseline = int(os.environ.get('BASELINE_NUM', ensemble.baseline_num))
    job_id = os.environ.get('JOB_ID', 'Unknown')

    print("\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {total_combinations}")
    print(f"  Mapping: preference_grid_index={grid_index}, delay_year={delay_year}")
    print(f"  Label: {labels[grid_index]}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")
    print(f"  Output folder: {out_folder}")
    print(f"  Baseline (SSP): {baseline}")
    print(f"  Preference levels: {levels}")

    return grid_index, task_id, delay_year, out_folder, baseline


def main():
    print("\nEZClimate PREFERENCE-GRID DELAYED ACTION ANALYSIS - CLUSTER ARRAY JOB\n")

    param_vals, labels, levels = preference_grid_values()
    grid_index, task_id, delay_year, out_folder, baseline = get_cluster_config(
        param_vals, labels, levels
    )
    ensemble.setup_cluster_directories(out_folder)

    if task_id == 1:
        samples_copy = os.path.join(
            ensemble.DATA_DIR, out_folder, 'samples',
            f'preference_grid_N{len(param_vals)}_{GAUSSIAN_PRIOR_SET_NAME}.csv'
        )
        np.savetxt(samples_copy, param_vals, delimiter=',')
        labels_copy = os.path.join(
            ensemble.DATA_DIR, out_folder, 'samples',
            f'preference_grid_labels_N{len(param_vals)}_{GAUSSIAN_PRIOR_SET_NAME}.csv'
        )
        with open(labels_copy, 'w') as f:
            f.write('grid_index,label\n')
            for index, label in enumerate(labels):
                f.write(f'{index},{label}\n')
        print(f"\nSaved preference grid values to: {samples_copy}")
        print(f"Saved preference grid labels to: {labels_copy}")

    print("\nExecution Configuration:")
    print(f"  Test mode:       {ensemble.test_mode}")
    print(f"  Import damages:  {ensemble.import_damages}")
    print(f"  Baseline (SSP):  {baseline}")

    try:
        ensemble.run_ensemble_delayed_analysis(
            sample_index=grid_index,
            delay_year=delay_year,
            param_vals=param_vals,
            out_folder=out_folder,
            baseline=baseline,
            test_mode=ensemble.test_mode,
            import_damages=ensemble.import_damages,
            run_type='preference_grid',
            tree_spec='default',
            sample_label=f'pref_grid_{grid_index:03d}',
        )
    except Exception as e:
        print(f"ERROR running preference-grid scenario {grid_index}, delay {delay_year}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nTASK COMPLETE: Preference grid {grid_index} (delay {delay_year})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python
"""
Technology-mechanism robustness runs for delayed-action analysis.

Each SGE task maps to one (technology-grid point, delay_year) combination.
The grid varies exogenous technical progress and endogenous learning while
holding preferences, backstop premium, and growth at research_runs.csv row-0 values.
Outputs use the same consolidated schema as main_ensemble_delayed_cluster.py.
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


output_folder = "technology-grid-analysis"


def parse_levels(env_name, default_values):
    env_value = os.environ.get(env_name)
    if not env_value:
        return list(default_values)
    return [float(value.strip()) for value in env_value.split(',') if value.strip()]


def technology_grid_values():
    tech_chg_index = PARAMETER_PRIOR_INDEX['tech_chg']
    tech_scale_index = PARAMETER_PRIOR_INDEX['tech_scale']
    tech_chg_levels = parse_levels(
        'TECH_CHG_LEVELS',
        [
            PARAMETER_PRIOR_LOWER_BOUNDS[tech_chg_index],
            PARAMETER_PRIOR_MEANS[tech_chg_index],
            PARAMETER_PRIOR_UPPER_BOUNDS[tech_chg_index],
        ],
    )
    tech_scale_levels = parse_levels(
        'TECH_SCALE_LEVELS',
        [
            PARAMETER_PRIOR_LOWER_BOUNDS[tech_scale_index],
            PARAMETER_PRIOR_MEANS[tech_scale_index],
            PARAMETER_PRIOR_UPPER_BOUNDS[tech_scale_index],
        ],
    )

    values = []
    labels = []
    for tech_chg in tech_chg_levels:
        for tech_scale in tech_scale_levels:
            row = RUN0_PARAMETER_VALUES.copy()
            row[tech_chg_index] = tech_chg
            row[tech_scale_index] = tech_scale
            values.append(row)
            labels.append(f"tech_chg{tech_chg:g}_tech_scale{tech_scale:g}")

    return np.asarray(values, dtype=float), labels, {
        'tech_chg': tech_chg_levels,
        'tech_scale': tech_scale_levels,
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
        print(f"Technology levels = {levels}")
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
    print(f"  Mapping: technology_grid_index={grid_index}, delay_year={delay_year}")
    print(f"  Label: {labels[grid_index]}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")
    print(f"  Output folder: {out_folder}")
    print(f"  Baseline (SSP): {baseline}")
    print(f"  Technology levels: {levels}")

    return grid_index, task_id, delay_year, out_folder, baseline


def main():
    print("\nEZClimate TECHNOLOGY-GRID DELAYED ACTION ANALYSIS - CLUSTER ARRAY JOB\n")

    param_vals, labels, levels = technology_grid_values()
    grid_index, task_id, delay_year, out_folder, baseline = get_cluster_config(
        param_vals, labels, levels
    )
    ensemble.setup_cluster_directories(out_folder)

    if task_id == 1:
        samples_copy = os.path.join(
            ensemble.DATA_DIR, out_folder, 'samples',
            f'technology_grid_N{len(param_vals)}_{GAUSSIAN_PRIOR_SET_NAME}.csv'
        )
        np.savetxt(samples_copy, param_vals, delimiter=',')
        labels_copy = os.path.join(
            ensemble.DATA_DIR, out_folder, 'samples',
            f'technology_grid_labels_N{len(param_vals)}_{GAUSSIAN_PRIOR_SET_NAME}.csv'
        )
        with open(labels_copy, 'w') as f:
            f.write('grid_index,label\n')
            for index, label in enumerate(labels):
                f.write(f'{index},{label}\n')
        print(f"\nSaved technology grid values to: {samples_copy}")
        print(f"Saved technology grid labels to: {labels_copy}")

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
            run_type='technology_grid',
            tree_spec='default',
            sample_label=f'tech_grid_{grid_index:03d}',
        )
    except Exception as e:
        print(f"ERROR running technology-grid scenario {grid_index}, delay {delay_year}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nTASK COMPLETE: Technology grid {grid_index} (delay {delay_year})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

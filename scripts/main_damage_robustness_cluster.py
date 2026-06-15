#!/usr/bin/env python
"""
Damage-specification robustness runs for delayed-action analysis.

Each SGE task maps to one (damage specification, delay_year) combination. By
default the runner varies the paper's main damage specification plus the three
pure damage functions, each with and without tipping points. Outputs use the
shared consolidated schema from main_ensemble_delayed_cluster.py.
"""

import os
import sys

import numpy as np

from _project_paths import configure_paths
configure_paths()

import main_ensemble_delayed_cluster as ensemble
from src.config import GAUSSIAN_PRIOR_SET_NAME, PARAMETER_PRIOR_DIMS, RUN0_PARAMETER_VALUES


output_folder = "damage-robustness-analysis"


def parse_int_levels(env_name, default_values):
    env_value = os.environ.get(env_name)
    if not env_value:
        return list(default_values)
    return [int(value.strip()) for value in env_value.split(',') if value.strip()]


def damage_specs():
    dam_funcs = parse_int_levels('DAMAGE_DAM_FUNCS', [0, 1, 2, 3])
    tip_on_levels = parse_int_levels('DAMAGE_TIP_ON_LEVELS', [ensemble.tip_on, 0])
    d_unc_levels = parse_int_levels('DAMAGE_D_UNC_LEVELS', [ensemble.d_unc])
    t_unc_levels = parse_int_levels('DAMAGE_T_UNC_LEVELS', [ensemble.t_unc])

    specs = []
    for dam_func in dam_funcs:
        for tip_on in tip_on_levels:
            for d_unc in d_unc_levels:
                for t_unc in t_unc_levels:
                    specs.append({
                        'dam_func': dam_func,
                        'tip_on': tip_on,
                        'd_unc': d_unc,
                        't_unc': t_unc,
                        'label': f"df{dam_func}_TP{tip_on}_dunc{d_unc}_tunc{t_unc}",
                    })
    return specs


def get_cluster_config(specs):
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
    total_combinations = len(specs) * num_delays
    task_index = task_id - 1

    if task_index >= total_combinations:
        print(f"ERROR: Task ID {task_id} exceeds total combinations ({total_combinations})")
        print(f"damage_specs = {[spec['label'] for spec in specs]}")
        print(f"delay_years = {ensemble.delay_years}")
        print(f"Expected task range: 1-{total_combinations}")
        sys.exit(1)

    spec_index = task_index // num_delays
    delay_idx = task_index % num_delays
    delay_year = ensemble.delay_years[delay_idx]
    out_folder = os.environ.get('OUTPUT_FOLDER', output_folder)
    baseline = int(os.environ.get('BASELINE_NUM', ensemble.baseline_num))
    job_id = os.environ.get('JOB_ID', 'Unknown')
    spec = specs[spec_index]

    print("\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {total_combinations}")
    print(f"  Mapping: damage_spec_index={spec_index}, delay_year={delay_year}")
    print(f"  Damage spec: {spec['label']}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")
    print(f"  Output folder: {out_folder}")
    print(f"  Baseline (SSP): {baseline}")

    return spec_index, task_id, delay_year, out_folder, baseline


def mean_parameter_values():
    return np.atleast_2d(RUN0_PARAMETER_VALUES)


def apply_damage_spec(spec):
    ensemble.dam_func = int(spec['dam_func'])
    ensemble.tip_on = int(spec['tip_on'])
    ensemble.d_unc = int(spec['d_unc'])
    ensemble.t_unc = int(spec['t_unc'])


def main():
    print("\nEZClimate DAMAGE ROBUSTNESS DELAYED ACTION ANALYSIS - CLUSTER ARRAY JOB\n")

    specs = damage_specs()
    spec_index, task_id, delay_year, out_folder, baseline = get_cluster_config(specs)
    spec = specs[spec_index]
    apply_damage_spec(spec)
    ensemble.setup_cluster_directories(out_folder)

    param_vals = mean_parameter_values()

    if task_id == 1:
        samples_copy = os.path.join(
            ensemble.DATA_DIR, out_folder, 'samples',
            f'damage_robustness_mean_params_DIMS{PARAMETER_PRIOR_DIMS}_{GAUSSIAN_PRIOR_SET_NAME}.csv'
        )
        np.savetxt(samples_copy, param_vals, delimiter=',')
        spec_copy = os.path.join(
            ensemble.DATA_DIR, out_folder, 'samples',
            f'damage_robustness_specs_{GAUSSIAN_PRIOR_SET_NAME}.csv'
        )
        with open(spec_copy, 'w') as f:
            f.write('damage_spec_index,label,dam_func,tip_on,d_unc,t_unc\n')
            for index, row in enumerate(specs):
                f.write(
                    f"{index},{row['label']},{row['dam_func']},"
                    f"{row['tip_on']},{row['d_unc']},{row['t_unc']}\n"
                )
        print(f"\nSaved damage robustness parameter values to: {samples_copy}")
        print(f"Saved damage robustness specs to: {spec_copy}")

    print("\nExecution Configuration:")
    print(f"  Test mode:       {ensemble.test_mode}")
    print(f"  Import damages:  {ensemble.import_damages}")
    print(f"  Baseline (SSP):  {baseline}")
    print(f"  Damage spec:     {spec['label']}")

    try:
        ensemble.run_ensemble_delayed_analysis(
            sample_index=0,
            delay_year=delay_year,
            param_vals=param_vals,
            out_folder=out_folder,
            baseline=baseline,
            test_mode=ensemble.test_mode,
            import_damages=ensemble.import_damages,
            run_type='damage_robustness',
            tree_spec='default',
            comparison_type='same_grid',
            sample_label=f"damage_{spec['label']}",
        )
    except Exception as e:
        print(f"ERROR running damage robustness spec {spec['label']}, delay {delay_year}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nTASK COMPLETE: Damage robustness {spec['label']} (delay {delay_year})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

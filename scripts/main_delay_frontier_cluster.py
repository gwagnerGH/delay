#!/usr/bin/env python
"""
Delay-frontier cluster runner.

The default annual-grid mode runs integer-year delays 0, 1, ..., 20. Set
FRONTIER_GRID=five_year to run the native 5-year model grid with delays
0, 5, 10, 15, 20. The 5-year mode is directly comparable to the partial
mitigation and main robustness runs.

By default this runs the research_runs.csv row-0 parameter vector so the frontier
is a compact main-specification mechanism run. Set FRONTIER_PARAMETER_SOURCE to
"ensemble" and FRONTIER_N_SAMPLES to use Gaussian ensemble draws instead.
"""

import os
import sys

import numpy as np

from _project_paths import configure_paths
configure_paths()

import main_ensemble_delayed_cluster as ensemble
from src.config import (
    DEFAULT_BASE_YEAR,
    DEFAULT_CALENDAR_YEARS,
    DEFAULT_DECISION_TIMES,
    GAUSSIAN_PRIOR_SET_NAME,
    PARAMETER_PRIOR_DIMS,
    PARAMETER_PRIOR_INDEX,
    PARAMETER_PRIOR_LOWER_BOUNDS,
    PARAMETER_PRIOR_MEANS,
    PARAMETER_PRIOR_UPPER_BOUNDS,
    RUN0_PARAMETER_VALUES,
)


output_folder = "delay-frontier-analysis"

FRONTIER_GRID = os.environ.get('FRONTIER_GRID', 'annual').strip().lower()
if FRONTIER_GRID in ('5', '5yr', '5-year', 'five-year', 'five_year', 'native'):
    FRONTIER_GRID = 'five_year'
elif FRONTIER_GRID in ('1', '1yr', '1-year', 'annual', 'yearly'):
    FRONTIER_GRID = 'annual'
else:
    raise ValueError(
        "FRONTIER_GRID must be 'annual' or 'five_year' "
        f"(got {FRONTIER_GRID!r})."
    )

FRONTIER_GRID_CONFIG = {
    'annual': {
        'period_len': 1.0,
        'emissions_time_step': 1,
        'damage_file_tag': '_GRID1',
        'default_delays': list(range(0, 21)),
    },
    'five_year': {
        'period_len': 5.0,
        'emissions_time_step': None,
        'damage_file_tag': '_GRID5',
        'default_delays': list(range(0, 21, 5)),
    },
}
FRONTIER_PERIOD_LEN = FRONTIER_GRID_CONFIG[FRONTIER_GRID]['period_len']
FRONTIER_EMISSIONS_TIME_STEP = FRONTIER_GRID_CONFIG[FRONTIER_GRID]['emissions_time_step']
FRONTIER_DAMAGE_FILE_TAG = FRONTIER_GRID_CONFIG[FRONTIER_GRID]['damage_file_tag']
DEFAULT_FRONTIER_DELAYS = FRONTIER_GRID_CONFIG[FRONTIER_GRID]['default_delays']
FRONTIER_PARAMETER_SOURCE = os.environ.get(
    'FRONTIER_PARAMETER_SOURCE', 'mean'
).strip().lower()
FRONTIER_N_SAMPLES = int(os.environ.get('FRONTIER_N_SAMPLES', '1'))
FRONTIER_SAMPLE_OFFSET = int(os.environ.get('FRONTIER_SAMPLE_OFFSET', '0'))
DEFAULT_FRONTIER_PARAMETER_SPECS = [
    'low_eis',
    'high_eis',
    'high_ra',
    'low_ra',
    'no_endogenous_learning',
]


def frontier_delay_years():
    env_value = os.environ.get('DELAY_YEARS')
    if not env_value:
        delays = DEFAULT_FRONTIER_DELAYS
    else:
        delays = [int(value.strip()) for value in env_value.split(',') if value.strip()]

    invalid_integer = [
        delay for delay in delays
        if not isinstance(delay, int)
    ]
    if invalid_integer:
        raise ValueError(
            "Delay-frontier years must be integer-year delays. "
            f"Invalid delay years: {invalid_integer}"
        )

    invalid_order = [
        delay for delay in delays
        if delay < 0 or delay >= DEFAULT_DECISION_TIMES[2]
    ]
    if invalid_order:
        raise ValueError(
            "Delay-frontier years must be nonnegative and earlier than the "
            f"next baseline decision time ({DEFAULT_DECISION_TIMES[2]}). "
            f"Invalid delay years: {invalid_order}"
        )

    if FRONTIER_GRID == 'five_year':
        off_grid = [delay for delay in delays if delay % 5 != 0]
        if off_grid:
            raise ValueError(
                "FRONTIER_GRID=five_year only supports 5-year-grid delays. "
                f"Invalid delay years: {off_grid}"
            )

    return delays


def parse_frontier_parameter_specs():
    env_value = os.environ.get('FRONTIER_PARAMETER_SPECS')
    if not env_value:
        return list(DEFAULT_FRONTIER_PARAMETER_SPECS)
    return [value.strip().lower() for value in env_value.split(',') if value.strip()]


def parameter_spec_values():
    specs = parse_frontier_parameter_specs()
    values = []
    labels = []

    for spec in specs:
        row = RUN0_PARAMETER_VALUES.copy()
        if spec == 'low_eis':
            row[PARAMETER_PRIOR_INDEX['EIS']] = PARAMETER_PRIOR_LOWER_BOUNDS[PARAMETER_PRIOR_INDEX['EIS']]
            label = spec
        elif spec == 'high_eis':
            row[PARAMETER_PRIOR_INDEX['EIS']] = PARAMETER_PRIOR_UPPER_BOUNDS[PARAMETER_PRIOR_INDEX['EIS']]
            label = spec
        elif spec == 'high_ra':
            row[PARAMETER_PRIOR_INDEX['RA']] = PARAMETER_PRIOR_UPPER_BOUNDS[PARAMETER_PRIOR_INDEX['RA']]
            label = spec
        elif spec == 'low_ra':
            row[PARAMETER_PRIOR_INDEX['RA']] = PARAMETER_PRIOR_LOWER_BOUNDS[PARAMETER_PRIOR_INDEX['RA']]
            label = spec
        elif spec in ('no_endogenous_learning', 'no_learning'):
            row[PARAMETER_PRIOR_INDEX['tech_scale']] = 0.0
            label = 'no_endogenous_learning'
        else:
            raise ValueError(
                "Unknown FRONTIER_PARAMETER_SPECS entry "
                f"{spec!r}. Known values: {DEFAULT_FRONTIER_PARAMETER_SPECS}"
            )
        values.append(row)
        labels.append(label)

    return np.asarray(values, dtype=float), labels


def parameter_values_and_labels():
    if FRONTIER_PARAMETER_SOURCE == 'mean':
        return np.atleast_2d(RUN0_PARAMETER_VALUES), ['run0_params']
    if FRONTIER_PARAMETER_SOURCE == 'ensemble':
        param_vals = ensemble.load_or_generate_gaussian_samples()
        labels = [f'sample{index:04d}' for index in range(len(param_vals))]
        return param_vals, labels
    if FRONTIER_PARAMETER_SOURCE in ('robustness', 'spec', 'specs'):
        return parameter_spec_values()
    raise ValueError(
        "FRONTIER_PARAMETER_SOURCE must be 'mean', 'ensemble', or 'robustness', "
        f"not {FRONTIER_PARAMETER_SOURCE!r}"
    )


def total_sample_count(param_vals):
    if FRONTIER_PARAMETER_SOURCE in ('mean', 'robustness', 'spec', 'specs'):
        return len(param_vals)
    return min(FRONTIER_N_SAMPLES, len(param_vals) - FRONTIER_SAMPLE_OFFSET)


def get_cluster_config(param_vals, labels):
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

    delays = frontier_delay_years()
    num_delays = len(delays)
    num_samples = total_sample_count(param_vals)
    total_combinations = num_samples * num_delays
    task_index = task_id - 1

    if task_index >= total_combinations:
        print(f"ERROR: Task ID {task_id} exceeds total combinations ({total_combinations})")
        print(f"FRONTIER_PARAMETER_SOURCE = {FRONTIER_PARAMETER_SOURCE}")
        print(f"FRONTIER_N_SAMPLES = {FRONTIER_N_SAMPLES}")
        print(f"delay_years = {delays}")
        print(f"Expected task range: 1-{total_combinations}")
        sys.exit(1)

    frontier_sample_index = task_index // num_delays
    delay_idx = task_index % num_delays
    delay_year = delays[delay_idx]
    sample_index = (
        FRONTIER_SAMPLE_OFFSET + frontier_sample_index
        if FRONTIER_PARAMETER_SOURCE == 'ensemble'
        else frontier_sample_index
    )

    out_folder = os.environ.get('OUTPUT_FOLDER', output_folder)
    baseline = int(os.environ.get('BASELINE_NUM', ensemble.baseline_num))
    job_id = os.environ.get('JOB_ID', 'Unknown')

    print("\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {total_combinations}")
    print(f"  Mapping: sample={sample_index}, delay_year={delay_year}")
    print(f"  Parameter source: {FRONTIER_PARAMETER_SOURCE}")
    print(f"  Parameter label: {labels[sample_index]}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")
    print(f"  Output folder: {out_folder}")
    print(f"  Baseline (SSP): {baseline}")
    print(f"  Frontier grid: {FRONTIER_GRID}")

    return sample_index, frontier_sample_index, task_id, delay_year, out_folder, baseline, delays


def frontier_decision_times(delay_year):
    decision_times_delay = DEFAULT_DECISION_TIMES.copy()
    delay_periods = 0

    if delay_year > 0:
        decision_times_delay[1] = delay_year
        delay_periods = 1

    decision_times_baseline = decision_times_delay.copy()
    return decision_times_baseline, decision_times_delay, delay_periods


def main():
    print("\nEZClimate DELAY FRONTIER ANALYSIS - CLUSTER ARRAY JOB\n")

    param_vals, labels = parameter_values_and_labels()
    sample_index, frontier_sample_index, task_id, delay_year, out_folder, baseline, delays = (
        get_cluster_config(param_vals, labels)
    )
    ensemble.setup_cluster_directories(out_folder)

    if task_id == 1:
        samples_copy = os.path.join(
            ensemble.DATA_DIR, out_folder, 'samples',
            f'delay_frontier_{FRONTIER_PARAMETER_SOURCE}_DIMS{PARAMETER_PRIOR_DIMS}_{GAUSSIAN_PRIOR_SET_NAME}.csv'
        )
        np.savetxt(samples_copy, param_vals, delimiter=',')
        labels_copy = os.path.join(
            ensemble.DATA_DIR, out_folder, 'samples',
            f'delay_frontier_{FRONTIER_PARAMETER_SOURCE}_labels.csv'
        )
        with open(labels_copy, 'w') as f:
            f.write('sample_index,label\n')
            for index, label in enumerate(labels):
                f.write(f'{index},{label}\n')
        print(f"\nSaved frontier parameter values to: {samples_copy}")
        print(f"Saved frontier parameter labels to: {labels_copy}")

    decision_times_baseline, decision_times_delay, delay_periods = frontier_decision_times(delay_year)
    common_years = sorted(set(
        DEFAULT_CALENDAR_YEARS
        + [DEFAULT_BASE_YEAR + delay for delay in delays]
        + [DEFAULT_BASE_YEAR + dt for dt in decision_times_baseline]
        + [DEFAULT_BASE_YEAR + dt for dt in decision_times_delay]
    ))

    sample_label = labels[sample_index]

    print("\nExecution Configuration:")
    print(f"  Test mode:        {ensemble.test_mode}")
    print(f"  Import damages:   {ensemble.import_damages}")
    print(f"  Delay year:       {delay_year}")
    print(f"  Delay periods:    {delay_periods}")
    print(f"  Frontier grid:    {FRONTIER_GRID}")
    print(f"  Period length:    {FRONTIER_PERIOD_LEN}")
    print(f"  Emissions step:   {FRONTIER_EMISSIONS_TIME_STEP}")
    print(f"  Damage file tag:  {FRONTIER_DAMAGE_FILE_TAG}")
    print(f"  Baseline times:   {decision_times_baseline}")
    print(f"  Delayed times:    {decision_times_delay}")

    try:
        ensemble.run_ensemble_delayed_analysis(
            sample_index=sample_index,
            delay_year=delay_year,
            param_vals=param_vals,
            out_folder=out_folder,
            baseline=baseline,
            test_mode=ensemble.test_mode,
            import_damages=ensemble.import_damages,
            run_type='delay_frontier',
            tree_spec='default',
            decision_times_baseline=decision_times_baseline,
            decision_times_delay=decision_times_delay,
            sample_label=sample_label,
            common_years=common_years,
            delay_periods=delay_periods,
            period_len=FRONTIER_PERIOD_LEN,
            emissions_time_step=FRONTIER_EMISSIONS_TIME_STEP,
            damage_file_tag=FRONTIER_DAMAGE_FILE_TAG,
        )
    except Exception as e:
        print(f"ERROR running delay-frontier task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nTASK COMPLETE: Delay frontier sample {frontier_sample_index}, delay {delay_year}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python
"""
Annual-grid mechanism decomposition runner for delayed-action analysis.

Each SGE task maps to one (mechanism_scenario, delay_year) pair. The runner
uses the annual-grid delay-frontier setup so all scenarios are directly
comparable for integer-year delays.
"""

import os
import sys
import csv

import numpy as np

from _project_paths import configure_paths
configure_paths()

import main_ensemble_delayed_cluster as ensemble
from src.config import (
    DEFAULT_BASE_YEAR,
    DEFAULT_CALENDAR_YEARS,
    DEFAULT_DECISION_TIMES,
    GAUSSIAN_PRIOR_SET_NAME,
    PARAMETER_PRIOR_INDEX,
    RUN0_PARAMETER_VALUES,
)


output_folder = "mechanism-decomposition-analysis"

MECHANISM_PERIOD_LEN = 1.0
MECHANISM_EMISSIONS_TIME_STEP = 1
MECHANISM_DAMAGE_FILE_TAG = "_GRID1_MECH"
DEFAULT_MECHANISM_DELAYS = list(range(0, 21))
DEFAULT_MECHANISM_SCENARIOS = [
    "baseline",
    "no_endogenous_learning",
    "low_eis",
    "high_eis",
    "low_ra",
    "high_ra",
    "no_tipping",
    "deterministic_damages",
]


def parse_csv_env(env_name, default_values, cast=str):
    env_value = os.environ.get(env_name)
    if not env_value:
        return list(default_values)
    return [cast(value.strip()) for value in env_value.split(",") if value.strip()]


def mechanism_delay_years():
    delays = parse_csv_env("DELAY_YEARS", DEFAULT_MECHANISM_DELAYS, int)
    invalid = [
        delay for delay in delays
        if delay < 0 or delay >= DEFAULT_DECISION_TIMES[2]
    ]
    if invalid:
        raise ValueError(
            "Mechanism delays must be nonnegative integer-year delays earlier "
            f"than {DEFAULT_DECISION_TIMES[2]}. Invalid delays: {invalid}"
        )
    return delays


def scenario_catalog():
    run0 = RUN0_PARAMETER_VALUES.copy()
    ra_idx = PARAMETER_PRIOR_INDEX["RA"]
    eis_idx = PARAMETER_PRIOR_INDEX["EIS"]
    tech_chg_idx = PARAMETER_PRIOR_INDEX["tech_chg"]
    tech_scale_idx = PARAMETER_PRIOR_INDEX["tech_scale"]

    scenarios = {}

    def add(name, label, description, param_updates=None, global_updates=None):
        row = run0.copy()
        for index, value in (param_updates or {}).items():
            row[index] = value
        scenarios[name] = {
            "name": name,
            "label": label,
            "description": description,
            "param_values": row,
            "global_updates": dict(global_updates or {}),
        }

    add(
        "baseline",
        "Baseline",
        "Mean-parameter annual frontier with endogenous learning on.",
    )
    add(
        "no_endogenous_learning",
        "No endogenous learning",
        "Sets tech_scale to zero to isolate the learning-by-doing channel.",
        {tech_scale_idx: 0.0},
    )
    add(
        "low_eis",
        "Low EIS",
        "Sets EIS to the lower prior support value.",
        {eis_idx: 0.55},
    )
    add(
        "high_eis",
        "High EIS",
        "Sets EIS to the upper prior support value.",
        {eis_idx: 1.86},
    )
    add(
        "low_ra",
        "Low RA",
        "Sets risk aversion to the lower prior support value.",
        {ra_idx: 3.0},
    )
    add(
        "high_ra",
        "High RA",
        "Sets risk aversion to the upper prior support value.",
        {ra_idx: 15.0},
    )
    add(
        "fast_exogenous_tech",
        "Fast exogenous technology",
        "Sets exogenous technical progress to the upper prior support value.",
        {tech_chg_idx: 3.0},
    )
    add(
        "no_tipping",
        "No tipping",
        "Turns off tipping risk while retaining damage and temperature uncertainty.",
        global_updates={"tip_on": 0},
    )
    add(
        "deterministic_damages",
        "Deterministic damages",
        "Turns off tipping, damage uncertainty, and temperature uncertainty.",
        global_updates={"tip_on": 0, "d_unc": 0, "t_unc": 0},
    )

    return scenarios


def mechanism_scenarios():
    catalog = scenario_catalog()
    names = parse_csv_env("MECHANISM_SCENARIOS", DEFAULT_MECHANISM_SCENARIOS, str)
    missing = [name for name in names if name not in catalog]
    if missing:
        valid = ", ".join(sorted(catalog))
        raise ValueError(
            f"Unknown MECHANISM_SCENARIOS entries {missing}. Valid names: {valid}"
        )
    return [catalog[name] for name in names]


def total_task_count(scenarios, delays):
    return len(scenarios) * len(delays)


def get_cluster_config(scenarios, delays):
    sge_task_id = os.environ.get("SGE_TASK_ID")
    if sge_task_id is None:
        print("ERROR: SGE_TASK_ID environment variable not found!")
        print("This script is designed to run as part of an SGE array job.")
        sys.exit(1)

    try:
        task_id = int(sge_task_id)
    except ValueError:
        print(f"ERROR: Invalid SGE_TASK_ID: {sge_task_id}")
        sys.exit(1)

    total_combinations = total_task_count(scenarios, delays)
    task_index = task_id - 1
    if task_index >= total_combinations:
        print(f"ERROR: Task ID {task_id} exceeds total combinations ({total_combinations})")
        print(f"Mechanism scenarios = {[scenario['name'] for scenario in scenarios]}")
        print(f"Delay years = {delays}")
        print(f"Expected task range: 1-{total_combinations}")
        sys.exit(1)

    scenario_index = task_index // len(delays)
    delay_index = task_index % len(delays)
    scenario = scenarios[scenario_index]
    delay_year = delays[delay_index]
    out_folder = os.environ.get("OUTPUT_FOLDER", output_folder)
    baseline = int(os.environ.get("BASELINE_NUM", ensemble.baseline_num))
    job_id = os.environ.get("JOB_ID", "Unknown")

    print("\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {total_combinations}")
    print(f"  Mapping: scenario={scenario['name']}, delay_year={delay_year}")
    print(f"  Scenario label: {scenario['label']}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")
    print(f"  Output folder: {out_folder}")
    print(f"  Baseline (SSP): {baseline}")

    return scenario_index, scenario, task_id, delay_year, out_folder, baseline


def mechanism_decision_times(delay_year):
    decision_times_delay = DEFAULT_DECISION_TIMES.copy()
    delay_periods = 0
    if delay_year > 0:
        decision_times_delay[1] = delay_year
        delay_periods = 1
    decision_times_baseline = decision_times_delay.copy()
    return decision_times_baseline, decision_times_delay, delay_periods


def write_scenario_metadata(out_folder, scenarios):
    samples_dir = os.path.join(ensemble.DATA_DIR, out_folder, "samples")
    param_values = np.vstack([scenario["param_values"] for scenario in scenarios])
    values_path = os.path.join(
        samples_dir,
        f"mechanism_decomposition_N{len(scenarios)}_{GAUSSIAN_PRIOR_SET_NAME}.csv",
    )
    np.savetxt(values_path, param_values, delimiter=",")

    labels_path = os.path.join(
        samples_dir,
        f"mechanism_decomposition_labels_N{len(scenarios)}_{GAUSSIAN_PRIOR_SET_NAME}.csv",
    )
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_index", "name", "label", "description", "global_updates"])
        for index, scenario in enumerate(scenarios):
            writer.writerow([
                index,
                scenario["name"],
                scenario["label"],
                scenario["description"],
                scenario["global_updates"],
            ])

    print(f"\nSaved mechanism parameter values to: {values_path}")
    print(f"Saved mechanism scenario labels to: {labels_path}")


def apply_global_updates(global_updates):
    fields = ["dam_func", "tip_on", "d_unc", "t_unc", "no_free_lunch"]
    original = {field: getattr(ensemble, field) for field in fields}
    for field, value in global_updates.items():
        if field not in original:
            raise ValueError(f"Unsupported scenario global override: {field}")
        setattr(ensemble, field, value)
    return original


def restore_global_updates(original):
    for field, value in original.items():
        setattr(ensemble, field, value)


def main():
    print("\nEZClimate MECHANISM DECOMPOSITION ANALYSIS - CLUSTER ARRAY JOB\n")

    delays = mechanism_delay_years()
    scenarios = mechanism_scenarios()
    scenario_index, scenario, task_id, delay_year, out_folder, baseline = (
        get_cluster_config(scenarios, delays)
    )
    dry_run = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes")
    if not dry_run:
        ensemble.setup_cluster_directories(out_folder)

    if task_id == 1 and not dry_run:
        write_scenario_metadata(out_folder, scenarios)

    decision_times_baseline, decision_times_delay, delay_periods = (
        mechanism_decision_times(delay_year)
    )
    common_years = sorted(set(
        DEFAULT_CALENDAR_YEARS
        + [DEFAULT_BASE_YEAR + delay for delay in delays]
        + [DEFAULT_BASE_YEAR + dt for dt in decision_times_baseline]
        + [DEFAULT_BASE_YEAR + dt for dt in decision_times_delay]
    ))
    param_vals = np.atleast_2d(scenario["param_values"])
    sample_label = scenario["name"]
    tree_spec = f"default__{scenario['name']}"
    damage_file_tag = MECHANISM_DAMAGE_FILE_TAG

    print("\nExecution Configuration:")
    print(f"  Test mode:        {ensemble.test_mode}")
    print(f"  Import damages:   {ensemble.import_damages}")
    print(f"  Scenario:         {scenario['name']} ({scenario['label']})")
    print(f"  Description:      {scenario['description']}")
    print(f"  Global updates:   {scenario['global_updates']}")
    print(f"  Delay year:       {delay_year}")
    print(f"  Delay periods:    {delay_periods}")
    print(f"  Period length:    {MECHANISM_PERIOD_LEN}")
    print(f"  Emissions step:   {MECHANISM_EMISSIONS_TIME_STEP}")
    print(f"  Damage file tag:  {damage_file_tag}")
    print(f"  Baseline times:   {decision_times_baseline}")
    print(f"  Delayed times:    {decision_times_delay}")

    if dry_run:
        print("\nDRY_RUN=1: task mapping validated; model optimization skipped.")
        return

    original_globals = apply_global_updates(scenario["global_updates"])
    try:
        ensemble.run_ensemble_delayed_analysis(
            sample_index=0,
            delay_year=delay_year,
            param_vals=param_vals,
            out_folder=out_folder,
            baseline=baseline,
            test_mode=ensemble.test_mode,
            import_damages=ensemble.import_damages,
            run_type="mechanism_decomposition",
            tree_spec=tree_spec,
            decision_times_baseline=decision_times_baseline,
            decision_times_delay=decision_times_delay,
            sample_label=sample_label,
            common_years=common_years,
            delay_periods=delay_periods,
            period_len=MECHANISM_PERIOD_LEN,
            emissions_time_step=MECHANISM_EMISSIONS_TIME_STEP,
            damage_file_tag=damage_file_tag,
        )
    except Exception as e:
        print(
            "ERROR running mechanism-decomposition scenario "
            f"{scenario['name']}, delay {delay_year}: {e}"
        )
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_global_updates(original_globals)

    print(
        "\nTASK COMPLETE: Mechanism scenario "
        f"{scenario_index} ({scenario['name']}), delay {delay_year}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

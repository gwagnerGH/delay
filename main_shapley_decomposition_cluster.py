#!/apps/anaconda3/bin/python
"""Five-year-grid Shapley decomposition runner for delayed-action costs.

Each SGE task maps to one (coalition, delay_year) pair. Coalitions are bitmasks
for theory-core economic channels, and the value function is the welfare loss
from the shared delayed-action solver.
"""

import csv
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, PROJECT_ROOT)

import main_ensemble_delayed_cluster as ensemble
from src.config import (
    DEFAULT_BASE_YEAR,
    DEFAULT_CALENDAR_YEARS,
    DEFAULT_DECISION_TIMES,
    GAUSSIAN_PRIOR_SET_NAME,
    PARAMETER_PRIOR_DIMS,
    PARAMETER_PRIOR_INDEX,
    RUN0_PARAMETER_VALUES,
)


output_folder = "shapley-decomposition-BY2025-fiveyear-run0-v1"

SHAPLEY_PLAYERS = (
    "climate_damages",
    "endogenous_learning",
    "exogenous_tech_progress",
    "uncertainty_tipping",
)
SHAPLEY_DELAYS = [5, 10, 15, 20]
SHAPLEY_PERIOD_LEN = 5.0
SHAPLEY_EMISSIONS_TIME_STEP = None
SHAPLEY_DAMAGE_FILE_TAG = "_GRID5_SHAPLEY"


def coalition_flags(mask):
    return {
        player: bool(mask & (1 << index))
        for index, player in enumerate(SHAPLEY_PLAYERS)
    }


def coalition_players_label(flags):
    players = [player for player in SHAPLEY_PLAYERS if flags[player]]
    return "|".join(players) if players else "none"


def total_task_count():
    return (2 ** len(SHAPLEY_PLAYERS)) * len(SHAPLEY_DELAYS)


def get_cluster_config():
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

    total = total_task_count()
    task_index = task_id - 1
    if task_index < 0 or task_index >= total:
        print(f"ERROR: Task ID {task_id} exceeds total combinations ({total})")
        print(f"Expected task range: 1-{total}")
        sys.exit(1)

    coalition_index = task_index // len(SHAPLEY_DELAYS)
    delay_index = task_index % len(SHAPLEY_DELAYS)
    delay_year = SHAPLEY_DELAYS[delay_index]
    flags = coalition_flags(coalition_index)
    out_folder = os.environ.get("OUTPUT_FOLDER", output_folder)
    baseline = int(os.environ.get("BASELINE_NUM", ensemble.baseline_num))
    job_id = os.environ.get("JOB_ID", "Unknown")

    print("\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {total}")
    print(f"  Coalition mask: {coalition_index}")
    print(f"  Coalition players: {coalition_players_label(flags)}")
    print(f"  Delay year: {delay_year}")
    print(f"  Output folder: {out_folder}")
    print(f"  Baseline (SSP): {baseline}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")

    return task_id, coalition_index, flags, delay_year, out_folder, baseline


def shapley_decision_times(delay_year):
    decision_times_delay = DEFAULT_DECISION_TIMES.copy()
    decision_times_delay[1] = delay_year
    decision_times_baseline = decision_times_delay.copy()
    return decision_times_baseline, decision_times_delay, 1


def coalition_param_values(flags):
    row = RUN0_PARAMETER_VALUES.copy()
    if not flags["endogenous_learning"]:
        row[PARAMETER_PRIOR_INDEX["tech_scale"]] = 0.0
    if not flags["exogenous_tech_progress"]:
        row[PARAMETER_PRIOR_INDEX["tech_chg"]] = 0.0
    return np.atleast_2d(row)


def metadata_for_coalition(mask, flags):
    metadata = {
        "decomposition_type": "true_shapley",
        "coalition_mask": int(mask),
        "coalition_players": coalition_players_label(flags),
    }
    for player in SHAPLEY_PLAYERS:
        metadata[f"player_{player}"] = int(flags[player])
    return metadata


def write_coalition_metadata(out_folder):
    samples_dir = os.path.join(ensemble.DATA_DIR, out_folder, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    path = os.path.join(samples_dir, f"shapley_coalitions_N{2 ** len(SHAPLEY_PLAYERS)}_{GAUSSIAN_PRIOR_SET_NAME}.csv")
    with open(path, "w", newline="") as f:
        fieldnames = ["coalition_mask", "coalition_players"] + [f"player_{p}" for p in SHAPLEY_PLAYERS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mask in range(2 ** len(SHAPLEY_PLAYERS)):
            flags = coalition_flags(mask)
            row = metadata_for_coalition(mask, flags)
            writer.writerow({name: row[name] for name in fieldnames})
    print(f"Saved Shapley coalition metadata to: {path}")


def apply_uncertainty_state(flags):
    fields = ["tip_on", "d_unc", "t_unc"]
    original = {field: getattr(ensemble, field) for field in fields}
    if not flags["uncertainty_tipping"]:
        ensemble.tip_on = 0
        ensemble.d_unc = 0
        ensemble.t_unc = 0
    return original


def restore_uncertainty_state(original):
    for field, value in original.items():
        setattr(ensemble, field, value)


def main():
    print("\nEZClimate TRUE SHAPLEY DELAY DECOMPOSITION - CLUSTER ARRAY JOB\n")

    task_id, coalition_mask, flags, delay_year, out_folder, baseline = get_cluster_config()
    dry_run = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes")

    decision_times_baseline, decision_times_delay, delay_periods = shapley_decision_times(delay_year)
    common_years = sorted(set(
        DEFAULT_CALENDAR_YEARS
        + [DEFAULT_BASE_YEAR + delay for delay in SHAPLEY_DELAYS]
        + [DEFAULT_BASE_YEAR + dt for dt in decision_times_baseline]
        + [DEFAULT_BASE_YEAR + dt for dt in decision_times_delay]
    ))
    param_vals = coalition_param_values(flags)
    output_metadata = metadata_for_coalition(coalition_mask, flags)
    sample_label = f"coalition_{coalition_mask:02d}"
    tree_spec = f"shapley__mask_{coalition_mask:02d}"
    zero_climate_damages = not flags["climate_damages"]

    print("\nExecution Configuration:")
    print(f"  Test mode:              {ensemble.test_mode}")
    print(f"  Import damages:          {ensemble.import_damages}")
    print(f"  Coalition metadata:      {output_metadata}")
    print(f"  Zero climate damages:    {zero_climate_damages}")
    print(f"  Period length:           {SHAPLEY_PERIOD_LEN}")
    print(f"  Emissions step:          {SHAPLEY_EMISSIONS_TIME_STEP}")
    print(f"  Damage file tag:         {SHAPLEY_DAMAGE_FILE_TAG}")
    print(f"  Baseline times:          {decision_times_baseline}")
    print(f"  Delayed times:           {decision_times_delay}")

    if dry_run:
        print("\nDRY_RUN=1: task mapping validated; model optimization skipped.")
        return

    ensemble.setup_cluster_directories(out_folder)
    if task_id == 1:
        write_coalition_metadata(out_folder)
        samples_copy = os.path.join(
            ensemble.DATA_DIR,
            out_folder,
            "samples",
            f"shapley_run0_params_DIMS{PARAMETER_PRIOR_DIMS}_{GAUSSIAN_PRIOR_SET_NAME}.csv",
        )
        np.savetxt(samples_copy, RUN0_PARAMETER_VALUES.reshape(1, -1), delimiter=",")
        print(f"Saved run-0 parameter vector to: {samples_copy}")

    original_uncertainty = apply_uncertainty_state(flags)
    try:
        ensemble.run_ensemble_delayed_analysis(
            sample_index=0,
            delay_year=delay_year,
            param_vals=param_vals,
            out_folder=out_folder,
            baseline=baseline,
            test_mode=ensemble.test_mode,
            import_damages=ensemble.import_damages,
            run_type="shapley_decomposition",
            tree_spec=tree_spec,
            comparison_type="same_grid",
            decision_times_baseline=decision_times_baseline,
            decision_times_delay=decision_times_delay,
            sample_label=sample_label,
            common_years=common_years,
            delay_periods=delay_periods,
            period_len=SHAPLEY_PERIOD_LEN,
            emissions_time_step=SHAPLEY_EMISSIONS_TIME_STEP,
            damage_file_tag=SHAPLEY_DAMAGE_FILE_TAG,
            output_metadata=output_metadata,
            zero_climate_damages=zero_climate_damages,
        )
    except Exception as e:
        print(
            "ERROR running Shapley coalition "
            f"{coalition_mask}, delay {delay_year}: {e}"
        )
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_uncertainty_state(original_uncertainty)

    print(
        "\nTASK COMPLETE: Shapley coalition "
        f"{coalition_mask} ({coalition_players_label(flags)}), delay {delay_year}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

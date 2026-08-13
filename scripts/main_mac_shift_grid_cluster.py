#!/usr/bin/env python
"""MAC-shift diagnostic grid runner."""

import itertools
import os
import numpy as np

from _project_paths import configure_paths
configure_paths()

import main_ensemble_delayed_cluster as ensemble
from src.analysis.delayed_action import (
    FIXED_DELAY_DAMAGE_FILE_TAG, FIXED_DELAY_EMISSIONS_TIME_STEP,
    FIXED_DELAY_PERIOD_LEN,
    fixed_delay_decision_times, get_delay_periods_for_year,
)
from src.config import (
    DEFAULT_BASE_YEAR, DEFAULT_CALENDAR_YEARS, PARAMETER_PRIOR_INDEX,
    RUN0_PARAMETER_VALUES,
)

HORIZONTAL_SHIFTS = tuple(np.round(np.arange(0.0, 0.5001, 0.05), 10))
VERTICAL_SHIFTS = tuple(np.arange(0.0, 250.01, 25.0))
DELAY_YEARS = (5, 10, 15)


def grid_points():
    return tuple(itertools.product(HORIZONTAL_SHIFTS, VERTICAL_SHIFTS))


def task_configuration():
    task_value = os.environ.get("SGE_TASK_ID") or os.environ.get("TASK_ID")
    if task_value is None:
        raise ValueError("SGE_TASK_ID or TASK_ID is required for the MAC-shift grid")
    task_id = int(task_value)
    points = grid_points()
    total = len(points) * len(DELAY_YEARS)
    if task_id < 1 or task_id > total:
        raise ValueError("Task {} is outside the valid range 1-{}".format(task_id, total))
    offset = task_id - 1
    point_index = offset // len(DELAY_YEARS)
    delay_year = DELAY_YEARS[offset % len(DELAY_YEARS)]
    horizontal_shift, vertical_shift = points[point_index]
    return task_id, point_index, delay_year, horizontal_shift, vertical_shift, total


def shift_label(point_index, horizontal_shift, vertical_shift):
    return "mac_p{:03d}_h{:g}_v{:g}".format(
        point_index + 1, horizontal_shift, vertical_shift
    ).replace(".", "p").replace("-", "m")


def parameter_row():
    """Return run-0 parameters with an optional complete preference override."""

    names = ("MAC_GRID_RA", "MAC_GRID_EIS", "MAC_GRID_PRTP")
    present = [name for name in names if os.environ.get(name, "") != ""]
    if present and len(present) != len(names):
        raise ValueError("Provide all of MAC_GRID_RA, MAC_GRID_EIS, and MAC_GRID_PRTP")
    row = np.asarray(RUN0_PARAMETER_VALUES, dtype=float).copy()
    if present:
        row[PARAMETER_PRIOR_INDEX["RA"]] = float(os.environ["MAC_GRID_RA"])
        row[PARAMETER_PRIOR_INDEX["EIS"]] = float(os.environ["MAC_GRID_EIS"])
        row[PARAMETER_PRIOR_INDEX["PRTP"]] = float(os.environ["MAC_GRID_PRTP"])
    return row


def main():
    task_id, point_index, delay_year, horizontal_shift, vertical_shift, total = (
        task_configuration()
    )
    out_folder = os.environ.get("OUTPUT_FOLDER", "paper-mac-shift-grid-v1")
    baseline = int(os.environ.get("BASELINE_NUM", ensemble.baseline_num))
    if ensemble.backstop_smoothing_width() != 0.0:
        raise ValueError("MAC-shift grid requires BACKSTOP_SMOOTHING_WIDTH=0")
    if ensemble.lbfgsb_policy_upper_bound() != 1.0:
        raise ValueError("MAC-shift grid requires LBFGSB_POLICY_UPPER_BOUND=1.0")

    ensemble.setup_cluster_directories(out_folder)
    decision_times = fixed_delay_decision_times()
    delay_periods = get_delay_periods_for_year(decision_times, delay_year)
    common_years = sorted(set(
        DEFAULT_CALENDAR_YEARS
        + [DEFAULT_BASE_YEAR + delay for delay in DELAY_YEARS]
        + [DEFAULT_BASE_YEAR + dt for dt in decision_times]
    ))
    row = parameter_row()
    label = shift_label(point_index, horizontal_shift, vertical_shift)
    metadata = {
        "mac_grid_point": int(point_index + 1),
        "mac_horizontal_shift": float(horizontal_shift),
        "mac_vertical_shift": float(vertical_shift),
        "mac_shift_formula": "MAC_base(m+h)+v",
        "mac_policy_cap": 1.0,
        "mac_grid_ra": float(row[PARAMETER_PRIOR_INDEX["RA"]]),
        "mac_grid_eis": float(row[PARAMETER_PRIOR_INDEX["EIS"]]),
        "mac_grid_prtp": float(row[PARAMETER_PRIOR_INDEX["PRTP"]]),
    }
    print(
        "MAC-shift task {}/{}: point {}/{} (h={}, v={} dollars/t), delay={} years".format(
            task_id, total, point_index + 1, len(grid_points()), horizontal_shift,
            vertical_shift, delay_year,
        )
    )
    ensemble.run_ensemble_delayed_analysis(
        sample_index=0, delay_year=delay_year,
        param_vals=np.atleast_2d(row),
        out_folder=out_folder, baseline=baseline, test_mode=ensemble.test_mode,
        import_damages=ensemble.import_damages, run_type="mac_shift_grid",
        tree_spec="default", comparison_type="mac_base_m_plus_h_plus_v",
        decision_times_baseline=decision_times, decision_times_delay=decision_times,
        sample_label=label, common_years=common_years, delay_periods=delay_periods,
        period_len=FIXED_DELAY_PERIOD_LEN,
        emissions_time_step=FIXED_DELAY_EMISSIONS_TIME_STEP,
        damage_file_tag=FIXED_DELAY_DAMAGE_FILE_TAG, output_metadata=metadata,
        mac_horizontal_shift=horizontal_shift, mac_vertical_shift=vertical_shift,
        delay_window_years=delay_year,
    )
    print("TASK COMPLETE: {}".format(label))


if __name__ == "__main__":
    main()

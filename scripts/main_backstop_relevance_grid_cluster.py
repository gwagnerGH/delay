#!/usr/bin/env python
"""Three-dimensional sharp-backstop relevance grid for SGE array jobs.

Each array task evaluates one EIS--RA--PRTP grid point at one delay length,
first restricting mitigation to m <= 1 and then allowing the original sharp
removal technology through m <= 1.5.  Tasks write the detailed model output
through the normal ensemble writer, plus an independent per-task JSON status
file.  A separate serial summarizer creates the pairwise relevance table only
after the array has completed; this avoids concurrent readers of large CSVs.
"""

import itertools
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from _project_paths import configure_paths

configure_paths()

import main_backstop_relevance_cluster as relevance
import main_ensemble_delayed_cluster as ensemble
from src.config import (
    PARAMETER_PRIOR_INDEX,
    PARAMETER_PRIOR_LOWER_BOUNDS,
    PARAMETER_PRIOR_MEANS,
    PARAMETER_PRIOR_UPPER_BOUNDS,
)


# 15 x 10 x 10 parameter points, each at three delay lengths, produces
# 4,500 tasks.  It runs as five 900-task scheduler waves while retaining fine
# coverage of the high-EIS region where the targeted diagnostic found removal.
DEFAULT_EIS_COUNT = 15
DEFAULT_RA_COUNT = 10
DEFAULT_PRTP_COUNT = 10


def _grid_values(name, lower, upper, default_count):
    """Read explicit comma-separated values or use an inclusive linear grid."""

    raw = os.environ.get(name, "")
    if raw:
        values = np.asarray(
            [float(value.strip()) for value in raw.split(",") if value.strip()],
            dtype=float,
        )
        if not len(values):
            raise ValueError("{} must contain at least one value".format(name))
    else:
        count = int(os.environ.get("{}_COUNT".format(name), default_count))
        if count < 2:
            raise ValueError("{}_COUNT must be at least 2".format(name))
        values = np.linspace(lower, upper, count)
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError(
            "{} values must lie in [{}, {}], got {}".format(
                name, lower, upper, values.tolist()
            )
        )
    return tuple(float(value) for value in values)


def grid_definition():
    """Return deterministic parameter points and the selected delay lengths."""

    index = PARAMETER_PRIOR_INDEX
    lower = PARAMETER_PRIOR_LOWER_BOUNDS
    upper = PARAMETER_PRIOR_UPPER_BOUNDS
    eis = _grid_values(
        "BACKSTOP_GRID_EIS", lower[index["EIS"]], upper[index["EIS"]],
        DEFAULT_EIS_COUNT,
    )
    ra = _grid_values(
        "BACKSTOP_GRID_RA", lower[index["RA"]], upper[index["RA"]],
        DEFAULT_RA_COUNT,
    )
    prtp = _grid_values(
        "BACKSTOP_GRID_PRTP", lower[index["PRTP"]], upper[index["PRTP"]],
        DEFAULT_PRTP_COUNT,
    )
    points = tuple(itertools.product(eis, ra, prtp))
    return points, tuple(ensemble.delay_years)


def task_configuration():
    task_value = os.environ.get("SGE_TASK_ID") or os.environ.get("TASK_ID")
    if task_value is None:
        raise ValueError("SGE_TASK_ID is required for the backstop grid array")
    task_id = int(task_value)
    points, delays = grid_definition()
    total = len(points) * len(delays)
    if task_id < 1 or task_id > total:
        raise ValueError("Task {} is outside the valid range 1-{}".format(task_id, total))
    offset = task_id - 1
    point_index = offset // len(delays)
    delay_year = delays[offset % len(delays)]
    return task_id, point_index, points[point_index], delay_year, points, delays


def parameter_row(eis, ra, prtp):
    row = np.asarray(PARAMETER_PRIOR_MEANS, dtype=float).copy()
    row[PARAMETER_PRIOR_INDEX["EIS"]] = eis
    row[PARAMETER_PRIOR_INDEX["RA"]] = ra
    row[PARAMETER_PRIOR_INDEX["PRTP"]] = prtp
    return row


def point_label(point_index, eis, ra, prtp):
    """Unique, short output label; numerical values are also written as metadata."""

    return "grid_p{:03d}_eis{:g}_ra{:g}_prtp{:g}".format(
        point_index + 1, eis, ra, prtp
    ).replace("-", "m").replace(".", "p")


def status_directory(out_folder):
    return Path(ensemble.DATA_DIR) / out_folder / "analysis" / "backstop_grid_task_status"


def write_task_status(out_folder, task_id, payload):
    """Atomically publish the task status without any shared append contention."""

    directory = status_directory(out_folder)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "task_{:04d}.json".format(task_id)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(directory), delete=False
    ) as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, destination)


def run_with_cap(cap, task_id, point_index, label, delay_year, row, out_folder,
                 baseline, eis, ra, prtp, comparison_type):
    previous = os.environ.get("LBFGSB_POLICY_UPPER_BOUND")
    os.environ["LBFGSB_POLICY_UPPER_BOUND"] = "{:.12g}".format(cap)
    try:
        ensemble.run_ensemble_delayed_analysis(
            # ``param_vals`` contains exactly this grid point, so the model row index is zero.
            # The task ID remains in metadata and the unique sample label.
            sample_index=0,
            delay_year=delay_year,
            param_vals=np.atleast_2d(row),
            out_folder=out_folder,
            baseline=baseline,
            test_mode=ensemble.test_mode,
            import_damages=ensemble.import_damages,
            run_type="backstop_relevance_grid",
            tree_spec="default",
            comparison_type=comparison_type,
            sample_label=label,
            output_metadata={
                "backstop_grid_point": int(point_index + 1),
                "backstop_grid_eis": float(eis),
                "backstop_grid_ra": float(ra),
                "backstop_grid_prtp": float(prtp),
                "backstop_policy_cap": float(cap),
                "backstop_available": bool(cap > 1.0),
            },
        )
    finally:
        if previous is None:
            os.environ.pop("LBFGSB_POLICY_UPPER_BOUND", None)
        else:
            os.environ["LBFGSB_POLICY_UPPER_BOUND"] = previous


def main():
    task_id, point_index, point, delay_year, points, delays = task_configuration()
    eis, ra, prtp = point
    out_folder = os.environ.get("OUTPUT_FOLDER", "paper-backstop-relevance-grid-v1")
    baseline = int(os.environ.get("BASELINE_NUM", ensemble.baseline_num))
    backstop_cap = float(os.environ.get("BACKSTOP_GRID_BACKSTOP_CAP", "1.5"))
    if backstop_cap <= 1.0:
        raise ValueError("BACKSTOP_GRID_BACKSTOP_CAP must exceed 1.0")
    if ensemble.backstop_smoothing_width() != 0.0:
        raise ValueError("This diagnostic requires BACKSTOP_SMOOTHING_WIDTH=0")

    ensemble.setup_cluster_directories(out_folder)
    row = parameter_row(eis, ra, prtp)
    label = point_label(point_index, eis, ra, prtp)
    cap_one_label = "{}_cap1".format(label)
    backstop_label = "{}_backstop".format(label)
    status = {
        "task_id": task_id,
        "point_index": point_index + 1,
        "delay_year": int(delay_year),
        "eis": float(eis),
        "ra": float(ra),
        "prtp": float(prtp),
        "cap_one_label": cap_one_label,
        "backstop_label": backstop_label,
        "backstop_policy_cap": backstop_cap,
        "cap_one_completed": False,
        "backstop_completed": False,
        "backstop_error": "",
    }
    print(
        "Backstop grid task {}/{}: point {}/{} (EIS={}, RA={}, PRTP={}), "
        "delay={}, caps=(1.0, {})".format(
            task_id, len(points) * len(delays), point_index + 1, len(points),
            eis, ra, prtp, delay_year, backstop_cap,
        )
    )
    try:
        run_with_cap(
            1.0, task_id, point_index, cap_one_label, delay_year, row,
            out_folder, baseline, eis, ra, prtp, "ordinary_mitigation_only",
        )
        status["cap_one_completed"] = True
    except Exception as exc:
        status["cap_one_error"] = "{}: {}".format(type(exc).__name__, exc)
        write_task_status(out_folder, task_id, status)
        print("CAP-ONE SOLVE FAILED: {}".format(status["cap_one_error"]))
        raise

    try:
        run_with_cap(
            backstop_cap, task_id, point_index, backstop_label, delay_year, row,
            out_folder, baseline, eis, ra, prtp, "sharp_backstop_available",
        )
        status["backstop_completed"] = True
    except Exception as exc:
        status["backstop_error"] = "{}: {}".format(type(exc).__name__, exc)
        print("BACKSTOP SOLVE FAILED: {}".format(status["backstop_error"]))
    write_task_status(out_folder, task_id, status)
    if status["backstop_completed"]:
        print("TASK COMPLETE: backstop grid point {} delay {}".format(point_index + 1, delay_year))
    else:
        print("TASK COMPLETE WITH EXPLICIT BACKSTOP FAILURE STATUS")


if __name__ == "__main__":
    main()

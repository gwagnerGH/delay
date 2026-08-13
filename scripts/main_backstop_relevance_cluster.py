#!/usr/bin/env python
"""Targeted sharp-backstop relevance experiment.

Each task solves one economically informative parameter case at one delay
length twice: first with ordinary mitigation only (``m <= 1``), and then with
the original, sharp removal technology available (``m <= 1.5`` by default).
The runner writes the ordinary consolidated scenario outputs plus a compact
pairwise summary.  It is intentionally a small diagnostic grid, not a second
paper-wide ensemble.
"""

import csv
import os
import sys

import numpy as np

from _project_paths import configure_paths

configure_paths()

import main_ensemble_delayed_cluster as ensemble
from src.config import (
    PARAMETER_PRIOR_INDEX,
    PARAMETER_PRIOR_LOWER_BOUNDS,
    PARAMETER_PRIOR_MEANS,
    PARAMETER_PRIOR_UPPER_BOUNDS,
)


DEFAULT_CASES = (
    "mean",
    "low_eis",
    "high_eis",
    "low_ra",
    "high_ra",
    "high_discount_rate",
    "no_endogenous_learning",
)


def selected_case_names():
    raw = os.environ.get("BACKSTOP_RELEVANCE_CASES", "")
    names = tuple(
        item.strip().lower() for item in raw.split(",") if item.strip()
    ) if raw else DEFAULT_CASES
    unknown = [name for name in names if name not in DEFAULT_CASES]
    if unknown:
        raise ValueError(
            "Unknown BACKSTOP_RELEVANCE_CASES values {}. Valid values: {}"
            .format(unknown, ", ".join(DEFAULT_CASES))
        )
    if not names:
        raise ValueError("BACKSTOP_RELEVANCE_CASES must select at least one case")
    return names


def parameter_row(case_name):
    """Return the run-0 parameter vector with one interpretable perturbation."""

    row = np.asarray(PARAMETER_PRIOR_MEANS, dtype=float).copy()
    index = PARAMETER_PRIOR_INDEX
    lower = PARAMETER_PRIOR_LOWER_BOUNDS
    upper = PARAMETER_PRIOR_UPPER_BOUNDS
    if case_name == "low_eis":
        row[index["EIS"]] = lower[index["EIS"]]
    elif case_name == "high_eis":
        row[index["EIS"]] = upper[index["EIS"]]
    elif case_name == "low_ra":
        row[index["RA"]] = lower[index["RA"]]
    elif case_name == "high_ra":
        row[index["RA"]] = upper[index["RA"]]
    elif case_name == "high_discount_rate":
        row[index["PRTP"]] = upper[index["PRTP"]]
    elif case_name == "no_endogenous_learning":
        row[index["tech_scale"]] = lower[index["tech_scale"]]
    return row


def task_configuration():
    task_value = os.environ.get("SGE_TASK_ID") or os.environ.get("TASK_ID")
    if task_value is None:
        raise ValueError("SGE_TASK_ID is required for the backstop relevance array")
    task_id = int(task_value)
    cases = selected_case_names()
    delays = list(ensemble.delay_years)
    total = len(cases) * len(delays)
    if task_id < 1 or task_id > total:
        raise ValueError("Task {} is outside the valid range 1-{}".format(task_id, total))
    offset = task_id - 1
    return task_id, cases[offset // len(delays)], delays[offset % len(delays)], cases


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row, field):
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        return np.nan


def _find_result(rows, sample_label, delay_year):
    """Find one paired result, distinguishing the three delay tasks per case."""
    matching = [
        row for row in rows
        if row.get("sample_index") == sample_label
        and str(row.get("delay_year")) == str(delay_year)
    ]
    if len(matching) != 1:
        raise RuntimeError(
            "Expected one consolidated result for {!r} at delay {}, found {}".format(
                sample_label, delay_year, len(matching)
            )
        )
    return matching[0]


def _removal_summary(node_rows, sample_label, delay_year, scenario):
    rows = [
        row for row in node_rows
        if row.get("sample_index") == sample_label
        and str(row.get("delay_year")) == str(delay_year)
        and row.get("scenario") == scenario
    ]
    mitigation = np.asarray([_float(row, "mitigation") for row in rows], dtype=float)
    probabilities = np.asarray([_float(row, "node_probability") for row in rows], dtype=float)
    excess = np.maximum(mitigation - 1.0, 0.0)
    active = np.isfinite(excess) & (excess > 1e-8)
    return {
        "removal_node_count": int(np.sum(active)),
        "max_removal_fraction": float(np.nanmax(excess)) if len(excess) else np.nan,
        "probability_weighted_removal_fraction": float(
            np.nansum(np.where(active, probabilities * excess, 0.0))
        ),
    }


def append_pair_summary(out_folder, task_id, case_name, delay_year,
                        cap_one_label, available_label, backstop_cap,
                        available_error=""):
    """Create a compact, explicitly status-labelled comparison row."""

    analysis = os.path.join(ensemble.DATA_DIR, out_folder, "analysis")
    result_rows = _read_csv(
        os.path.join(analysis, "{}_consolidated_results.csv".format(out_folder))
    )
    cap_one = _find_result(result_rows, cap_one_label, delay_year)
    summary = {
        "task_id": int(task_id),
        "case": case_name,
        "delay_year": int(delay_year),
        "cap_one_label": cap_one_label,
        "backstop_label": available_label,
        "backstop_policy_cap": float(backstop_cap),
        "status": "backstop_failed" if available_error else "completed",
        "backstop_error": available_error,
        "ra": _float(cap_one, "ra"),
        "eis": _float(cap_one, "eis"),
        "pref": _float(cap_one, "pref"),
        "tech_chg": _float(cap_one, "tech_chg"),
        "tech_scale": _float(cap_one, "tech_scale"),
        "cap_one_u_optimal": _float(cap_one, "u_optimal"),
        "cap_one_u_delayed": _float(cap_one, "u_delayed"),
        "cap_one_price_optimal": _float(cap_one, "carbon_price_optimal"),
        "cap_one_price_delayed": _float(cap_one, "carbon_price_delayed"),
    }
    if not available_error:
        available = _find_result(result_rows, available_label, delay_year)
        summary.update({
            "backstop_u_optimal": _float(available, "u_optimal"),
            "backstop_u_delayed": _float(available, "u_delayed"),
            "backstop_price_optimal": _float(available, "carbon_price_optimal"),
            "backstop_price_delayed": _float(available, "carbon_price_delayed"),
        })
        summary["optimal_utility_gain"] = (
            summary["backstop_u_optimal"] - summary["cap_one_u_optimal"]
        )
        summary["delayed_utility_gain"] = (
            summary["backstop_u_delayed"] - summary["cap_one_u_delayed"]
        )
        node_rows = _read_csv(
            os.path.join(analysis, "{}_node_prices.csv".format(out_folder))
        )
        for scenario in ("optimal", "delayed"):
            for key, value in _removal_summary(
                node_rows, available_label, delay_year, scenario
            ).items():
                summary["{}_{}".format(scenario, key)] = value
        summary["backstop_relevant_optimal"] = bool(
            summary["optimal_utility_gain"] > 1e-8
            and summary["optimal_removal_node_count"] > 0
        )
        summary["backstop_relevant_delayed"] = bool(
            summary["delayed_utility_gain"] > 1e-8
            and summary["delayed_removal_node_count"] > 0
        )
    else:
        summary.update({
            "backstop_u_optimal": np.nan,
            "backstop_u_delayed": np.nan,
            "optimal_utility_gain": np.nan,
            "delayed_utility_gain": np.nan,
            "backstop_relevant_optimal": False,
            "backstop_relevant_delayed": False,
        })
    path = os.path.join(analysis, "{}_backstop_relevance.csv".format(out_folder))
    if not ensemble.append_results_to_csv(summary, path):
        raise RuntimeError("Could not append backstop relevance summary to {}".format(path))


def run_with_cap(cap, case_name, label, delay_year, row, out_folder, baseline,
                 comparison_type):
    previous = os.environ.get("LBFGSB_POLICY_UPPER_BOUND")
    os.environ["LBFGSB_POLICY_UPPER_BOUND"] = "{:.12g}".format(cap)
    try:
        ensemble.run_ensemble_delayed_analysis(
            sample_index=0,
            delay_year=delay_year,
            param_vals=np.atleast_2d(row),
            out_folder=out_folder,
            baseline=baseline,
            test_mode=ensemble.test_mode,
            import_damages=ensemble.import_damages,
            run_type="backstop_relevance",
            tree_spec="default",
            comparison_type=comparison_type,
            sample_label=label,
            output_metadata={
                "backstop_relevance_case": case_name,
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
    task_id, case_name, delay_year, cases = task_configuration()
    out_folder = os.environ.get("OUTPUT_FOLDER", "paper-backstop-relevance-v1")
    baseline = int(os.environ.get("BASELINE_NUM", ensemble.baseline_num))
    backstop_cap = float(os.environ.get("BACKSTOP_RELEVANCE_BACKSTOP_CAP", "1.5"))
    if backstop_cap <= 1.0:
        raise ValueError("BACKSTOP_RELEVANCE_BACKSTOP_CAP must exceed 1.0")
    if ensemble.backstop_smoothing_width() != 0.0:
        raise ValueError("This diagnostic requires BACKSTOP_SMOOTHING_WIDTH=0")

    ensemble.setup_cluster_directories(out_folder)
    row = parameter_row(case_name)
    cap_one_label = "{}_cap1".format(case_name)
    available_label = "{}_backstop".format(case_name)
    print(
        "Backstop relevance task {}/{}: case={}, delay={}, caps=(1.0, {})".format(
            task_id, len(cases) * len(ensemble.delay_years), case_name,
            delay_year, backstop_cap,
        )
    )
    run_with_cap(
        1.0, case_name, cap_one_label, delay_year, row, out_folder, baseline,
        "ordinary_mitigation_only",
    )
    try:
        run_with_cap(
            backstop_cap, case_name, available_label, delay_year, row,
            out_folder, baseline, "sharp_backstop_available",
        )
    except Exception as exc:
        message = "{}: {}".format(type(exc).__name__, exc)
        print("BACKSTOP SOLVE FAILED: {}".format(message))
        append_pair_summary(
            out_folder, task_id, case_name, delay_year, cap_one_label,
            available_label, backstop_cap, available_error=message,
        )
        print("TASK COMPLETE WITH EXPLICIT BACKSTOP FAILURE STATUS")
        return
    append_pair_summary(
        out_folder, task_id, case_name, delay_year, cap_one_label,
        available_label, backstop_cap,
    )
    print("TASK COMPLETE: backstop relevance {} delay {}".format(case_name, delay_year))


if __name__ == "__main__":
    main()

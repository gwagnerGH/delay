#!/usr/bin/env python
"""Full-tree delay robustness to deterministic and IID lognormal post-2250 growth.

Only the continuation growth process changes. Consumption growth through 2250,
preferences, damages, costs, technology, and decision times stay at the shared
baseline. Annual deterministic NET growth g and annual mean LOG gross growth mu
are different inputs; both their arithmetic means and certainty equivalents are
recorded. This is a sensitivity exercise, not a calibrated macro-risk process.

Default SGE tasks 1..39: 13 cases x (5, 10, 15)-year delay. Each task reoptimizes
both the optimal and delayed policies using the revised adjoint runner. No old
outputs are substituted. Use --list-cases to validate the entire grid without
solving. DELAY_YEARS=10 reduces the array to 1..13.
"""

import argparse
import csv
from dataclasses import dataclass
import json
import os
import sys

import numpy as np

from src.config.parameter_priors import RUN0_PARAMETER_VALUES
from src.config.revised_baseline import BASELINE
from src.config.terminal_growth import TerminalGrowthSettings, NonFiniteTerminalError


OUTPUT_FOLDER = "revised-growth-delay-RA10-EIS15-post2250-v1"
DELAY_YEARS = (5, 10, 15)
DETERMINISTIC_GROWTH = (0.005, 0.010, 0.013)
LOG_MEANS = (0.005, 0.010, 0.015)
LOG_SIGMAS = (0.010, 0.020, 0.030)


@dataclass(frozen=True)
class GrowthCase:
    label: str
    settings: TerminalGrowthSettings
    matched_control_for: str = ""


def _number_label(value):
    return f"{value:g}".replace(".", "p")


def build_cases():
    cases = [GrowthCase(f"det_g{_number_label(g)}",
                        TerminalGrowthSettings(mode="deterministic", growth=g))
             for g in DETERMINISTIC_GROWTH]
    for mu in LOG_MEANS:
        for sigma in LOG_SIGMAS:
            cases.append(GrowthCase(
                f"log_mu{_number_label(mu)}_sigma{_number_label(sigma)}",
                TerminalGrowthSettings(mode="lognormal", log_mean=mu, log_sigma=sigma)))
    # Exact control: the homogeneous IID tail enters through one EZ certainty
    # equivalent. Matching it must reproduce fixed-policy terminal utility.
    settings = TerminalGrowthSettings()
    info = settings.describe(ra=BASELINE.ra, eis=BASELINE.eis, prtp=BASELINE.prtp)
    cases.append(GrowthCase("det_baseline_effective_growth",
        TerminalGrowthSettings(mode="deterministic",
                               growth=info["effective_annual_net_growth"]),
        matched_control_for="log_mu0p015_sigma0p02"))
    return tuple(cases)


CASES = build_cases()


def validate_grid(cases=None):
    """Fail before any solve if ANY requested continuation is non-finite."""
    if cases is None:
        cases = CASES
    rows = []
    for case in cases:
        try:
            info = case.settings.describe(ra=BASELINE.ra, eis=BASELINE.eis,
                prtp=BASELINE.prtp, modeled_growth=BASELINE.growth)
        except NonFiniteTerminalError as exc:
            raise NonFiniteTerminalError(f"Growth case {case.label}: {exc}") from exc
        rows.append(dict(label=case.label, **info,
            finite=True, convergence_margin=1-info["continuation_ratio"],
            matched_control_for=case.matched_control_for))
    return rows


def parse_delay_years(value):
    try:
        years = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("DELAY_YEARS must be a comma-separated subset of 5,10,15") from exc
    if not years or len(set(years)) != len(years) or any(y not in DELAY_YEARS for y in years):
        raise ValueError("DELAY_YEARS must be a nonempty, nonduplicated subset of 5,10,15")
    return years


def task_configuration(task_id, delay_years=DELAY_YEARS):
    count = len(CASES)*len(delay_years)
    if not delay_years or not 1 <= task_id <= count:
        raise ValueError(f"Task {task_id} is outside 1-{count}")
    index = task_id-1
    return CASES[index//len(delay_years)], delay_years[index % len(delay_years)]


def print_grid(rows, delay_years):
    print(f"Post-2250 growth only; RA={BASELINE.ra:g}, EIS={BASELINE.eis:g}, "
          f"PRTP={100*BASELINE.prtp:g}%, pre-2250 growth={100*BASELINE.growth:g}%")
    print(f"{len(rows)} finite cases x delays {delay_years}: "
          f"SGE array 1-{len(rows)*len(delay_years)}")
    print("Rates below are annual percentages. g is net growth; mu/sigma refer to log gross growth.")
    writer = csv.writer(sys.stdout)
    writer.writerow(["case", "mode", "g_pct", "mu_pct", "sigma_pct",
                     "mean_net_growth_pct", "effective_net_growth_pct",
                     "terminal_multiplier", "continuation_ratio", "tasks"])
    for index, row in enumerate(rows):
        stochastic = row["mode"] == "lognormal"
        first = index*len(delay_years)+1
        writer.writerow([row["label"], row["mode"],
            "" if stochastic else f"{100*row['growth']:.8g}",
            f"{100*row['log_mean']:.8g}" if stochastic else "",
            f"{100*row['log_sigma']:.8g}" if stochastic else "",
            f"{100*row['expected_annual_net_growth']:.8g}",
            f"{100*row['effective_annual_net_growth']:.8g}",
            f"{row['multiplier']:.10g}", f"{row['continuation_ratio']:.12g}",
            f"{first}-{first+len(delay_years)-1}"])
    print("Finite does not mean insensitive to the tail: mu=1.5%, sigma=1% "
          "has a multiplier of about 28,282 under these preferences.")
    print("The deterministic effective-growth control matches the baseline tail "
          "certainty equivalent; it does not match its arithmetic mean growth.")


def _set_baseline_environment(case):
    # Do not let an inherited terminal or cost environment turn a growth-only
    # comparison into a backstop, learning, or mitigation-cap comparison.
    for key, value in {
        "EZDELAY_LEGACY_CLUSTER": 0,
        "REVISED_BACKSTOP": BASELINE.backstop,
        "REVISED_GAMMA": BASELINE.gamma,
        "REVISED_BACKSTOP_CURVATURE": BASELINE.backstop_curvature,
        "REVISED_NUMERICAL_GUARD": 4.0,
        "LBFGSB_POLICY_UPPER_BOUND": 4.0,
        "REVISED_TERMINAL_GROWTH_MODE": case.settings.mode,
        "REVISED_TERMINAL_GROWTH": case.settings.growth,
        "REVISED_TERMINAL_LOG_MEAN": case.settings.log_mean,
        "REVISED_TERMINAL_LOG_SIGMA": case.settings.log_sigma,
    }.items():
        os.environ[key] = str(value)


def execute_task(task_id, delay_years=DELAY_YEARS, output_folder=OUTPUT_FOLDER):
    rows = validate_grid()  # All cases, not just the selected array task.
    case, delay_year = task_configuration(task_id, delay_years)
    _set_baseline_environment(case)
    # Keep explicit --task-id paths consistent with the case actually executed.
    os.environ["TASK_ID"] = str(task_id)
    if "SGE_TASK_ID" in os.environ:
        os.environ["SGE_TASK_ID"] = str(task_id)
    from scripts.run_revised_cluster import run_cluster_pair
    print(f"Growth sensitivity {case.label}; {delay_year}-year delay", flush=True)
    result = run_cluster_pair(
        sample_index=0, delay_year=delay_year,
        param_vals=np.atleast_2d(np.append(RUN0_PARAMETER_VALUES, BASELINE.backstop_curvature)),
        out_folder=output_folder, baseline=BASELINE.baseline_num,
        run_type="growth_sensitivity", tree_spec="full", sample_label=case.label,
        delay_window_years=delay_year,
        output_metadata={"growth_sensitivity": case.label,
            "growth_sensitivity_scope": "post-2250 continuation only; IID annual shocks",
            "fixed_preterminal_growth": BASELINE.growth,
            "growth_grid_preflight": rows,
            "growth_grid_delays": list(delay_years), "growth_grid_task_id": task_id,
            "matched_control_for": case.matched_control_for},
    )
    if not result["optimal_certified"] or not result["delayed_certified"]:
        raise RuntimeError("Growth task saved but at least one solve failed local stationarity checks; inspect manifests")
    print(f"Completed task {task_id}/{len(CASES)*len(delay_years)}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cases", "--dry-run", dest="list_cases", action="store_true",
                        help="Print the finite grid and task mapping without solving")
    parser.add_argument("--json", action="store_true", help="Print preflight as JSON (requires --list-cases)")
    parser.add_argument("--task-id", type=int, help="Override SGE_TASK_ID/TASK_ID")
    parser.add_argument("--delay-years", default=os.environ.get("DELAY_YEARS", "5,10,15"))
    parser.add_argument("--output-folder", default=os.environ.get("OUTPUT_FOLDER", OUTPUT_FOLDER))
    args = parser.parse_args(argv)
    if args.json and not args.list_cases:
        parser.error("--json requires --list-cases")
    years = parse_delay_years(args.delay_years)
    rows = validate_grid()
    if args.list_cases:
        if args.json:
            print(json.dumps(dict(cases=rows, delays=list(years),
                array_tasks=len(rows)*len(years), preferences=dict(
                    ra=BASELINE.ra, eis=BASELINE.eis, prtp=BASELINE.prtp),
                preterminal_growth=BASELINE.growth), indent=2, allow_nan=False))
        else:
            print_grid(rows, years)
        return
    raw_task = args.task_id
    if raw_task is None:
        raw_task = os.environ.get("SGE_TASK_ID") or os.environ.get("TASK_ID")
    if raw_task is None:
        parser.error("SGE_TASK_ID/TASK_ID or --task-id is required; use --list-cases for preflight")
    execute_task(int(raw_task), years, args.output_folder)


if __name__ == "__main__":
    main()

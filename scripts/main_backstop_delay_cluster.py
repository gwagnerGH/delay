#!/usr/bin/env python
"""Baseline plus seven backstop sensitivities at 5-, 10-, and 15-year delay.

One SGE array task solves an optimal/delayed pair on the full EZDelay tree.
All non-backstop parameters come from the shared revised baseline. Results
are written by the revised adjoint cluster runner; no old sharp-backstop code
is used.
"""

import os

import numpy as np

from src.config.parameter_priors import RUN0_PARAMETER_VALUES
from scripts.run_revised_cluster import run_cluster_pair


OUTPUT_FOLDER = "revised-backstop-delay-RA10-EIS15-logterminal-curvature-v3"
DELAY_YEARS = (5, 10, 15)
CASES = (
    dict(label="baseline_s_curve", backstop="s_curve", gamma=0.0,
         numerical_guard=4.0, economic_cap=4.0),
    dict(label="cubic_g0", backstop="cubic", gamma=0.0,
         numerical_guard=4.0, economic_cap=4.0),
    dict(label="cubic_g200000", backstop="cubic", gamma=200000.0,
         numerical_guard=4.0, economic_cap=4.0),
    dict(label="exp_cubic_g200000_a1000", backstop="exp_cubic", gamma=200000.0,
         numerical_guard=1.05, economic_cap=1.05),
    dict(label="no_backstop_m1", backstop="s_curve", gamma=0.0,
         numerical_guard=4.0, economic_cap=1.0),
    dict(label="s_curve_curvature_0p5", backstop="s_curve", gamma=0.0,
         curvature=0.5, numerical_guard=4.0, economic_cap=4.0),
    dict(label="s_curve_curvature_1p5", backstop="s_curve", gamma=0.0,
         curvature=1.5, numerical_guard=4.0, economic_cap=4.0),
    dict(label="s_curve_curvature_2", backstop="s_curve", gamma=0.0,
         curvature=2.0, numerical_guard=4.0, economic_cap=4.0),
)


def task_configuration(task_id):
    """Map SGE task 1..24 to a cost specification and delay duration."""
    if not 1 <= task_id <= len(CASES)*len(DELAY_YEARS):
        raise ValueError(f"Task {task_id} is outside 1-{len(CASES)*len(DELAY_YEARS)}")
    index = task_id-1
    return CASES[index//len(DELAY_YEARS)], DELAY_YEARS[index%len(DELAY_YEARS)]


def main():
    raw_task = os.environ.get("SGE_TASK_ID") or os.environ.get("TASK_ID")
    if raw_task is None:
        raise ValueError("SGE_TASK_ID or TASK_ID is required")
    task_id = int(raw_task)
    case, delay_year = task_configuration(task_id)
    print(f"Backstop sensitivity {case['label']}; {delay_year}-year delay", flush=True)
    os.environ["REVISED_BACKSTOP"] = case["backstop"]
    os.environ["REVISED_GAMMA"] = str(case["gamma"])
    os.environ["REVISED_BACKSTOP_CURVATURE"] = str(case.get("curvature", 1.0))
    os.environ["REVISED_NUMERICAL_GUARD"] = str(case["numerical_guard"])
    os.environ["LBFGSB_POLICY_UPPER_BOUND"] = str(case["economic_cap"])
    result = run_cluster_pair(
        sample_index=0, delay_year=delay_year,
        param_vals=np.atleast_2d(np.append(
            RUN0_PARAMETER_VALUES, case.get("curvature", 1.0))),
        out_folder=os.environ.get("OUTPUT_FOLDER", OUTPUT_FOLDER),
        baseline=2, run_type="backstop_delay", tree_spec="full",
        sample_label=case["label"], delay_window_years=delay_year,
        output_metadata={"backstop_sensitivity": case["label"],
                         "backstop_curvature": case.get("curvature", 1.0),
                         "exponential_coefficient": (1000.0 if case["backstop"] == "exp_cubic" else None),
                         "no_removal": case["label"] == "no_backstop_m1"},
    )
    if not result["optimal_certified"] or not result["delayed_certified"]:
        raise RuntimeError("Backstop task saved but at least one solve is not certified; inspect manifests")
    print(f"Completed task {task_id}/{len(CASES)*len(DELAY_YEARS)}", flush=True)


if __name__ == "__main__":
    main()

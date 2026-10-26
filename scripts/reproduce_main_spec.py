#!/usr/bin/env python
"""Print or run the main-specification EZDelay run suite.

Default behavior is a dry run that documents the exact local and cluster
commands. Use --execute-local for the local benchmark scripts and --submit-cluster
for SGE array jobs.
"""

import argparse
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


LOCAL_RUNS = (
    ("full-tree baseline", ("python", "-m", "scripts.run_revised_adjoint")),
)

# Array lengths describe the complete configured grids. Certified bundled cases
# can be fewer where cluster runs failed the recorded stationarity checks.
CLUSTER_RUNS = (
    {
        "name": "mean parameter vector",
        "array": "1-3",
        "command": str(ROOT / "scripts" / "run_mean_parameter_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "revised-mean-RA10-EIS15-logterminal-no-learning-v2", "TEST_MODE": "0"},
    },
    {
        "name": "delay frontier baseline",
        "array": "1-4",
        "command": str(ROOT / "scripts" / "run_delay_frontier_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "revised-frontier-RA10-EIS15-logterminal-no-learning-v2",
                "FRONTIER_GRID": "fixed_learning", "FRONTIER_PARAMETER_SOURCE": "mean",
                "DELAY_YEARS": "0,5,10,15",
                "TEST_MODE": "0"},
    },
    {
        "name": "delay frontier preference and learning robustness",
        "array": "1-20",
        "command": str(ROOT / "scripts" / "run_delay_frontier_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "revised-frontier-robustness-RA10-EIS15-logterminal-v3",
                "FRONTIER_GRID": "fixed_learning", "FRONTIER_PARAMETER_SOURCE": "robustness",
                "DELAY_YEARS": "0,5,10,15",
                "FRONTIER_PARAMETER_SPECS": "high_ra,high_eis,endogenous_learning,low_ra,low_eis",
                "TEST_MODE": "0"},
    },
    {
        "name": "tree structure and fragility probability robustness",
        "array": "1-15",
        "command": str(ROOT / "scripts" / "run_tree_robustness_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "reproduced-tree-robustness"},
    },
    {
        "name": "damage function and tipping robustness",
        "array": "1-24",
        "command": str(ROOT / "scripts" / "run_damage_robustness_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "reproduced-damage-robustness"},
    },
    {
        "name": "partial mitigation",
        "array": "1-63",
        "command": str(ROOT / "scripts" / "run_partial_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "revised-partial-RA10-EIS15-logterminal-no-learning-v2", "TEST_MODE": "0"},
    },
    {
        "name": "preference grid (RA 7/10/15/45, EIS 0.9/1.5/2.2/3, PRTP 0.2/0.5/2.4%)",
        "array": "1-144",
        "command": str(ROOT / "scripts" / "run_preference_grid_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "revised-preference-grid-RA10-EIS15-logterminal-no-learning-v3", "TEST_MODE": "0"},
    },
    {
        "name": "technology grid",
        "array": "1-27",
        "command": str(ROOT / "scripts" / "run_technology_grid_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "revised-technology-grid-RA10-EIS15-logterminal-no-learning-v2", "TEST_MODE": "0"},
    },
    {
        "name": "backstop delay sensitivities (baseline + 7 curves x 3 delays)",
        "array": "1-24",
        "command": str(ROOT / "scripts" / "run_backstop_delay_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "revised-backstop-delay-RA10-EIS15-logterminal-curvature-v3"},
    },
    {
        "name": "post-2250 growth sensitivity (13 finite cases x 3 delays)",
        "array": "1-39",
        "command": str(ROOT / "scripts" / "run_growth_sensitivity_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "revised-growth-delay-RA10-EIS15-post2250-v1",
                "DELAY_YEARS": "5,10,15"},
    },
    {
        "name": "marginal abatement cost shifts (11 x 11 x 3 delays)",
        "array": "1-363",
        "command": str(ROOT / "scripts" / "run_mac_shift_grid_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "reproduced-mac-shift"},
    },
    {
        "name": "Gaussian ensemble (10,000 draws x 3 delays)",
        "array": "1-30000",
        "command": str(ROOT / "scripts" / "run_ensemble_delayed_array_job.sh"),
        "env": {"OUTPUT_FOLDER": "reproduced-ensemble-N10000",
                "N_SAMPLES": "10000", "TEST_MODE": "0"},
    },
)


def env_prefix(env):
    return " ".join(f"{key}={value}" for key, value in env.items())


def cluster_command(run, mem, ncpus):
    return (
        f"grid_run --grid_mem={mem} "
        f"--grid_submit=batch --grid_array={run['array']} "
        f"--grid_ncpus={ncpus} {shlex.quote(run['command'])} {env_prefix(run['env'])}"
    )


def print_plan(args):
    print("EZDelay main-specification run suite\n")
    print("Local benchmark runs:")
    for label, command in LOCAL_RUNS:
        print(f"  {label}: {' '.join(command)}")

    print("\nCluster array runs:")
    for run in CLUSTER_RUNS:
        print(f"  {run['name']}: {cluster_command(run, args.grid_mem, args.grid_ncpus)}")


def execute_local():
    for label, command in LOCAL_RUNS:
        print(f"\nRunning local step: {label}")
        subprocess.run(command, cwd=ROOT, check=True)


def submit_cluster(args):
    for run in CLUSTER_RUNS:
        command = cluster_command(run, args.grid_mem, args.grid_ncpus)
        print(f"\nSubmitting cluster step: {run['name']}")
        subprocess.run(command, shell=True, check=True, env=os.environ.copy())


def main():
    parser = argparse.ArgumentParser(
        description="Document or run the EZDelay main-specification run suite."
    )
    parser.add_argument(
        "--execute-local",
        action="store_true",
        help="Run the local benchmark scripts after printing the plan.",
    )
    parser.add_argument(
        "--submit-cluster",
        action="store_true",
        help="Submit the cluster array jobs after printing the plan.",
    )
    parser.add_argument("--grid-mem", default="200G", help="Memory for grid_run.")
    parser.add_argument("--grid-ncpus", default="8", help="CPU count for grid_run.")
    args = parser.parse_args()

    print_plan(args)

    if args.execute_local:
        execute_local()

    if args.submit_cluster:
        submit_cluster(args)

    if not args.execute_local and not args.submit_cluster:
        print("\nDry run only. Add --execute-local and/or --submit-cluster to run commands.")


if __name__ == "__main__":
    main()

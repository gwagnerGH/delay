#!/usr/bin/env python
"""Print or run the main-specification EZDelay run suite.

Default behavior is a dry run that documents the exact local and cluster
commands. Use --execute-local for the local benchmark scripts and --submit-cluster
for SGE array jobs.
"""

import argparse
import os
import subprocess


LOCAL_RUNS = (
    ("single benchmark", ("python", "scripts/main.py")),
    ("default delayed action", ("python", "scripts/main_delayed.py")),
)

CLUSTER_RUNS = (
    {
        "name": "mean parameter vector",
        "array": "1-3",
        "command": "bash scripts/run_mean_parameter_array_job.sh",
        "env": {
            "OUTPUT_FOLDER": "mean-parameter-BY2025-samegrid-run0-v3",
            "TEST_MODE": "0",
        },
    },
    {
        "name": "annual delay frontier",
        "array": "1-21",
        "command": "bash scripts/run_delay_frontier_array_job.sh",
        "env": {
            "OUTPUT_FOLDER": "delay-frontier-BY2025-samegrid-run0-v4",
            "FRONTIER_GRID": "annual",
            "FRONTIER_PARAMETER_SOURCE": "mean",
            "TEST_MODE": "0",
        },
    },
    {
        "name": "annual delay-frontier robustness",
        "array": "1-105",
        "command": "bash scripts/run_delay_frontier_array_job.sh",
        "env": {
            "OUTPUT_FOLDER": "delay-frontier-BY2025-robustness-v2",
            "FRONTIER_GRID": "annual",
            "FRONTIER_PARAMETER_SOURCE": "robustness",
            "FRONTIER_PARAMETER_SPECS": "low_eis,high_eis,high_ra,low_ra,no_endogenous_learning",
            "IMPORT_DAMAGES": "1",
            "TEST_MODE": "0",
        },
    },
    {
        "name": "partial mitigation",
        "array": "1-63",
        "command": "bash scripts/run_partial_array_job.sh",
        "env": {
            "OUTPUT_FOLDER": "partial-mitigation-BY2025-samegrid-run0-cap-v3",
            "TEST_MODE": "0",
        },
    },
    {
        "name": "tree robustness",
        "array": "1-15",
        "command": "bash scripts/run_tree_robustness_array_job.sh",
        "env": {
            "OUTPUT_FOLDER": "tree-robustness-BY2025-samegrid-run0-v3",
            "TEST_MODE": "0",
        },
    },
    {
        "name": "preference-grid robustness",
        "array": "1-81",
        "command": "bash scripts/run_preference_grid_array_job.sh",
        "env": {
            "OUTPUT_FOLDER": "preference-grid-BY2025-samegrid-run0-v3",
            "TEST_MODE": "0",
        },
    },
    {
        "name": "technology-grid robustness",
        "array": "1-27",
        "command": "bash scripts/run_technology_grid_array_job.sh",
        "env": {
            "OUTPUT_FOLDER": "technology-grid-BY2025-samegrid-run0-v3",
            "TEST_MODE": "0",
        },
    },
    {
        "name": "damage-specification robustness",
        "array": "1-24",
        "command": "bash scripts/run_damage_robustness_array_job.sh",
        "env": {
            "OUTPUT_FOLDER": "damage-robustness-BY2025-samegrid-run0-v3",
            "TEST_MODE": "0",
        },
    },
    {
        "name": "Gaussian ensemble delayed action",
        "array": "1-9000",
        "command": "bash scripts/run_ensemble_delayed_array_job.sh",
        "env": {
            "OUTPUT_FOLDER": "ensemble-BY2025-samegrid-run0gauss-v5",
            "TEST_MODE": "0",
        },
    },
)


def env_prefix(env):
    return " ".join(f"{key}={value}" for key, value in env.items())


def cluster_command(run, mem, ncpus):
    return (
        f"grid_run --grid_mem={mem} "
        f"--grid_submit=batch --grid_array={run['array']} "
        f"--grid_ncpus={ncpus} {run['command']} {env_prefix(run['env'])}"
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
        subprocess.run(command, check=True)


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
    parser.add_argument("--grid-mem", default="150G", help="Memory for grid_run.")
    parser.add_argument("--grid-ncpus", default="4", help="CPU count for grid_run.")
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

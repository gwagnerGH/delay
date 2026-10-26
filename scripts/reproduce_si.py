"""Print or execute the SI learning-measure and cost-floor sensitivities."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def jobs():
    for reference_year in (2100, 2200):
        for task in (1, 2, 3):
            yield (
                f"quantity-learning reference {reference_year}, delay task {task}",
                [sys.executable, "-m", "scripts.main_technology_grid_cluster"],
                {
                    "SGE_TASK_ID": str(task),
                    "TECH_CHG_LEVELS": "1.5",
                    "TECH_SCALE_LEVELS": "1.5",
                    "REVISED_LEARNING_REFERENCE_YEAR": str(reference_year),
                    "OUTPUT_FOLDER": f"reproduced-quantity-learning-ref{reference_year}",
                },
            )
    yield (
        "long-run cost floor, no delay plus 5/10/15-year delays",
        [sys.executable, str(ROOT / "scripts/run_cost_floor_sensitivity.py"),
         "--output", str(ROOT / "data/new_outputs/reproduced_cost_floor")],
        {},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="Run the expensive model solves; default is a dry run")
    args = parser.parse_args()
    for label, command, overrides in jobs():
        print(label, flush=True)
        print("  " + " ".join(f"{key}={value}" for key, value in overrides.items())
              + " " + " ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, env={**os.environ, **overrides}, check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Compute Shapley values from five-year-grid delay decomposition outputs."""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict


PLAYERS = (
    "climate_damages",
    "endogenous_learning",
    "exogenous_tech_progress",
    "uncertainty_tipping",
)
DELAYS = (5, 10, 15, 20)
FULL_MASK = (1 << len(PLAYERS)) - 1


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value, label):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Could not parse {label} as float: {value!r}")


def parse_int(value, label):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Could not parse {label} as int: {value!r}")


def player_count(mask):
    return bin(mask).count("1")


def shapley_weight(mask, player_index, n_players):
    subset_size = player_count(mask)
    return (
        math.factorial(subset_size)
        * math.factorial(n_players - subset_size - 1)
        / math.factorial(n_players)
    )


def coalition_label(mask):
    active = [player for index, player in enumerate(PLAYERS) if mask & (1 << index)]
    return "|".join(active) if active else "none"


def values_from_results(rows):
    values = {}
    coalition_rows = []
    required = {"delay_year", "coalition_mask", "utility_loss"}
    missing_columns = required - set(rows[0].keys()) if rows else required
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    for row_number, row in enumerate(rows, start=2):
        delay = parse_int(row["delay_year"], f"delay_year on row {row_number}")
        mask = parse_int(row["coalition_mask"], f"coalition_mask on row {row_number}")
        utility_loss = parse_float(row["utility_loss"], f"utility_loss on row {row_number}")
        key = (delay, mask)
        if key in values:
            raise ValueError(f"Duplicate coalition result for delay={delay}, mask={mask}")
        values[key] = utility_loss
        coalition_rows.append({
            "delay_year": delay,
            "coalition_mask": mask,
            "coalition_players": row.get("coalition_players") or coalition_label(mask),
            "utility_loss": utility_loss,
        })

    expected = {(delay, mask) for delay in DELAYS for mask in range(FULL_MASK + 1)}
    missing = sorted(expected - set(values))
    extras = sorted(set(values) - expected)
    if missing:
        formatted = ", ".join(f"delay={delay}:mask={mask}" for delay, mask in missing[:20])
        if len(missing) > 20:
            formatted += f", ... ({len(missing)} total)"
        raise ValueError(f"Missing Shapley coalition rows: {formatted}")
    if extras:
        formatted = ", ".join(f"delay={delay}:mask={mask}" for delay, mask in extras[:20])
        if len(extras) > 20:
            formatted += f", ... ({len(extras)} total)"
        raise ValueError(f"Unexpected Shapley coalition rows: {formatted}")

    coalition_rows.sort(key=lambda r: (r["delay_year"], r["coalition_mask"]))
    return values, coalition_rows


def compute_shapley(values):
    n_players = len(PLAYERS)
    shapley_rows = []

    for delay in DELAYS:
        empty_value = values[(delay, 0)]
        full_value = values[(delay, FULL_MASK)]
        full_minus_empty = full_value - empty_value
        contributions = {}

        for player_index, player in enumerate(PLAYERS):
            bit = 1 << player_index
            phi = 0.0
            for mask in range(FULL_MASK + 1):
                if mask & bit:
                    continue
                with_player = mask | bit
                marginal = values[(delay, with_player)] - values[(delay, mask)]
                phi += shapley_weight(mask, player_index, n_players) * marginal
            contributions[player] = phi

        total_contribution = sum(contributions.values())
        efficiency_residual = total_contribution - full_minus_empty
        for player in PLAYERS:
            phi = contributions[player]
            share = phi / full_minus_empty if full_minus_empty != 0 else math.nan
            shapley_rows.append({
                "delay_year": delay,
                "player": player,
                "shapley_value_utility_loss": phi,
                "empty_value": empty_value,
                "full_value": full_value,
                "full_minus_empty": full_minus_empty,
                "efficiency_residual": efficiency_residual,
                "contribution_share": share,
            })

    return shapley_rows


def run_self_test():
    # Additive synthetic game: v(S) = base_delay + sum_i contribution_i.
    contributions = {
        "climate_damages": 1.5,
        "endogenous_learning": 2.0,
        "exogenous_tech_progress": -0.5,
        "uncertainty_tipping": 0.75,
    }
    values = {}
    for delay in DELAYS:
        base = delay / 100.0
        for mask in range(FULL_MASK + 1):
            value = base
            for index, player in enumerate(PLAYERS):
                if mask & (1 << index):
                    value += contributions[player]
            values[(delay, mask)] = value

    rows = compute_shapley(values)
    for row in rows:
        expected = contributions[row["player"]]
        actual = row["shapley_value_utility_loss"]
        if abs(actual - expected) > 1e-12:
            raise AssertionError(f"Self-test failed for {row['player']}: {actual} != {expected}")
        if abs(row["efficiency_residual"]) > 1e-12:
            raise AssertionError(f"Efficiency residual too large: {row['efficiency_residual']}")
    print("Shapley self-test passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Shapley values from true Shapley coalition outputs."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing output folders.")
    parser.add_argument("--folder", help="Shapley output folder name under --data-dir.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run an in-memory additive-game Shapley formula test and exit.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.folder:
        parser.error("--folder is required unless --self-test is set")

    analysis_dir = os.path.join(args.data_dir, args.folder, "analysis")
    results_path = os.path.join(analysis_dir, f"{args.folder}_consolidated_results.csv")
    coalition_path = os.path.join(analysis_dir, f"{args.folder}_coalition_values.csv")
    shapley_path = os.path.join(analysis_dir, f"{args.folder}_shapley_values.csv")

    rows = read_rows(results_path)
    values, coalition_rows = values_from_results(rows)
    shapley_rows = compute_shapley(values)

    write_rows(
        coalition_path,
        coalition_rows,
        ["delay_year", "coalition_mask", "coalition_players", "utility_loss"],
    )
    write_rows(
        shapley_path,
        shapley_rows,
        [
            "delay_year",
            "player",
            "shapley_value_utility_loss",
            "empty_value",
            "full_value",
            "full_minus_empty",
            "efficiency_residual",
            "contribution_share",
        ],
    )
    print(f"Wrote coalition values to: {coalition_path}")
    print(f"Wrote Shapley values to: {shapley_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

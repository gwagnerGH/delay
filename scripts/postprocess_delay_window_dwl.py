#!/usr/bin/env python3
"""Recompute delay-window consumption equivalents from saved policy vectors.

This script deliberately never calls an optimizer.  It reconstructs the
utility objects from the provenance columns in each consolidated-results CSV,
loads the corresponding cached damages, and evaluates the saved optimal and
constrained policies.  The resulting CSV is kept separate from the original
run output so that the original optimizer evidence remains immutable.

By default it processes the current top-level paper output folders in
``data/new_outputs``.  Historical/archive folders are intentionally excluded:
their provenance predates the fixed-learning output schema and cannot be
reconstructed safely from the current code.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from scripts._project_paths import PROJECT_ROOT, configure_paths
configure_paths()

from src.tree import TreeModel
from src.emit_baseline import BPWEmissionBaseline
from src.cost import BPWCost
from src.climate import BPWClimate
from src.damage import BPWDamage
from src.utility import EZUtility
from src.analysis.delayed_action import find_consumption_equivalence


OUTPUT_ROOT = Path(PROJECT_ROOT) / "data" / "new_outputs"
UTILITY_MATCH_ABS_TOL = 2e-6


def as_float(value, default=0.0):
    if value in (None, "", "nan", "NaN"):
        return default
    return float(value)


def as_bool(value, default=False):
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def decision_times(value):
    return [int(x) for x in str(value).split("|") if x != ""]


def damage_cache_tag(times, prob_scale, prefix):
    dt_tag = "-".join(str(int(x)) for x in times)
    prob_tag = f"{float(prob_scale):g}".replace(".", "p")
    return f"{prefix or ''}_DT{dt_tag}_PS{prob_tag}"


def tree_probability_scale(row):
    """Recover the only non-unit probability scalings used in the tree suite."""
    spec = row.get("tree_spec", "default")
    return {"higher_fragility_weight": 0.75,
            "lower_fragility_weight": 1.5}.get(spec, 1.0)


class UtilityFactory:
    """Cache imported damage objects; preferences/costs remain case-specific."""

    def __init__(self):
        self.components = {}

    def _components(self, row, times, prob_scale):
        baseline = int(as_float(row.get("baseline_num"), 2))
        emissions_step = int(as_float(row.get("emissions_time_step"), 5))
        dam_func = int(as_float(row.get("dam_func"), 0))
        tip_on = int(as_float(row.get("tip_on"), 1))
        d_unc = int(as_float(row.get("d_unc"), 1))
        t_unc = int(as_float(row.get("t_unc"), 1))
        tag_prefix = row.get("damage_file_tag", "_FIXEDLEARN_GRID5")
        tag = damage_cache_tag(times, prob_scale, tag_prefix)
        key = (tuple(times), prob_scale, baseline, emissions_step, dam_func,
               tip_on, d_unc, t_unc, tag_prefix)
        if key in self.components:
            return self.components[key]

        tree = TreeModel(decision_times=times, prob_scale=prob_scale)
        emissions = BPWEmissionBaseline(
            tree=tree, baseline_num=baseline, emissions_time_step=emissions_step
        )
        emissions.baseline_emission_setup()
        climate = BPWClimate(tree, emissions, draws=3 * 10**6)
        damage = BPWDamage(
            tree=tree, emit_baseline=emissions, climate=climate,
            mitigation_constants=np.arange(0, 1.1, 0.1)[::-1],
            draws=3 * 10**6,
        )
        filename = (
            f"simulated_damages_df{dam_func}_TP{tip_on}_SSP{baseline}"
            f"_BY{tree.base_year}_dunc{d_unc}_tunc{t_unc}{tag}"
        )
        damage.import_damages(file_name=filename)
        self.components[key] = (tree, emissions, damage)
        return self.components[key]

    def utility(self, row, which):
        times_column = "decision_times_optimal" if which == "optimal" else None
        if times_column is None or not row.get(times_column):
            times_column = "decision_times_partial" if which == "delayed" else times_column
        if times_column is None or not row.get(times_column):
            times_column = "decision_times_delayed"
        times = decision_times(row[times_column])
        prob_scale = tree_probability_scale(row)
        tree, emissions, damage = self._components(row, times, prob_scale)
        baseline = int(as_float(row.get("baseline_num"), 2))
        emit_at_0 = np.interp(2030, emissions.times, emissions.baseline_gtco2)
        cost = BPWCost(
            tree, emit_at_0=emit_at_0, baseline_num=baseline,
            tech_const=as_float(row.get("tech_chg"), 1.5),
            tech_scale=as_float(row.get("tech_scale"), 1.5),
            cons_at_0=86252.0,
            backstop_premium=as_float(row.get("bs_premium"), 10000.0),
            no_free_lunch=as_bool(row.get("no_free_lunch"), False),
            backstop_smoothing_width=as_float(row.get("backstop_smoothing_width"), 0.0),
            backstop_smoothing_mode=row.get("backstop_smoothing_mode") or "one_sided_huber",
            mac_horizontal_shift=as_float(row.get("mac_horizontal_shift"), 0.0),
            mac_vertical_shift=as_float(row.get("mac_vertical_shift"), 0.0),
        )
        return EZUtility(
            tree=tree, damage=damage, cost=cost,
            period_len=as_float(row.get("period_len"), 5.0),
            eis=as_float(row.get("eis")), ra=as_float(row.get("ra")),
            time_pref=as_float(row.get("pref")),
            cons_growth=as_float(row.get("growth"), 0.015),
        )


def result_key(row):
    return (row.get("sample_index", ""), row.get("delay_year", ""),
            row.get("run_type", ""), row.get("tree_spec", "default"))


def parse_policy(value):
    return np.asarray([float(x) for x in value.split("|") if x != ""], dtype=float)


def has_complete_native_delay_window_dwl(results):
    """Return whether every result row already carries a usable native DWL.

    Older runners wrote the delay-window columns with NaN values because they
    passed ``delay_window_years=None``. Column presence alone is therefore
    not evidence that recovery can be skipped.
    """
    required = (
        "delay_window_years",
        "delay_window_dwl_fraction",
        "delay_window_dwl_pct",
    )
    return bool(results) and all(
        all(math.isfinite(as_float(row.get(column), math.nan))
            for column in required)
        for row in results
    )


def load_policies_from_local_optima(results, local_path):
    """Select the stored local result that exactly corresponds to each reported utility."""
    targets = {}
    for row in results:
        key = result_key(row)
        targets[(key, "optimal")] = as_float(row.get("u_optimal"), np.nan)
        targets[(key, "delayed")] = as_float(row.get("u_delayed"), np.nan)

    selected = {}
    with local_path.open(newline="") as handle:
        for candidate in csv.DictReader(handle):
            key = result_key(candidate)
            scenario = candidate.get("scenario")
            target = targets.get((key, scenario))
            if target is None or not np.isfinite(target):
                continue
            utility = as_float(candidate.get("final_utility"), np.nan)
            if not np.isfinite(utility) or not candidate.get("mitigation"):
                continue
            distance = abs(utility - target)
            old = selected.get((key, scenario))
            if old is None or distance < old[0]:
                selected[(key, scenario)] = (distance, parse_policy(candidate["mitigation"]))
    return selected


def load_partial_policies(folder, row):
    """Partial runs save full node policies in CSV rather than local-optima CSV."""
    delay = int(as_float(row["delay_year"]))
    cap = int(as_float(row["partial_mitigation_pct"]))
    name = f"damage0_delay{delay}yr_pm{cap}"
    base = folder / "analysis"

    def node_policy(path):
        values = []
        with path.open(newline="") as handle:
            for entry in csv.DictReader(handle, delimiter=";"):
                if not (entry.get("Node") or "").isdigit():
                    break
                values.append(float(entry["Mitigation"]))
        return np.asarray(values, dtype=float)

    return (node_policy(folder / "partial" / f"{name}_partial_node_period_output.csv"),
            node_policy(folder / "optimal" / f"{name}_optimal_node_period_output.csv"))


def postprocess_folder(folder, overwrite=False, shard_index=None, shard_count=None):
    analysis = folder / "analysis"
    results_path = analysis / f"{folder.name}_consolidated_results.csv"
    output_path = analysis / f"{folder.name}_delay_window_dwl.csv"
    if shard_index is not None:
        output_path = analysis / (
            f"{folder.name}_delay_window_dwl_part-{shard_index:04d}-of-{shard_count:04d}.csv"
        )
    if not results_path.exists():
        return "no consolidated results"
    with results_path.open(newline="") as handle:
        results = list(csv.DictReader(handle))
    if not results:
        return "empty consolidated results"
    if has_complete_native_delay_window_dwl(results) and not overwrite:
        return "already has native delay-window DWL"
    if output_path.exists() and not overwrite:
        return "postprocessed CSV already exists"

    local_path = analysis / f"{folder.name}_local_optima.csv"
    if shard_index is not None:
        results = [row for number, row in enumerate(results, start=1)
                   if (number - 1) % shard_count == shard_index - 1]
        if not results:
            return "no cases in this shard"
    policies = None if results[0].get("run_type") == "partial_mitigation" else load_policies_from_local_optima(results, local_path)
    factory = UtilityFactory()
    output = []
    for number, row in enumerate(results, start=1):
        record = {
            "sample_index": row.get("sample_index"), "delay_year": row.get("delay_year"),
            "task_id": row.get("task_id"), "run_type": row.get("run_type"),
            "tree_spec": row.get("tree_spec", "default"),
            "source_results_file": results_path.name,
            "delay_window_years": row.get("delay_year"),
            "delay_window_dwl_fraction": "", "delay_window_dwl_pct": "",
            "status": "", "utility_match_optimal": "", "utility_match_delayed": "",
        }
        key = result_key(row)
        try:
            if policies is None:
                m_delayed, m_optimal = load_partial_policies(folder, row)
            else:
                delayed = policies.get((key, "delayed"))
                optimal = policies.get((key, "optimal"))
                if delayed is None or optimal is None:
                    raise ValueError("missing saved optimal or delayed policy")
                m_delayed, m_optimal = delayed[1], optimal[1]
                if max(delayed[0], optimal[0]) > UTILITY_MATCH_ABS_TOL:
                    raise ValueError("saved policy does not match reported optimizer utility")
            u_optimal = factory.utility(row, "optimal")
            u_delayed = factory.utility(row, "delayed")
            reported_opt = as_float(row.get("u_optimal"), np.nan)
            reported_delayed = as_float(row.get("u_delayed"), np.nan)
            evaluated_opt = float(np.asarray(u_optimal.utility(m_optimal)).reshape(-1)[0])
            evaluated_delayed = float(np.asarray(u_delayed.utility(m_delayed)).reshape(-1)[0])
            record["utility_match_optimal"] = evaluated_opt - reported_opt
            record["utility_match_delayed"] = evaluated_delayed - reported_delayed
            if (not np.isfinite(evaluated_opt) or not np.isfinite(evaluated_delayed)
                    or abs(evaluated_opt - reported_opt) > UTILITY_MATCH_ABS_TOL
                    or abs(evaluated_delayed - reported_delayed) > UTILITY_MATCH_ABS_TOL):
                raise ValueError("reconstructed utilities fail provenance check")
            g = find_consumption_equivalence(
                m_delayed, m_optimal, u_delayed, u_optimal,
                method="delay_window_proportional",
                compensation_years=as_float(row["delay_year"]),
            )
            if g is None or not np.isfinite(g):
                raise ValueError("compensation root failed")
            record["delay_window_dwl_fraction"] = g
            record["delay_window_dwl_pct"] = 100.0 * g
            record["status"] = "passed"
        except Exception as exc:  # keep failures visible and never fabricate a DWL
            record["status"] = f"skipped: {exc}"
        output.append(record)
        if number % 100 == 0 or number == len(results):
            print(f"{folder.name}: {number}/{len(results)} cases processed")

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    passed = sum(x["status"] == "passed" for x in output)
    return f"wrote {output_path.name}: {passed}/{len(output)} passed"


def selected_folders(root, requested):
    if requested:
        return [root / item for item in requested]
    return sorted(path for path in root.glob("paper-*") if path.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--folder", action="append", help="Top-level output folder to process; repeatable")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int, help="One-based shard index; requires --shard-count")
    parser.add_argument("--shard-count", type=int, help="Number of deterministic result-row shards")
    args = parser.parse_args()
    if (args.shard_index is None) != (args.shard_count is None):
        parser.error("--shard-index and --shard-count must be supplied together")
    if args.shard_index is not None and not 1 <= args.shard_index <= args.shard_count:
        parser.error("--shard-index must lie between 1 and --shard-count")
    for folder in selected_folders(args.root, args.folder):
        print(f"{folder.name}: {postprocess_folder(folder, args.overwrite, args.shard_index, args.shard_count)}")


if __name__ == "__main__":
    main()

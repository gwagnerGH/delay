#!/usr/bin/env python3
"""Reoptimize EZDelay under non-compressible shares of abatement costs.

One no-delay solution is shared by the three matched delay comparisons for
each floor. Existing completed solves are reused on restart; outputs are
never overwritten. All runs use the revised full-tree baseline otherwise.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_revised_adjoint import TREES
from src.revised_experiment import build_saved_revised_model


def _read_json(path):
    with path.open() as stream:
        return json.load(stream)


def _policy(path):
    with np.load(path, allow_pickle=False) as saved:
        return np.asarray(saved['mitigation'], float)


def _solve(folder, floor, delay, start):
    manifest_path = folder / 'manifest.json'
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if abs(manifest['cost_floor']-floor) > 1e-12 or manifest['fixed_years'] != delay:
            raise ValueError(f'Existing result does not match requested run: {folder}')
        return manifest
    if folder.exists():
        raise RuntimeError(f'Incomplete output directory; inspect before retrying: {folder}')
    folder.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT/'scripts/run_revised_adjoint.py'),
        '--tree', 'full', '--cost-floor', str(floor),
        '--fixed-years', str(delay), '--start', str(start),
        '--safe-refinement', '--output', str(folder)]
    with (folder.parent / (folder.name+'.log')).open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    return _read_json(manifest_path)


def _dwl(objective, optimal_policy, delayed_policy, years):
    target = objective.utility(optimal_policy)
    def difference(compensation):
        return (objective.delay_window_compensated_utility(
            delayed_policy, years, compensation)-target)
    if difference(1.0) > 1e-8:
        raise RuntimeError('Delayed policy has greater welfare than the optimal comparator')
    high = 1.1
    while difference(high) < 0:
        high *= 2
        if high > 1e12:
            raise RuntimeError('Could not bracket delay-window compensation')
    return 100*(brentq(difference, 1.0, high, xtol=1e-11)-1.0)


def _rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def _weighted_quantile(values, probabilities, quantile):
    order = np.argsort(values)
    return float(np.interp(quantile, np.cumsum(probabilities[order]), values[order]))


def _trajectory(folder, floor, policy_name):
    rows = _rows(folder/'five_year_state_nodes.csv')
    with (folder/'interval_summary.csv').open(newline='') as stream:
        prices = {int(row['year']): float(row['mean_interval_mac'])
                  for row in csv.DictReader(stream)}
    result = []
    for year in sorted({int(row['year']) for row in rows}):
        year_rows = [row for row in rows if int(row['year']) == year and
                     row['is_terminal_continuation'] == 'False']
        if not year_rows:
            continue
        weights = np.asarray([float(row['probability']) for row in year_rows])
        weights /= weights.sum()
        entry = dict(cost_floor=floor, policy=policy_name, year=year,
                     mean_interval_mac=prices.get(year, ''))
        for key, output in [('net_gtco2_per_year', 'net_emissions'),
                            ('removal_gtco2_per_year', 'removal')]:
            values = np.asarray([float(row[key]) for row in year_rows])
            entry[f'mean_{output}'] = float(weights @ values)
            entry[f'p10_{output}'] = _weighted_quantile(values, weights, .1)
            entry[f'p90_{output}'] = _weighted_quantile(values, weights, .9)
        controls = _rows(folder/'decision_state_nodes.csv')
        if year in {int(row['year']) for row in controls}:
            current = [row for row in controls if int(row['year']) == year]
            probs = np.asarray([float(row['probability']) for row in current])
            probs /= probs.sum()
            mitigation = np.asarray([1.0-float(row['net_gtco2_per_year'])/
                float(row['baseline_gtco2_per_year']) for row in current])
            entry.update(mean_mitigation=float(probs @ mitigation),
                         p10_mitigation=_weighted_quantile(mitigation, probs, .1),
                         p90_mitigation=_weighted_quantile(mitigation, probs, .9))
        result.append(entry)
    return result


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--floors', type=float, nargs='+', default=(0, .25, .4, .6))
    parser.add_argument('--delays', type=int, nargs='+', default=(5, 10, 15))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary, paths = [], []
    previous = {}
    baseline_start = ROOT/'data/revised_default/full_baseline_start.npz'
    certified_start = (ROOT/'data/new_outputs/revised_cluster/'
        'default_10yr_logterminal_2026-10-04_01/'
        'mean_parameter_vector_mean_params_delay10_task2/optimal/policy.npz')
    for floor in args.floors:
        label = f'floor_{int(round(100*floor)):02d}'
        group = args.output/label
        optimal_start = previous.get(0, certified_start if certified_start.is_file()
                                     else baseline_start)
        optimal = _solve(group/'optimal', floor, 0, optimal_start)
        previous[0] = group/'optimal/policy.npz'
        objective = build_saved_revised_model(ROOT/'data/revised_default/full_damages.npz',
            decision_times=TREES['full'], cost_floor=floor)
        optimal_policy = _policy(group/'optimal/policy.npz')
        paths += _trajectory(group/'optimal', floor, 'optimal')
        for delay in args.delays:
            delay_start = previous.get(delay, baseline_start)
            delayed = _solve(group/f'delay_{delay}', floor, delay, delay_start)
            previous[delay] = group/f'delay_{delay}/policy.npz'
            delayed_policy = _policy(group/f'delay_{delay}/policy.npz')
            compensation = _dwl(objective, optimal_policy, delayed_policy, delay)
            path_rows = _trajectory(group/f'delay_{delay}', floor, f'delay_{delay}')
            paths += path_rows
            delay_prices = {int(row['year']): float(row['mean_interval_mac'])
                for row in _rows(group/f'delay_{delay}/interval_summary.csv')}
            summary.append(dict(cost_floor=floor, delay_years=delay,
                dwl_pct_per_year=compensation,
                initial_price=float(_rows(group/'optimal/interval_nodes.csv')[0]['interval_mac']),
                reentry_price=delay_prices[2025+delay],
                optimal_certified=optimal['local_stationarity_certified'],
                delayed_certified=delayed['local_stationarity_certified'],
                optimal_foc_gap=optimal['final_feasible_foc_gap_dollars_per_ton'],
                delayed_foc_gap=delayed['final_feasible_foc_gap_dollars_per_ton'],
                optimal_utility=optimal['final_utility'],
                delayed_utility=delayed['final_utility']))
            _write_csv(args.output/'summary.csv', summary)
            _write_csv(args.output/'trajectories.csv', paths)
            print(f'floor={floor:.2f} delay={delay} DWL={compensation:.3f}% '
                  f'certified={optimal["local_stationarity_certified"]}/'
                  f'{delayed["local_stationarity_certified"]}', flush=True)


if __name__ == '__main__':
    main()

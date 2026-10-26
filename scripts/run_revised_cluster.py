"""Run one revised EZDelay cluster pair with the same economics as main.py.

Historical cluster wrappers call this through main_ensemble_delayed_cluster.py.
A result is a pair of independent adjoint solves, plus a delay-window
consumption-equivalent calculation. Per-task files avoid concurrent CSV writes.
"""
import csv
import fcntl
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from src.revised_experiment import build_saved_revised_model
from scripts.run_revised_adjoint import ROOT, TREES, main as solve
from scripts.revised_robustness_inputs import ensure_grid
from src.config.terminal_growth import TerminalGrowthSettings, NonFiniteTerminalError


def _times(value):
    return None if value is None else [int(x) for x in value]


def run_cluster_pair(*, sample_index, delay_year, param_vals, out_folder,
                     baseline, run_type='ensemble', tree_spec='default',
                     decision_times_baseline=None, decision_times_delay=None,
                     prob_scale_baseline=1.0, prob_scale_delay=1.0,
                     sample_label=None, output_metadata=None,
                     early_cap=None, cap_years=None,
                     zero_climate_damages=False, legacy_damage_settings=None,
                     mac_horizontal_shift=0.0,
                     mac_vertical_shift=0.0, delay_window_years=None, **kwargs):
    values = np.asarray(param_vals, float)
    if values.ndim != 2 or values.shape[1] not in (7, 8):
        raise ValueError('Expected seven legacy columns or eight revised ensemble columns')
    ra, eis, tech_const, tech_scale, prtp, old_premium, growth = values[sample_index, :7]
    terminal_settings = TerminalGrowthSettings.from_environment()
    try:
        terminal_metadata = terminal_settings.describe(
            ra=ra, eis=eis, prtp=prtp, modeled_growth=growth)
    except NonFiniteTerminalError as exc:
        # This is an economic inadmissibility, not an optimizer failure. Do not
        # silently change preferences or the growth assumption to obtain a solve.
        label = str(sample_label if sample_label is not None else sample_index)
        label = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in label)
        task = os.environ.get('SGE_TASK_ID', os.environ.get('TASK_ID', 'local'))
        folder = ROOT / 'data/new_outputs/revised_cluster' / out_folder / (
            f'{run_type}_{label}_delay{float(delay_year):g}_task{task}')
        folder.mkdir(parents=True, exist_ok=False)
        record = dict(status='terminal_inadmissible', run_type=run_type,
            sample_index=int(sample_index), sample_label=label, delay_year=float(delay_year),
            ra=float(ra), eis=float(eis), prtp=float(prtp), growth=float(growth),
            terminal_settings=terminal_settings.model_kwargs(), reason=str(exc),
            optimal_certified=False, delayed_certified=False)
        (folder / 'terminal_inadmissible.json').write_text(json.dumps(record, indent=2)+'\n')
        analysis = ROOT / 'data' / out_folder / 'analysis'
        analysis.mkdir(parents=True, exist_ok=True)
        _append_csv(analysis / f'{out_folder}_terminal_inadmissible.csv', [{
            **record, 'terminal_settings': json.dumps(record['terminal_settings'], sort_keys=True)}])
        print('REVISED_TERMINAL_INADMISSIBLE', json.dumps(record), flush=True)
        return record

    # Historical seven-column runners keep curvature at one unless explicitly
    # overridden. The revised ensemble stores curvature in its eighth column.
    backstop_curvature = (float(values[sample_index, 7]) if values.shape[1] == 8
                          else float(os.environ.get('REVISED_BACKSTOP_CURVATURE', '1')))
    if not np.isfinite(backstop_curvature) or backstop_curvature <= 0:
        raise ValueError('Backstop curvature must be finite and positive')
    if baseline != 2 and not os.environ.get('REVISED_DAMAGE_NPZ'):
        raise ValueError('Non-SSP2 runs require REVISED_DAMAGE_NPZ for the matching baseline')
    if zero_climate_damages:
        raise ValueError('Zero-damage runs require a separately validated revised objective')
    mac_horizontal_shift = float(mac_horizontal_shift)
    mac_vertical_shift = float(mac_vertical_shift)
    if (not np.isfinite(mac_horizontal_shift) or mac_horizontal_shift < 0 or
            not np.isfinite(mac_vertical_shift) or mac_vertical_shift < 0):
        raise ValueError('MAC shifts must be finite and nonnegative')
    if prob_scale_baseline != prob_scale_delay:
        raise ValueError('Optimal and delayed tree probabilities must match')
    dates0, dates1 = _times(decision_times_baseline), _times(decision_times_delay)
    if dates0 is None:
        dates0 = TREES['full']
    if dates1 is None:
        dates1 = dates0
    if dates0 != dates1:
        raise ValueError('Different optimal and delayed trees require paired revised damage grids')
    matches = [name for name, dates in TREES.items() if list(dates) == dates0]
    if not matches:
        raise ValueError('This tree requires a revised saved damage grid and matching runner')
    tree = matches[0]
    backstop = os.environ.get('REVISED_BACKSTOP', 's_curve')
    if (mac_horizontal_shift or mac_vertical_shift) and backstop != 's_curve':
        raise ValueError('MAC shifts require the revised smooth S-curve backstop')
    gamma = float(os.environ.get('REVISED_GAMMA', '0'))
    if backstop == 's_curve' and gamma:
        raise ValueError('REVISED_GAMMA requires REVISED_BACKSTOP=cubic or exp_cubic')
    if backstop != 's_curve' and backstop_curvature != 1.0:
        raise ValueError('Backstop curvature applies only to REVISED_BACKSTOP=s_curve')
    guard = float(os.environ.get('REVISED_NUMERICAL_GUARD', '4'))
    if backstop == 'exp_cubic' and guard > 1.5:
        raise ValueError('exp_cubic requires REVISED_NUMERICAL_GUARD <= 1.5')
    if guard <= 0:
        raise ValueError('REVISED_NUMERICAL_GUARD must be positive')
    damage = Path(os.environ.get('REVISED_DAMAGE_NPZ',
        ROOT / 'data/revised_default' / f'{tree}_damages.npz')).resolve()
    start = Path(os.environ.get('REVISED_START_NPZ',
        ROOT / 'data/revised_default' / f'{tree}_baseline_start.npz')).resolve()
    baseline_damage = dict(dam_func=0, tip_on=1, d_unc=1, t_unc=1)
    if run_type == 'tree_robustness':
        if os.environ.get('REVISED_DAMAGE_NPZ') or os.environ.get('REVISED_START_NPZ'):
            raise ValueError('Tree robustness selects matching saved inputs automatically')
        if legacy_damage_settings != baseline_damage:
            raise ValueError('Tree robustness requires baseline damage settings')
        if tree_spec == 'default':
            if tree != 'full' or prob_scale_baseline != 1.0:
                raise ValueError('Default tree does not match revised full tree')
        else:
            damage, start = ensure_grid('tree', tree_spec, dates0,
                prob_scale_baseline, baseline_damage)
    elif run_type == 'damage_robustness':
        if os.environ.get('REVISED_DAMAGE_NPZ') or os.environ.get('REVISED_START_NPZ'):
            raise ValueError('Damage robustness selects matching saved inputs automatically')
        settings = {key: int(legacy_damage_settings[key]) for key in baseline_damage}
        if tree != 'full' or prob_scale_baseline != 1.0:
            raise ValueError('Damage robustness requires the revised full tree')
        if settings != baseline_damage:
            spec = (f"df{settings['dam_func']}_TP{settings['tip_on']}_"
                    f"dunc{settings['d_unc']}_tunc{settings['t_unc']}")
            damage, start = ensure_grid('damage', spec, dates0, 1.0, settings)
    elif tree_spec not in ('default', 'short', 'full'):
        raise ValueError('Use tree_robustness for alternate tree specifications')
    economic_cap = float(os.environ.get('LBFGSB_POLICY_UPPER_BOUND', str(guard)))
    if not 0 < economic_cap <= guard:
        raise ValueError('LBFGSB_POLICY_UPPER_BOUND must be in (0, numerical guard]')
    if run_type == 'mac_shift_grid' and (backstop != 's_curve' or guard != 5.0 or economic_cap != 4.0):
        raise ValueError('Paper MAC-shift grid requires smooth S-curve, economic cap 4, numerical guard 5')
    label = str(sample_label if sample_label is not None else sample_index)
    label = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in label)
    task_suffix = os.environ.get('SGE_TASK_ID')
    if not task_suffix or not task_suffix.isdecimal():
        task_suffix = os.environ.get('TASK_ID', 'local')
    run_id = f'{run_type}_{label}_delay{float(delay_year):g}_task{task_suffix}'
    parent = ROOT / 'data/new_outputs/revised_cluster' / out_folder / run_id
    resume_solved = (run_type == 'partial_mitigation' and
        os.environ.get('REVISED_CLUSTER_RESUME_COMPLETE_SOLVES') == '1')
    if resume_solved:
        if not parent.is_dir() or (parent / 'result.json').exists():
            raise ValueError(f'Resume requires one incomplete solved case: {parent}')
        for scenario in ('optimal', 'delay'):
            for name in ('manifest.json', 'policy.npz', 'interval_nodes.csv',
                         'decision_state_nodes.csv'):
                if not (parent / scenario / name).is_file():
                    raise ValueError(f'Cannot resume missing {scenario}/{name}: {parent}')
    else:
        if parent.exists():
            raise FileExistsError(parent)
        parent.mkdir(parents=True)
    common = ['--tree', tree, '--damage', str(damage), '--start', str(start),
              '--prob-scale', str(prob_scale_baseline),
              '--ra', str(ra), '--eis', str(eis), '--prtp', str(prtp),
              '--growth', str(growth), '--tech-const', str(tech_const),
              '--tech-scale', str(tech_scale), '--backstop', backstop,
              '--gamma', str(gamma), '--backstop-curvature', str(backstop_curvature),
              '--mac-horizontal-shift', str(mac_horizontal_shift),
              '--mac-vertical-shift', str(mac_vertical_shift),
              '--guard', str(guard), '--safe-refinement'] + terminal_settings.cli_args()
    learning_reference_year = os.environ.get('REVISED_LEARNING_REFERENCE_YEAR')
    if learning_reference_year is not None:
        common += ['--learning-reference-year', learning_reference_year]
    if economic_cap < guard:
        common += ['--economic-upper-bound', str(economic_cap)]
    if os.environ.get('REVISED_CLUSTER_MAXITER'):
        common += ['--maxiter', os.environ['REVISED_CLUSTER_MAXITER']]
    if os.environ.get('REVISED_CLUSTER_REFINEMENT_MAXITER'):
        common += ['--refinement-maxiter', os.environ['REVISED_CLUSTER_REFINEMENT_MAXITER']]
    if os.environ.get('REVISED_CLUSTER_OBJECTIVE_SCALE'):
        common += ['--objective-scale', os.environ['REVISED_CLUSTER_OBJECTIVE_SCALE']]
    if os.environ.get('REVISED_CLUSTER_COORDINATE_STEPS'):
        common += ['--coordinate-refinement-steps', os.environ['REVISED_CLUSTER_COORDINATE_STEPS']]
    if os.environ.get('REVISED_CLUSTER_COUPLED_ROUNDS'):
        common += ['--coupled-refinement-rounds', os.environ['REVISED_CLUSTER_COUPLED_ROUNDS']]
    if os.environ.get('REVISED_CLUSTER_INITIAL_MAXITER'):
        common += ['--initial-maxiter', os.environ['REVISED_CLUSTER_INITIAL_MAXITER']]
    same_policy = (float(delay_year) == 0 or
                   (early_cap is not None and early_cap >= economic_cap))
    for name, fixed in ([] if resume_solved else [('optimal', 0), ('delay', delay_year)]):
        if name == 'delay' and same_policy:
            shutil.copytree(parent / 'optimal', parent / 'delay')
            continue
        extra = []
        scenario_start = start
        retry_source = os.environ.get('REVISED_CLUSTER_RETRY_SOURCE')
        if retry_source:
            candidates = list(Path(retry_source).glob(
                f'ensemble_{sample_index}_delay{float(delay_year):g}_task*'
            ))
            if len(candidates) > 1:
                raise ValueError(f'Ambiguous original ensemble case for {sample_index}, {delay_year}')
            if candidates:
                saved_policy = candidates[0] / name / 'policy.npz'
                if saved_policy.is_file():
                    scenario_start = saved_policy.resolve()
        scenario_common = common + ['--start', str(scenario_start)]
        if name == 'delay' and early_cap is not None:
            extra = ['--early-cap', str(early_cap), '--cap-years',
                     str(cap_years if cap_years is not None else delay_year)]
            fixed = 0
        solve(scenario_common + ['--fixed-years', str(fixed), '--output', str(parent / name)] + extra)
    optimal_recovered_from_delayed = False
    if run_type == 'mac_shift_grid':
        initial_optimal = json.loads((parent / 'optimal/manifest.json').read_text())
        initial_delayed = json.loads((parent / 'delay/manifest.json').read_text())
        if initial_delayed['final_utility'] > initial_optimal['final_utility'] + 1e-10:
            # A delayed policy is feasible for the unconstrained problem.
            # Use it to escape an inferior unconstrained local solution.
            solve(common + ['--start', str(parent / 'delay/policy.npz'),
                            '--fixed-years', '0',
                            '--output', str(parent / 'optimal_from_delay')])
            recovered = json.loads((parent / 'optimal_from_delay/manifest.json').read_text())
            if recovered['final_utility'] < initial_delayed['final_utility'] - 1e-10:
                raise RuntimeError('Delayed-policy warm start did not recover a valid optimal comparator')
            shutil.move(str(parent / 'optimal'), str(parent / 'optimal_initial'))
            shutil.move(str(parent / 'optimal_from_delay'), str(parent / 'optimal'))
            optimal_recovered_from_delayed = True
    optimal = json.loads((parent / 'optimal/manifest.json').read_text())
    delayed = json.loads((parent / 'delay/manifest.json').read_text())
    with np.load(parent / 'delay/policy.npz') as saved:
        delayed_policy = np.asarray(saved['mitigation'], float)
    objective = build_saved_revised_model(damage, decision_times=dates0, ra=ra,
        prob_scale=prob_scale_baseline,
        eis=eis, prtp=prtp, growth=growth, tech_const=tech_const,
        tech_scale=tech_scale, learning_reference_year=(
            int(learning_reference_year) if learning_reference_year is not None else None),
        backstop=backstop, gamma=gamma,
        backstop_curvature=backstop_curvature,
        mac_horizontal_shift=mac_horizontal_shift,
        mac_vertical_shift=mac_vertical_shift,
        baseline_num=baseline, **terminal_settings.model_kwargs())
    window = float(delay_window_years if delay_window_years is not None else delay_year)
    for name, manifest in (('optimal', optimal), ('delay', delayed)):
        if manifest.get('terminal_continuation') != terminal_metadata:
            raise ValueError(f'{name} solve uses different or missing terminal settings; cannot reuse')
    target = optimal['final_utility']
    def welfare_gap(factor):
        return objective.delay_window_compensated_utility(delayed_policy, window, factor)-target
    gap_at_one = welfare_gap(1.0)
    if not np.isclose(gap_at_one, delayed['final_utility']-target,
                      rtol=0, atol=1e-8):
        raise RuntimeError('Welfare-equivalent objective does not match solved delayed utility')
    if same_policy or abs(gap_at_one) <= 1e-9:
        factor = 1.0
    else:
        if gap_at_one > 0:
            raise RuntimeError('Delayed policy welfare exceeds its optimal comparator')
        high = 1.01
        while welfare_gap(high) < 0 and high < 1e12:
            high *= 2
        if welfare_gap(high) < 0:
            raise RuntimeError(f'Could not bracket delay-window welfare equivalent below factor {high:g}')
        factor = brentq(welfare_gap, 1.0, high, xtol=1e-10)
    # Compensation scales the consumption FLOW at every date during the delay.
    # It is already a rate paid each year, not a cumulative loss to divide again.
    annual = factor-1.0 if window else 0.0
    result = dict(model='revised_smooth_learning_five_year', run_type=run_type,
        tree_spec=tree_spec, prob_scale=prob_scale_baseline,
        damage_settings=legacy_damage_settings,
        sample_index=sample_index, sample_label=label, delay_year=delay_year,
        preferences=dict(ra=ra, eis=eis, prtp=prtp), growth=growth,
        terminal_continuation=terminal_metadata, terminal_start_year=2025+dates0[-1],
        tech_const=tech_const, tech_scale=tech_scale, backstop=backstop,
        gamma=gamma, backstop_curvature=backstop_curvature,
        mac_horizontal_shift=mac_horizontal_shift,
        mac_vertical_shift=mac_vertical_shift,
        optimal_recovered_from_delayed=optimal_recovered_from_delayed,
        economic_upper_bound=(economic_cap if economic_cap < guard else None),
        numerical_guard=guard,
        early_cap=early_cap, cap_years=cap_years,
        legacy_bs_premium_column_ignored=float(old_premium),
        flat_bau_after_2100=True, five_year_cost_accounting=True,
        technology_timing='five_year',
        learning_timing=(('cumulative_tons_over_fixed_BAU_to_reference_year'
                          if learning_reference_year is not None else
                          'BAU_weighted_mitigation_to_five_year_date')
                         if tech_scale else None),
        learning_reference_year=(int(learning_reference_year)
                                 if learning_reference_year is not None else None),
        pchip_damages=True, solver='adjoint_lbfgsb', damage_source=str(damage),
        optimal_utility=optimal['final_utility'], delayed_utility=delayed['final_utility'],
        dwl_definition="uniform proportional annual consumption compensation during delay window",
        delay_window_ce_fraction=factor-1.0,
        delay_window_ce_pct_per_year=100.0*annual,
        optimal_certified=optimal['local_stationarity_certified'],
        delayed_certified=delayed['local_stationarity_certified'],
        output_metadata=output_metadata or {})
    (parent / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    analysis = ROOT / 'data' / out_folder / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    with np.load(parent / 'optimal/policy.npz') as saved:
        optimal_policy = np.asarray(saved['mitigation'], float)
    with np.load(parent / 'delay/policy.npz') as saved:
        delay_policy = np.asarray(saved['mitigation'], float)
    with (parent/'optimal/interval_nodes.csv').open(newline='') as handle:
        optimal_price = float(next(csv.DictReader(handle))['interval_mac'])
    with (parent/'delay/interval_nodes.csv').open(newline='') as handle:
        delayed_price = float(next(csv.DictReader(handle))['interval_mac'])
    summary = dict(sample_index=label, delay_year=delay_year,
        **(dict(tree_spec=tree_spec, prob_scale=prob_scale_baseline,
                damage_settings=json.dumps(legacy_damage_settings, sort_keys=True))
           if run_type in ('tree_robustness', 'damage_robustness') else {}),
        run_type=run_type, ra=ra, eis=eis, pref=prtp,
        tech_chg=tech_const, tech_scale=tech_scale, growth=growth,
        terminal_growth_mode=terminal_settings.mode,
        terminal_growth=terminal_settings.growth,
        terminal_log_mean=terminal_settings.log_mean,
        terminal_log_sigma=terminal_settings.log_sigma,
        terminal_effective_annual_growth=terminal_metadata["effective_annual_net_growth"],
        terminal_multiplier=terminal_metadata["multiplier"],
        baseline_num=baseline, cost_formulation=backstop,
        gamma=gamma,
        economic_upper_bound=(economic_cap if economic_cap < guard else None),
        numerical_guard=guard,
        early_cap=early_cap, cap_years=cap_years,
        no_free_lunch=True,
        m_optimal_period0=float(optimal_policy[0]),
        m_delayed_period0=float(delay_policy[0]),
        u_optimal=optimal['final_utility'], u_delayed=delayed['final_utility'],
        dwl_definition="uniform proportional annual consumption compensation during delay window",
        delay_window_years=window,
        delay_window_dwl_fraction=factor-1.0,
        delay_window_dwl_pct=100.0*(factor-1.0),
        delay_window_dwl_pct_per_year=100.0*annual,
        carbon_price_optimal=optimal_price,
        carbon_price_delayed=delayed_price,
        model='revised_smooth_learning_five_year')
    mac_fields = (dict(mac_horizontal_shift=mac_horizontal_shift,
                       mac_vertical_shift=mac_vertical_shift,
                       mac_shift_formula='MAC_revised_s_curve(m+h)+v',
                       mac_policy_cap=4.0, backstop_curvature=backstop_curvature,
                       optimal_certified=bool(optimal['local_stationarity_certified']),
                       delayed_certified=bool(delayed['local_stationarity_certified']),
                       optimal_recovered_from_delayed=optimal_recovered_from_delayed)
                  if run_type == 'mac_shift_grid' else {})
    summary.update(mac_fields)
    if values.shape[1] == 8 or backstop_curvature != 1.0:
        summary['backstop_curvature'] = backstop_curvature
    per_task_export = run_type in ('ensemble', 'mac_shift_grid')
    if per_task_export:
        _write_csv(parent / 'summary.csv', [summary])
    else:
        _append_csv(analysis / f'{out_folder}_consolidated_results.csv', [summary])
    node_rows = []
    metric_rows = []
    for scenario in ('optimal', 'delay'):
        with (parent/scenario/'decision_state_nodes.csv').open(newline='') as handle:
            physical = {row['node']: row for row in csv.DictReader(handle)}
        with (parent/scenario/'interval_nodes.csv').open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row['node'] not in physical or row['year'] != physical[row['node']]['year']:
                    raise RuntimeError('Interval and physical node exports do not align')
                node_rows.append(dict(sample_index=label, delay_year=delay_year,
                    **mac_fields,
                    scenario='optimal' if scenario == 'optimal' else 'delayed',
                    year=row['year'], node=row['node'],
                    node_probability=row['probability'],
                    mitigation=row['mitigation'],
                    interval_mac=row['interval_mac'],
                    carbon_price=row['interval_mac']))
                metric_rows.append(dict(sample_index=label, delay_year=delay_year,
                    **mac_fields,
                    scenario='optimal' if scenario == 'optimal' else 'delayed',
                    ra=ra, eis=eis, prtp=prtp, backstop=backstop, gamma=gamma,
                    **({'backstop_curvature': backstop_curvature}
                       if run_type != 'mac_shift_grid' and
                       (values.shape[1] == 8 or backstop_curvature != 1.0) else {}),
                    **row, **{key: val for key, val in physical[row['node']].items()
                              if key not in row}))
    if per_task_export:
        _write_csv(parent / 'node_prices.csv', node_rows)
    else:
        _append_csv(analysis / f'{out_folder}_node_prices.csv', node_rows)
    _write_csv(parent / 'node_metrics.csv', metric_rows)
    if per_task_export:
        # Detailed state tables are needed to construct the node exports above,
        # but need not be stored for every Monte Carlo draw. Retain final
        # policies and compact certification manifests for later audit.
        if run_type == 'ensemble' and os.environ.get('REVISED_ENSEMBLE_LEAN_EXPORT', '1') != '0':
            try:
                from scripts.compact_revised_ensemble import compact_case
            except ModuleNotFoundError as exc:
                if exc.name != 'scripts.compact_revised_ensemble':
                    raise
                print('Optional ensemble compactor unavailable; retaining full state exports', flush=True)
            else:
                compact_case(parent, require_complete=False)
        # The marker is written last, after every per-task export is complete.
        (parent / 'complete.json').write_text(json.dumps({
            'sample_index': int(sample_index), 'delay_year': int(delay_year),
            'optimal_certified': bool(result['optimal_certified']),
            'delayed_certified': bool(result['delayed_certified']),
        }) + '\n')
    else:
        _append_csv(analysis / f'{out_folder}_node_metrics_v2.csv', metric_rows)
    print('REVISED_CLUSTER_RESULT', json.dumps(result), flush=True)
    return result


def _append_csv(path, rows):
    """Append one task's rows atomically under a cluster-safe file lock."""
    if not rows:
        return
    fields = list(rows[0])
    with path.open('a+', newline='') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing = csv.reader(handle)
            header = next(existing, None)
            if header is not None and header != fields:
                raise ValueError(f'Existing CSV schema differs from revised model: {path}')
            handle.seek(0, 2)
            writer = csv.DictWriter(handle, fieldnames=fields)
            if header is None:
                writer.writeheader()
            writer.writerows(rows)
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_csv(path, rows):
    """Write one task's CSV without touching any shared output file."""
    if not rows:
        raise ValueError(f'Cannot write an empty task export: {path}')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

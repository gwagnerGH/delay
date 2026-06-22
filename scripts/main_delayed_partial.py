#!/usr/bin/env python

"""
Author: Theo Moers
Columbia University 2025

Partial Mitigation Analysis Script
Tests the effects of partial first-period mitigation caps
(0%, 5%, 10%, ..., 100%) on a common decision grid.
Designed for SGE array job parallelization.
"""


import os
import sys
import pickle
import pprint
import numpy as np

from _project_paths import PROJECT_ROOT, configure_paths
configure_paths()

from src.tree import TreeModel
from src.emit_baseline import BPWEmissionBaseline
from src.cost import BPWCost
from src.climate import BPWClimate
from src.damage import BPWDamage
from src.utility import EZUtility
from src.analysis.climate_output import ClimateOutput
from src.analysis.delayed_action import get_delay_nodes, ConstraintAnalysis
from src.config import DEFAULT_CALENDAR_YEARS, DEFAULT_DECISION_TIMES
from src.tools import import_csv
from src.optimization import GeneticAlgorithm, GradientSearch
import main_ensemble_delayed_cluster as consolidated_outputs

# Configuration
runs = [0]
delay_years = [5, 10, 15]  # List of delay periods to test (e.g., [5, 10, 15])
partial_mitigation_pct = list(range(0, 101, 5))  # [0, 5, 10, 15, ..., 95, 100]
output_folder = "partial-mitigation-analysis-samegrid-run0-cap"
test_mode = os.environ.get('TEST_MODE', '0').lower() in ('1', 'true', 'yes')
import_damages = os.environ.get('IMPORT_DAMAGES', '0').lower() in ('1', 'true', 'yes')

DATA_DIR = os.path.join(str(PROJECT_ROOT), "data", "new_outputs")


def get_cluster_config():
    sge_task_id = os.environ.get('SGE_TASK_ID')
    if sge_task_id is None:
        print("Error: SGE_TASK_ID environment variable not found!")
        print("This script is designed to run as part of an SGE array job.")
        sys.exit(1)
    
    try:
        task_id = int(sge_task_id)
    except ValueError:
        print(f"Error: Invalid SGE_TASK_ID: {sge_task_id}")
        sys.exit(1)
    
    num_runs = len(runs)
    num_delays = len(delay_years)
    num_partial = len(partial_mitigation_pct)
    total_combinations = num_runs * num_delays * num_partial

    task_index = task_id - 1
    
    if task_index >= total_combinations:
        print(f"Error: Task ID {task_id} exceeds total combinations ({total_combinations})")
        print(f"runs = {runs} (length {num_runs})")
        print(f"delay_years = {delay_years} (length {num_delays})")
        print(f"partial_mitigation_pct = {partial_mitigation_pct} (length {num_partial})")
        print(f"Expected task range: 1-{total_combinations}")
        sys.exit(1)
    
    # Map task index to (run_index, delay_year, partial_mitigation_level)
    # Grid structure: iterate through partial_mit fastest, then delays, then runs
    run_idx = task_index // (num_delays * num_partial)
    remainder = task_index % (num_delays * num_partial)
    delay_idx = remainder // num_partial
    partial_idx = remainder % num_partial
    
    run_index = runs[run_idx]
    delay_year = delay_years[delay_idx]
    partial_mit_pct = partial_mitigation_pct[partial_idx]
    partial_mit_value = partial_mit_pct / 100.0  # Convert to [0.0, 1.0]
    
    out_folder = os.environ.get('OUTPUT_FOLDER', output_folder)
    sge_task_first = os.environ.get('SGE_TASK_FIRST', 'Unknown')
    sge_task_last = os.environ.get('SGE_TASK_LAST', 'Unknown')
    job_id = os.environ.get('JOB_ID', 'Unknown')
    
    print(f"\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {total_combinations}")
    print(f"  Mapping: run={run_index} (runs[{run_idx}]), delay={delay_year} (delay_years[{delay_idx}]), partial_mit={partial_mit_pct}% (index {partial_idx})")
    print(f"  Array range: {sge_task_first} to {sge_task_last}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")
    print(f"  Output folder: {out_folder}")
    print(f"\nConfiguration:")
    print(f"  All runs: {runs}")
    print(f"  All delay years: {delay_years}")
    print(f"  All partial mitigation levels: {partial_mitigation_pct}")
    
    return run_index, task_id, delay_year, partial_mit_pct, partial_mit_value, out_folder


def setup_cluster_directories(out_folder):
    """
    Create directory structure for outputs.
    """
    directories = [
        os.path.join(DATA_DIR, out_folder),
        os.path.join(DATA_DIR, out_folder, 'optimal'),
        os.path.join(DATA_DIR, out_folder, 'partial'),
        os.path.join(DATA_DIR, out_folder, 'analysis'),
        os.path.join(DATA_DIR, out_folder, 'logs')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("\nCreated directories:")
    for d in directories:
        print(f"  {d}")
    
    return directories


def partial_common_years(baseline_tree, partial_tree, delay_year):
    """Common calendar grid for partial-mitigation raw outputs."""

    return sorted(set(DEFAULT_CALENDAR_YEARS
                      + list(baseline_tree.calendar_years)
                      + list(partial_tree.calendar_years)
                      + [baseline_tree.base_year + delay_year]))


def load_parameter_combination(run_index):
    """
    Load parameter combinations from research_runs.csv
    """
    data_csv_for_import = "research_runs"
    
    print(f"\nLoading parameters from: data/{data_csv_for_import}.csv")
    
    header, indices, data = import_csv(data_csv_for_import, delimiter=",", indices=2)

    return header, indices, data


def run_partial_mitigation_analysis(run_index, task_id, delay_year, partial_mit_pct, partial_mit_value, 
                                    name, header, data, out_folder, test_mode, import_damages):
    """
    Run optimization with partial mitigation constraint and compare against baseline.
    
    Parameters:
    -----------
    run_index : int
        Index into parameter combinations
    delay_year : int
        Number of years to delay action
    partial_mit_pct : int
        Percentage of mitigation allowed during delay (0-100)
    partial_mit_value : float
        Mitigation value allowed during delay (0.0-1.0)
    name : str
        Name of parameter scenario
    header : list
        Parameter names
    data : array
        Parameter values
    out_folder : str
        Output folder name
    test_mode : bool
        Whether to run in test mode (fewer iterations)
    import_damages : bool
        Whether to import pre-computed damages
    """
    print(f"\n{'='*80}")
    print(f"PARTIAL MITIGATION ANALYSIS")
    print(f"  Delay period: {delay_year} years")
    print(f"  Partial mitigation cap: {partial_mit_pct}% (value: {partial_mit_value:.4f})")
    print(f"{'='*80}\n")
    
    # Extract parameters
    ra, eis, pref, growth, tech_chg, tech_scale, dam_func, \
        baseline_num, tip_on, bs_premium, d_unc, t_unc, \
        no_free_lunch = data[run_index]
    
    baseline_num = int(baseline_num)
    dam_func = int(dam_func)
    tip_on = int(tip_on)
    d_unc = int(d_unc)
    t_unc = int(t_unc)
    no_free_lunch = int(no_free_lunch)
    
    print('\n**Model Parameters:')
    model_params = [ra, eis, pref, growth, tech_chg, tech_scale,
                    dam_func, baseline_num, tip_on, bs_premium, d_unc,
                    t_unc, no_free_lunch]
    pprint.pprint(dict(zip(header, model_params)))
    
    if test_mode:
        print("\n***RUNNING IN TEST MODE***")
        N_generations_ga = 2
        N_iters_gs = 2
    else:
        print("\n***RUNNING IN FULL MODE***")
        N_generations_ga = 150
        N_iters_gs = 100
    
    # Create tree models. The unconstrained comparator and partial-mitigation
    # run must use the same grid so welfare differences isolate the mitigation
    # cap rather than mixing in a decision-timing change.
    decision_times_baseline = DEFAULT_DECISION_TIMES.copy()
    decision_times_baseline[1] = delay_year
    decision_times_partial = decision_times_baseline.copy()
    
    print(f"\nDecision times:")
    print(f"  Comparator grid: {decision_times_baseline}")
    print(f"  Partial grid:    {decision_times_partial}")
    
    t_baseline = TreeModel(decision_times=decision_times_baseline, prob_scale=1.0)
    t_partial = TreeModel(decision_times=decision_times_partial, prob_scale=1.0)
    
    # Emission baseline models
    baseline_emission_model_baseline = BPWEmissionBaseline(tree=t_baseline,
                                                  baseline_num=baseline_num)
    baseline_emission_model_baseline.baseline_emission_setup()
    
    baseline_emission_model_partial = BPWEmissionBaseline(tree=t_partial,
                                                  baseline_num=baseline_num)
    baseline_emission_model_partial.baseline_emission_setup()
    
    # Climate models
    draws = 3 * 10**6
    climate_baseline = BPWClimate(t_baseline, baseline_emission_model_baseline, draws=draws)
    climate_partial = BPWClimate(t_partial, baseline_emission_model_partial, draws=draws)
    
    # Cost models
    emit_at_0_baseline = np.interp(2030, baseline_emission_model_baseline.times,
                          baseline_emission_model_baseline.baseline_gtco2)
    c_baseline = BPWCost(t_baseline, emit_at_0=emit_at_0_baseline, baseline_num=baseline_num,
                tech_const=tech_chg, tech_scale=tech_scale,
                cons_at_0=86252.0, # 2025 estimated from https://data.worldbank.org/indicator/NE.CON.TOTL.CD,
                backstop_premium=bs_premium,
                no_free_lunch=no_free_lunch)
    
    emit_at_0_partial = np.interp(2030, baseline_emission_model_partial.times,
                          baseline_emission_model_partial.baseline_gtco2)
    c_partial = BPWCost(t_partial, emit_at_0=emit_at_0_partial, baseline_num=baseline_num,
                tech_const=tech_chg, tech_scale=tech_scale,
                cons_at_0=86252.0, # 2025 estimated from https://data.worldbank.org/indicator/NE.CON.TOTL.CD
                backstop_premium=bs_premium,
                no_free_lunch=no_free_lunch)
    
    # Damage models
    d_m = 0.1
    mitigation_constants = np.arange(0, 1 + d_m, d_m)[::-1]
    
    print("\nRunning damage simulations...")
    df_baseline = BPWDamage(tree=t_baseline, emit_baseline=baseline_emission_model_baseline,
                   climate=climate_baseline, mitigation_constants=mitigation_constants,
                   draws=draws)
    
    df_partial = BPWDamage(tree=t_partial, emit_baseline=baseline_emission_model_partial,
                   climate=climate_partial, mitigation_constants=mitigation_constants,
                   draws=draws)

    baseline_damage_file_tag = consolidated_outputs.damage_cache_tag(
        decision_times_baseline, t_baseline.prob_scale
    )
    partial_damage_file_tag = consolidated_outputs.damage_cache_tag(
        decision_times_partial, t_partial.prob_scale
    )

    damsim_filename_baseline = ''.join(["simulated_damages_df", str(dam_func),
                               "_TP", str(tip_on), "_SSP", str(baseline_num),
                               "_BY", str(t_baseline.base_year),
                               "_dunc", str(d_unc), "_tunc", str(t_unc),
                               baseline_damage_file_tag])
    
    damsim_filename_partial = ''.join(["simulated_damages_df", str(dam_func),
                               "_TP", str(tip_on), "_SSP", str(baseline_num),
                               "_BY", str(t_partial.base_year),
                               "_dunc", str(d_unc), "_tunc", str(t_unc),
                               partial_damage_file_tag])
    
    if import_damages:
        try:
            df_baseline.import_damages(file_name=damsim_filename_baseline)
            df_partial.import_damages(file_name=damsim_filename_partial)
            print("Successfully imported damage simulations\n")
        except Exception as e:
            print(f"Warning: Could not import damages ({e})")
            print("Running damage simulation...")
            df_baseline.damage_simulation(filename=damsim_filename_baseline, save_simulation=True,
                                 dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                                 t_unc=t_unc)
            df_partial.damage_simulation(filename=damsim_filename_partial, save_simulation=True,
                                 dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                                 t_unc=t_unc)
    else:
        df_baseline.damage_simulation(filename=damsim_filename_baseline, save_simulation=True,
                             dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                             t_unc=t_unc)
        df_partial.damage_simulation(filename=damsim_filename_partial, save_simulation=True,
                             dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                             t_unc=t_unc)
    
    # Create utility instances
    u_baseline = EZUtility(tree=t_baseline, damage=df_baseline, cost=c_baseline, 
                          period_len=5.0, eis=eis, ra=ra, time_pref=pref, cons_growth=growth)

    u_partial = EZUtility(tree=t_partial, damage=df_partial, cost=c_partial, 
                         period_len=5.0, eis=eis, ra=ra, time_pref=pref, cons_growth=growth)

    print("Model components initialized\n")
    
    # =========================================================================
    # OPTIMAL SCENARIO (Baseline tree, no constraints)
    # =========================================================================
    print(f"\n{'='*80}")
    print("RUNNING OPTIMAL SCENARIO (same grid, no mitigation cap)")
    print(f"{'='*80}\n")
    
    fixed_indices_opt = None
    fixed_values_opt = None
    
    ga_model_opt = GeneticAlgorithm(
        pop_amount=400,
        num_generations=N_generations_ga,
        cx_prob=0.8, 
        mut_prob=0.50, 
        bound=1.5,
        num_feature=t_baseline.num_decision_nodes,
        utility=u_baseline, 
        fixed_values=fixed_values_opt,
        fixed_indices=fixed_indices_opt,
        print_progress=True
    )
    
    gs_model_opt = GradientSearch(
        var_nums=t_baseline.num_decision_nodes,  
        utility=u_baseline, 
        accuracy=5.e-7,
        iterations=N_iters_gs,
        fixed_values=fixed_values_opt,
        fixed_indices=fixed_indices_opt,
        print_progress=True
    )
    
    print("Running Genetic Algorithm (optimal)...")
    final_pop_opt, fitness_opt = ga_model_opt.run()
    sort_pop_opt = final_pop_opt[np.argsort(fitness_opt)][::-1]
    
    print("Running Gradient Search (optimal)...")
    m_optimal, u_optimal = gs_model_opt.run(initial_point_list=sort_pop_opt, topk=1)
    
    print(f"\nOptimal scenario results:")
    print(f"  First-period mitigation:  {m_optimal[0]:.6f}")
    print(f"  Carbon price:             ${c_baseline.price(0, m_optimal[0], 0):.2f} per ton")
    print(f"  Utility:                  {u_optimal:.10f}")
    
    # Save optimal scenario
    output_prefix_opt = f'{out_folder}/optimal/{name}_delay{delay_year}yr_pm{partial_mit_pct}_optimal'
    
    co_opt = ClimateOutput(u_baseline)
    co_opt.calculate_output(m_optimal)
    co_opt.save_output(m_optimal, prefix=output_prefix_opt)
    
    p_opt = {
        'df.d_rcomb': df_baseline.d_rcomb,
        'm_opt': m_optimal,
        'u_opt': u_optimal,
        'delay_action': False,
        'delay_years': 0,
        'partial_mitigation_pct': None,
        'partial_mitigation_value': None,
        'decision_times': t_baseline.decision_times,
        'parameters': dict(zip(header, model_params))
    }
    
    pickle_path_opt = os.path.join(DATA_DIR, out_folder, 'optimal',
                                    f'{name}_delay{delay_year}yr_pm{partial_mit_pct}_optimal_log.pickle')
    with open(pickle_path_opt, 'wb') as handle:
        pickle.dump(p_opt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Optimal scenario saved: {pickle_path_opt}\n")
    
    # =========================================================================
    # PARTIAL MITIGATION SCENARIO
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"RUNNING PARTIAL MITIGATION SCENARIO")
    print(f"  Mitigation cap: {partial_mit_pct}% (value: {partial_mit_value:.4f})")
    print(f"{'='*80}\n")
    
    # Set up constraints
    if partial_mit_pct == 100:
        # 100% means fully unconstrained, including the model's negative-emissions headroom.
        print("100% mitigation = unconstrained optimization on the same grid")
        fixed_indices_partial = None
        fixed_values_partial = None
        upper_bounds_partial = None
    else:
        # Allow mitigation up to the partial level in the first period.
        fixed_indices_partial = None
        fixed_values_partial = None
        upper_bounds_partial = np.full(t_partial.num_decision_nodes, 1.5)
        upper_bounds_partial[0] = partial_mit_value
        
        print(f"Constraint configuration:")
        print(f"  Total decision nodes:        {t_partial.num_decision_nodes}")
        print("  Upper-bound constrained node: 0")
        print(f"  First-period mitigation cap: {partial_mit_value:.6f}\n")
    
    ga_model_partial = GeneticAlgorithm(
        pop_amount=400,
        num_generations=N_generations_ga,
        cx_prob=0.8, 
        mut_prob=0.50, 
        bound=1.5,
        num_feature=t_partial.num_decision_nodes,
        utility=u_partial, 
        fixed_values=fixed_values_partial,
        fixed_indices=fixed_indices_partial,
        upper_bounds=upper_bounds_partial,
        print_progress=True
    )
    
    gs_model_partial = GradientSearch(
        var_nums=t_partial.num_decision_nodes,
        utility=u_partial, 
        accuracy=5.e-7,
        iterations=N_iters_gs,
        fixed_values=fixed_values_partial,
        fixed_indices=fixed_indices_partial,
        upper_bounds=upper_bounds_partial,
        print_progress=True
    )
    
    print("Running Genetic Algorithm (partial mitigation)...")
    final_pop_partial, fitness_partial = ga_model_partial.run()
    sort_pop_partial = final_pop_partial[np.argsort(fitness_partial)][::-1]
    
    print("Running Gradient Search (partial mitigation)...")
    m_partial, u_partial_result = gs_model_partial.run(initial_point_list=sort_pop_partial, topk=1)
    
    # Validate constraints
    if partial_mit_pct < 100:
        actual_value = m_partial[0]
        if actual_value - partial_mit_value > 1e-10:
            print("Warning: First-period mitigation exceeds cap!")
            print(f"  Cap:      {partial_mit_value:.10f}")
            print(f"  Actual:   {actual_value:.10f}")
        else:
            print(f"✓ First-period mitigation verified <= {partial_mit_value:.6f}")
    
    print(f"\nPartial mitigation scenario results:")
    cap_text = "unconstrained" if partial_mit_pct == 100 else f"cap {partial_mit_value:.6f}"
    print(f"  First-period mitigation:  {m_partial[0]:.6f} ({cap_text})")
    print(f"  Carbon price:             ${c_partial.price(0, m_partial[0], 0):.2f} per ton")
    print(f"  Utility:                  {u_partial_result:.10f}")
    
    # Save partial mitigation scenario
    output_prefix_partial = f'{out_folder}/partial/{name}_delay{delay_year}yr_pm{partial_mit_pct}_partial'
    
    co_partial = ClimateOutput(u_partial)
    co_partial.calculate_output(m_partial)
    co_partial.save_output(m_partial, prefix=output_prefix_partial)
    
    p_partial = {
        'df.d_rcomb': df_partial.d_rcomb,
        'm_opt': m_partial,
        'u_opt': u_partial_result,
        'delay_action': True,
        'delay_years': delay_year,
        'partial_mitigation_pct': partial_mit_pct,
        'partial_mitigation_value': partial_mit_value,
        'decision_times': t_partial.decision_times,
        'parameters': dict(zip(header, model_params))
    }
    
    pickle_path_partial = os.path.join(DATA_DIR, out_folder, 'partial',
                                       f'{name}_delay{delay_year}yr_pm{partial_mit_pct}_partial_log.pickle')
    with open(pickle_path_partial, 'wb') as handle:
        pickle.dump(p_partial, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Partial mitigation scenario saved: {pickle_path_partial}\n")
    
    print(f"\n{'='*80}")
    print("CONSTRAINT ANALYSIS")
    print(f"{'='*80}\n")
    
    ca = ConstraintAnalysis(u_partial, u_baseline, m_partial, m_optimal)
    analysis_prefix = f'{out_folder}/analysis/{name}_delay{delay_year}yr_pm{partial_mit_pct}_analysis'
    ca.save_output(prefix=analysis_prefix)

    run_type = "partial_mitigation"
    comparison_type = "same_grid"
    decision_times_label = '|'.join(str(int(x)) for x in decision_times_baseline)
    partial_decision_times_label = '|'.join(str(int(x)) for x in decision_times_partial)
    params_for_nodes = {
        'ra': ra,
        'eis': eis,
        'pref': pref,
        'growth': growth,
        'tech_chg': tech_chg,
        'tech_scale': tech_scale,
        'bs_premium': bs_premium,
    }

    results_dict = {
        'sample_index': run_index,
        'scenario_name': name,
        'delay_year': delay_year,
        'task_id': task_id,
        'run_type': run_type,
        'comparison_type': comparison_type,
        'partial_mitigation_pct': partial_mit_pct,
        'partial_mitigation_value': float(partial_mit_value),
        'decision_times_optimal': decision_times_label,
        'decision_times_partial': partial_decision_times_label,

        'ra': float(ra),
        'eis': float(eis),
        'pref': float(pref),
        'tech_chg': float(tech_chg),
        'tech_scale': float(tech_scale),
        'bs_premium': float(bs_premium),
        'growth': float(growth),

        'baseline_num': int(baseline_num),
        'dam_func': int(dam_func),
        'tip_on': int(tip_on),
        'd_unc': int(d_unc),
        't_unc': int(t_unc),
        'no_free_lunch': bool(no_free_lunch),

        'm_optimal_period0': float(m_optimal[0]),
        'm_partial_period0': float(m_partial[0]),
        'mitigation_foregone': float(max(m_optimal[0] - m_partial[0], 0.0)),

        'u_optimal': float(u_optimal),
        'u_partial': float(u_partial_result),
        'utility_loss': float(ca.con_cost),
        'utility_loss_pct': float((ca.con_cost / u_optimal) * 100) if u_optimal != 0 else np.nan,

        'delta_c': float(ca.delta_c) if ca.delta_c is not None else np.nan,
        'delta_c_pct': float(ca.delta_c_pct) if ca.delta_c_pct is not None else np.nan,
        'delta_c_billions': float(ca.delta_c_billions) if ca.delta_c_billions is not None else np.nan,
        'delta_c_5yr': float(ca.delta_c_5yr) if ca.delta_c_5yr is not None else np.nan,
        'delta_c_5yr_pct': float(ca.delta_c_5yr_pct) if ca.delta_c_5yr_pct is not None else np.nan,
        'delta_c_5yr_billions': (
            float(ca.delta_c_5yr_billions)
            if ca.delta_c_5yr is not None else np.nan
        ),
        'delta_c_5yr_total_billions': (
            float(ca.delta_c_5yr_total_billions)
            if ca.delta_c_5yr is not None else np.nan
        ),
        'year0_cons_partial': float(ca.year0_cons_delayed),
        'delta_emission_gton': float(ca.delta_emission_gton),
        'deadweight_per_ton': float(ca.deadweight) if ca.deadweight is not None else np.nan,

        'carbon_price_partial': float(c_partial.price(0, m_partial[0], 0)),
        'carbon_price_optimal': float(c_baseline.price(0, m_optimal[0], 0)),
    }

    print("\nCalculating climate metrics for consolidated timeseries...")
    common_years = partial_common_years(t_baseline, t_partial, delay_year)
    exp_temp_opt, exp_conc_opt, exp_dam_opt = consolidated_outputs.calculate_period_climate_metrics(
        m_optimal, t_baseline, df_baseline, climate_baseline, baseline_emission_model_baseline
    )
    exp_temp_partial, exp_conc_partial, exp_dam_partial = consolidated_outputs.calculate_period_climate_metrics(
        m_partial, t_partial, df_partial, climate_partial, baseline_emission_model_partial
    )
    exp_cumemit_partial = consolidated_outputs.calculate_period_cumemit_metrics(
        m_partial, t_partial, baseline_emission_model_partial
    )
    exp_cumemit_opt = consolidated_outputs.calculate_period_cumemit_metrics(
        m_optimal, t_baseline, baseline_emission_model_baseline
    )
    mechanism_metrics = consolidated_outputs.build_delay_frontier_metrics(
        delay_year, t_baseline, t_partial, co_opt, co_partial, m_partial,
        exp_temp_partial, exp_conc_partial, exp_cumemit_partial, common_years,
        exp_cumemit_opt=exp_cumemit_opt, welfare_loss=ca.con_cost
    )
    results_dict.update(mechanism_metrics)

    csv_path = os.path.join(DATA_DIR, out_folder, 'analysis',
                            f'{out_folder}_consolidated_results.csv')
    print(f"Appending consolidated results to: {csv_path}")
    consolidated_outputs.append_results_to_csv(results_dict, csv_path)

    m_opt_mapped = consolidated_outputs.map_to_calendar_years(
        t_baseline, co_opt.expected_period_mitigation, common_years
    )
    T_opt_mapped = consolidated_outputs.map_to_calendar_years(
        t_baseline, exp_temp_opt, common_years
    )
    conc_opt_mapped = consolidated_outputs.map_to_calendar_years(
        t_baseline, exp_conc_opt, common_years
    )
    dam_opt_mapped = consolidated_outputs.map_to_calendar_years(
        t_baseline, exp_dam_opt, common_years
    )
    price_opt_mapped = consolidated_outputs.map_to_calendar_years(
        t_baseline, co_opt.expected_period_price, common_years
    )

    m_partial_mapped = consolidated_outputs.map_to_calendar_years(
        t_partial, co_partial.expected_period_mitigation, common_years
    )
    T_partial_mapped = consolidated_outputs.map_to_calendar_years(
        t_partial, exp_temp_partial, common_years
    )
    conc_partial_mapped = consolidated_outputs.map_to_calendar_years(
        t_partial, exp_conc_partial, common_years
    )
    dam_partial_mapped = consolidated_outputs.map_to_calendar_years(
        t_partial, exp_dam_partial, common_years
    )
    price_partial_mapped = consolidated_outputs.map_to_calendar_years(
        t_partial, co_partial.expected_period_price, common_years
    )

    timeseries_dict = {
        'sample_index': run_index,
        'scenario_name': name,
        'delay_year': delay_year,
        'task_id': task_id,
        'run_type': run_type,
        'comparison_type': comparison_type,
        'partial_mitigation_pct': partial_mit_pct,
        'partial_mitigation_value': float(partial_mit_value),
        'decision_times_optimal': decision_times_label,
        'decision_times_partial': partial_decision_times_label,
        'ra': float(ra),
        'eis': float(eis),
        'pref': float(pref),
        'tech_chg': float(tech_chg),
        'tech_scale': float(tech_scale),
        'bs_premium': float(bs_premium),
        'growth': float(growth),
        'baseline_num': int(baseline_num),
        'dam_func': int(dam_func),
        'tip_on': int(tip_on),
        'd_unc': int(d_unc),
        't_unc': int(t_unc),
        'no_free_lunch': bool(no_free_lunch),
        'u_optimal': float(u_optimal),
        'u_partial': float(u_partial_result),
        'utility_loss': float(ca.con_cost),
    }

    for i, year in enumerate(common_years):
        timeseries_dict[f'm_opt_{year}'] = float(m_opt_mapped[i]) if not np.isnan(m_opt_mapped[i]) else np.nan
        timeseries_dict[f'm_partial_{year}'] = float(m_partial_mapped[i]) if not np.isnan(m_partial_mapped[i]) else np.nan
        timeseries_dict[f'T_opt_{year}'] = float(T_opt_mapped[i]) if not np.isnan(T_opt_mapped[i]) else np.nan
        timeseries_dict[f'T_partial_{year}'] = float(T_partial_mapped[i]) if not np.isnan(T_partial_mapped[i]) else np.nan
        timeseries_dict[f'conc_opt_{year}'] = float(conc_opt_mapped[i]) if not np.isnan(conc_opt_mapped[i]) else np.nan
        timeseries_dict[f'conc_partial_{year}'] = float(conc_partial_mapped[i]) if not np.isnan(conc_partial_mapped[i]) else np.nan
        timeseries_dict[f'dam_opt_{year}'] = float(dam_opt_mapped[i]) if not np.isnan(dam_opt_mapped[i]) else np.nan
        timeseries_dict[f'dam_partial_{year}'] = float(dam_partial_mapped[i]) if not np.isnan(dam_partial_mapped[i]) else np.nan
        timeseries_dict[f'price_opt_{year}'] = float(price_opt_mapped[i]) if not np.isnan(price_opt_mapped[i]) else np.nan
        timeseries_dict[f'price_partial_{year}'] = float(price_partial_mapped[i]) if not np.isnan(price_partial_mapped[i]) else np.nan

    timeseries_csv_path = os.path.join(DATA_DIR, out_folder, 'analysis',
                                       f'{out_folder}_consolidated_timeseries.csv')
    print(f"Appending consolidated timeseries to: {timeseries_csv_path}")
    consolidated_outputs.append_results_to_csv(timeseries_dict, timeseries_csv_path)

    node_price_rows = []
    node_price_rows.extend(consolidated_outputs.build_node_price_rows(
        run_index, delay_year, task_id, 'optimal', t_baseline, co_opt,
        m_optimal, run_type, 'default', decision_times_label, params_for_nodes,
        comparison_type
    ))
    node_price_rows.extend(consolidated_outputs.build_node_price_rows(
        run_index, delay_year, task_id, 'partial', t_partial, co_partial,
        m_partial, run_type, 'default', partial_decision_times_label,
        params_for_nodes, comparison_type
    ))
    for row in node_price_rows:
        row['scenario_name'] = name
        row['partial_mitigation_pct'] = partial_mit_pct
        row['partial_mitigation_value'] = float(partial_mit_value)

    node_prices_csv_path = os.path.join(DATA_DIR, out_folder, 'analysis',
                                        f'{out_folder}_node_prices.csv')
    print(f"Appending consolidated node prices to: {node_prices_csv_path}")
    consolidated_outputs.append_rows_to_csv(node_price_rows, node_prices_csv_path)
    
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY (Partial Mitigation: {partial_mit_pct}%)")
    print(f"{'='*80}\n")
    
    print(f"Optimization Results:")
    print(f"  Optimal first-period mitigation:   {m_optimal[0]:.6f}")
    print(f"  Partial first-period mitigation:   {m_partial[0]:.6f}")
    print(f"  First-period mitigation cap:       {partial_mit_value:.6f} ({partial_mit_pct}%)")
    print(f"  Mitigation foregone:               {max(m_optimal[0] - m_partial[0], 0.0):.6f}")
    
    print(f"\nUtility Comparison:")
    print(f"  Optimal utility:                   {u_optimal:.10f}")
    print(f"  Partial mitigation utility:        {u_partial_result:.10f}")
    print(f"  Utility loss:                      {ca.con_cost:.10f}")
    print(f"  Relative loss:                     {(ca.con_cost/u_optimal)*100:.4f}%")
    
    print(f"\nEconomic Impacts:")
    print(f"  Consumption compensation (abs):    {ca.delta_c:.6f}")
    print(f"  Compensation (% of year 0 cons):   {ca.delta_c_pct:.4f}%")
    print(f"  Compensation (billions $):         ${ca.delta_c_billions:.2f}B")
    print(f"  Emission reduction foregone:       {ca.delta_emission_gton:.4f} Gt CO2")
    
    if ca.deadweight is not None:
        print(f"\nDeadweight Analysis:")
        print(f"  Deadweight cost:                   ${ca.deadweight:.2f} per ton CO2")
    
    print(f"\nOutput files:")
    print(f"  Optimal scenario:      {output_prefix_opt}_*.csv")
    print(f"  Partial scenario:      {output_prefix_partial}_*.csv")
    print(f"  Constraint analysis:   {analysis_prefix}_constraint_output.csv")
    print(f"  Optimal pickle:        {pickle_path_opt}")
    print(f"  Partial pickle:        {pickle_path_partial}")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


def main():
    """
    Main execution function for SGE array job.
    """
    run_index, task_id, delay_year, partial_mit_pct, partial_mit_value, out_folder = get_cluster_config()
    
    setup_cluster_directories(out_folder)

    header, indices, data = load_parameter_combination(None)
    
    if run_index >= len(data):
        print(f"ERROR: Run index {run_index} exceeds available combinations ({len(data)})")
        sys.exit(1)
    
    name = indices[run_index][1]
    
    print(f"\n{'='*80}")
    print(f"STARTING TASK")
    print(f"  Scenario:           {name} (run index {run_index})")
    print(f"  Delay period:       {delay_year} years")
    print(f"  Partial mitigation: {partial_mit_pct}%")
    print(f"  Task ID:            {task_id}")
    print(f"{'='*80}\n")
    
    print(f"Execution Configuration:")
    print(f"  Test mode:       {test_mode}")
    print(f"  Import damages:  {import_damages}\n")
    
    try:
        run_partial_mitigation_analysis(run_index, task_id, delay_year, partial_mit_pct, partial_mit_value,
                                       name, header, data, out_folder, test_mode, import_damages)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"TASK COMPLETE")
    print(f"  Scenario:           {name} (run {run_index})")
    print(f"  Delay period:       {delay_year} years")
    print(f"  Partial mitigation: {partial_mit_pct}%")
    print(f"  Task ID:            {task_id}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

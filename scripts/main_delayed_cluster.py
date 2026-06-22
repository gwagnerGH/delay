#!/usr/bin/env python

"""
Author: Theo Moers
Columbia University 2025
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
from src.config import DEFAULT_DECISION_TIMES
from src.tools import import_csv
from src.optimization import GeneticAlgorithm, GradientSearch

runs = [0] # list(range(0, 22))

delay_years = [5, 10, 15]
output_folder = "common_baseline"
test_mode = False
import_damages = False

SAME_GRID_BASELINE = "SAME_GRID"
COMMON_BASELINE = "COMMON_BASELINE"
ALIGNED_BASELINE = "ALIGNED_BASELINE"
baseline_mode = os.environ.get("BASELINE_MODE", SAME_GRID_BASELINE)


DATA_DIR = os.path.join(str(PROJECT_ROOT), "data", "new_outputs")


def damage_cache_tag(decision_times, prob_scale=1.0):
    dt_tag = '-'.join(str(int(x)) for x in decision_times)
    prob_tag = f"{float(prob_scale):g}".replace('.', 'p')
    return f"_DT{dt_tag}_PS{prob_tag}"


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
    total_combinations = num_runs * num_delays

    task_index = task_id - 1
    
    if task_index >= total_combinations:
        print(f"Error: Task ID {task_id} exceeds total combinations ({total_combinations})")
        print(f"runs = {runs} (length {num_runs})")
        print(f"delay_years = {delay_years} (length {num_delays})")
        print(f"Expected task range: 1-{total_combinations}")
        sys.exit(1)
    
    run_idx = task_index // num_delays
    delay_idx = task_index % num_delays
    run_index = runs[run_idx]
    delay_year = delay_years[delay_idx]
    out_folder = output_folder
    sge_task_first = os.environ.get('SGE_TASK_FIRST', 'Unknown')
    sge_task_last = os.environ.get('SGE_TASK_LAST', 'Unknown')
    job_id = os.environ.get('JOB_ID', 'Unknown')
    
    print(f"\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {total_combinations}")
    print(f"  Mapping: run={run_index} (runs[{run_idx}]), delay={delay_year} years (delay_years[{delay_idx}])")
    print(f"  Array range: {sge_task_first} to {sge_task_last}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")
    print(f"  Output folder: {out_folder}")
    print(f"\nConfiguration:")
    print(f"  All runs: {runs}")
    print(f"  All delay years: {delay_years}")
    
    return run_index, task_id, delay_year, out_folder


def setup_cluster_directories(out_folder):

    directories = [
        os.path.join(DATA_DIR, out_folder),
        os.path.join(DATA_DIR, out_folder, 'optimal'),
        os.path.join(DATA_DIR, out_folder, 'delayed'),
        os.path.join(DATA_DIR, out_folder, 'analysis'),
        os.path.join(DATA_DIR, out_folder, 'logs'),
        os.path.join(DATA_DIR, out_folder, 'aligned')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    for d in directories:
        print(f"  {d}")
    
    return directories


def load_parameter_combination(run_index):
    
    data_csv_for_import = "research_runs"
    
    print(f"\nLoading parameters from: data/{data_csv_for_import}.csv")
    
    header, indices, data = import_csv(data_csv_for_import, delimiter=",", indices=2)

    return header, indices, data


def run_delayed_analysis(run_index, delay_year, name, header, data, out_folder, test_mode, import_damages):

    print(f"Delay year: {delay_year}")
    
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
    
    
    decision_times_delay = DEFAULT_DECISION_TIMES.copy()
    if delay_year > 0:
        decision_times_delay[1] = delay_year

    if baseline_mode in (SAME_GRID_BASELINE, ALIGNED_BASELINE):
        # ALIGNED_BASELINE is accepted as a legacy alias, but the current main
        # comparison always solves baseline and delayed policy on the same grid.
        decision_times_baseline = decision_times_delay.copy()
    elif baseline_mode == COMMON_BASELINE:
        raise ValueError(
            "COMMON_BASELINE is a legacy mixed-grid comparison and is disabled "
            "for current same-grid runs."
        )
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    print(f"Baseline mode: {baseline_mode}")
    print(f"Decision times for baseline: {decision_times_baseline}")
    print(f"Decision times for delay_year={delay_year}: {decision_times_delay}")
    t_baseline = TreeModel(decision_times=decision_times_baseline,
                           prob_scale=1.0)
    t_delay = TreeModel(decision_times=decision_times_delay, prob_scale=1.0)
    
    # Emission baseline model
    baseline_emission_model_baseline = BPWEmissionBaseline(tree=t_baseline,
                                                  baseline_num=baseline_num)
    baseline_emission_model_baseline.baseline_emission_setup()
    
    baseline_emission_model_delay = BPWEmissionBaseline(tree=t_delay,
                                                  baseline_num=baseline_num)
    baseline_emission_model_delay.baseline_emission_setup()
    
    # Climate class
    draws = 3 * 10**6
    climate_baseline = BPWClimate(t_baseline, baseline_emission_model_baseline, draws=draws)
    climate_delay = BPWClimate(t_delay, baseline_emission_model_delay, draws=draws)
    
    # Cost class
    emit_at_0_baseline = np.interp(2030, baseline_emission_model_baseline.times,
                          baseline_emission_model_baseline.baseline_gtco2)
    c_baseline = BPWCost(t_baseline, emit_at_0=emit_at_0_baseline, baseline_num=baseline_num,
                tech_const=tech_chg, tech_scale=tech_scale,
                cons_at_0=86252.0, # 2025 estimated from https://data.worldbank.org/indicator/NE.CON.TOTL.CD
                backstop_premium=bs_premium,
                no_free_lunch=no_free_lunch)
    
    emit_at_0_delay = np.interp(2030, baseline_emission_model_delay.times,
                          baseline_emission_model_delay.baseline_gtco2)
    c_delay = BPWCost(t_delay, emit_at_0=emit_at_0_delay, baseline_num=baseline_num,
                tech_const=tech_chg, tech_scale=tech_scale,
                cons_at_0=86252.0, # 2025 estimated from https://data.worldbank.org/indicator/NE.CON.TOTL.CD
                backstop_premium=bs_premium,
                no_free_lunch=no_free_lunch)
    
    # Damage class
    d_m = 0.1
    mitigation_constants = np.arange(0, 1 + d_m, d_m)[::-1]
    df_baseline = BPWDamage(tree=t_baseline, emit_baseline=baseline_emission_model_baseline,
                   climate=climate_baseline, mitigation_constants=mitigation_constants,
                   draws=draws)
    
    df_delay = BPWDamage(tree=t_delay, emit_baseline=baseline_emission_model_delay,
                   climate=climate_delay, mitigation_constants=mitigation_constants,
                   draws=draws)
    
    # Run damage simulation or import damages
    baseline_damage_file_tag = damage_cache_tag(
        decision_times_baseline, t_baseline.prob_scale
    )
    delay_damage_file_tag = damage_cache_tag(
        decision_times_delay, t_delay.prob_scale
    )

    damsim_filename_baseline = ''.join(["simulated_damages_df", str(dam_func),
                               "_TP", str(tip_on), "_SSP", str(baseline_num),
                               "_BY", str(t_baseline.base_year),
                               "_dunc", str(d_unc), "_tunc", str(t_unc),
                               baseline_damage_file_tag])
    
    damsim_filename_delay = ''.join(["simulated_damages_df", str(dam_func),
                               "_TP", str(tip_on), "_SSP", str(baseline_num),
                               "_BY", str(t_delay.base_year),
                               "_dunc", str(d_unc), "_tunc", str(t_unc),
                               delay_damage_file_tag])
    
    print(f"Damage simulation: {damsim_filename_baseline}")
    
    if import_damages:
        try:
            df_baseline.import_damages(file_name=damsim_filename_baseline)
            df_delay.import_damages(file_name=damsim_filename_delay)
            print("Successfully imported damage simulation\n")
        except Exception as e:
            print(f"Warning: Could not import damages ({e})")
            print("Running damage simulation...")
            df_baseline.damage_simulation(filename=damsim_filename_baseline, save_simulation=True,
                                 dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                                 t_unc=t_unc)
            df_delay.damage_simulation(filename=damsim_filename_delay, save_simulation=True,
                                 dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                                 t_unc=t_unc)
    else:
        print("Running damage simulation...")
        df_baseline.damage_simulation(filename=damsim_filename_baseline, save_simulation=True,
                             dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                             t_unc=t_unc)
        df_delay.damage_simulation(filename=damsim_filename_delay, save_simulation=True,
                             dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                             t_unc=t_unc)
    
    # Create single utility instance for all scenarios
    u_baseline = EZUtility(tree=t_baseline, damage=df_baseline, cost=c_baseline, period_len=5.0, eis=eis, ra=ra,
                  time_pref=pref, cons_growth=growth)

    u_delay = EZUtility(tree=t_delay, damage=df_delay, cost=c_delay, period_len=5.0, eis=eis, ra=ra,
                  time_pref=pref, cons_growth=growth)

    print("Model components initialized\n")
    
    print("Running optimal scenario...\n")
    
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
    
    print(f"\nOptimal scenario complete:")
    print(f"  First-period mitigation:  {m_optimal[0]:.6f}")
    print(f"  Carbon price:             ${c_baseline.price(0, m_optimal[0], 0):.2f} per ton")
    print(f"  Utility:                  {u_optimal:.10f}\n")
    
    output_prefix_opt = f'{out_folder}/optimal/{name}_delay{delay_year}yr_optimal'
    
    co_opt = ClimateOutput(u_baseline)
    co_opt.calculate_output(m_optimal)
    co_opt.save_output(m_optimal, prefix=output_prefix_opt)
    
    # Save pickle
    p_opt = {
        'df.d_rcomb': df_baseline.d_rcomb,
        'm_opt': m_optimal,
        'u_opt': u_optimal,
        'delay_action': False,
        'delay_years': 0,
        'decision_times': t_baseline.decision_times,
        'parameters': dict(zip(header, model_params))
    }
    
    pickle_path_opt = os.path.join(DATA_DIR, out_folder, 'optimal',
                                    f'{name}_delay{delay_year}yr_optimal_log.pickle')
    with open(pickle_path_opt, 'wb') as handle:
        pickle.dump(p_opt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Optimal scenario saved: {pickle_path_opt}\n")

    print("Running optimized scenario with time-aligned decision times...\n")
    fixed_indices_align = None
    fixed_values_align = None

    ga_model_align = GeneticAlgorithm(
        pop_amount=400,
        num_generations=N_generations_ga,
        cx_prob=0.8, 
        mut_prob=0.50, 
        bound=1.5,
        num_feature=t_delay.num_decision_nodes,
        utility=u_delay, 
        fixed_values=fixed_values_align,
        fixed_indices=fixed_indices_align,
        print_progress=True
    )

    gs_model_align = GradientSearch(
        var_nums=t_delay.num_decision_nodes,  
        utility=u_delay, 
        accuracy=5.e-7,
        iterations=N_iters_gs,
        fixed_values=fixed_values_align,
        fixed_indices=fixed_indices_align,
        print_progress=True
    )

    print("Running Genetic Algorithm (time-aligned)...")
    final_pop_align, fitness_align = ga_model_align.run()
    sort_pop_align = final_pop_align[np.argsort(fitness_align)][::-1]

    print("Running Gradient Search (time-aligned)...")
    m_aligned, u_aligned = gs_model_align.run(initial_point_list=sort_pop_align, topk=1)

    print(f"\nTime-aligned scenario complete:")
    print(f"  First-period mitigation:  {m_aligned[0]:.6f}")
    print(f"  Carbon price:             ${c_baseline.price(0, m_aligned[0], 0):.2f} per ton")
    print(f"  Utility:                  {u_aligned:.10f}\n")

    output_prefix_align = f'{out_folder}/aligned/{name}_delay{delay_year}yr_time_aligned'
    co_align = ClimateOutput(u_delay)
    co_align.calculate_output(m_aligned)
    co_align.save_output(m_aligned, prefix=output_prefix_align)

    p_align = {
        'df.d_rcomb': df_delay.d_rcomb,
        'm_opt': m_aligned,
        'u_opt': u_aligned,
        'delay_action': False,
        'delay_years': 0,
        'decision_times': t_delay.decision_times,
        'parameters': dict(zip(header, model_params))
    }

    pickle_path_align = os.path.join(DATA_DIR, out_folder, 'aligned',
                                    f'{name}_delay{delay_year}yr_time_aligned_log.pickle')
    with open(pickle_path_align, 'wb') as handle:
        pickle.dump(p_align, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Time-aligned (unconstrained) scenario saved: {pickle_path_align}\n")

    print("\n\n\n Running delayed action scenario...")
    
    fixed_indices_delay = [0]
    fixed_values_delay = np.zeros(len(fixed_indices_delay))
    
    print(f"Constraint configuration:")
    print(f"  Total decision nodes:        {t_delay.num_decision_nodes}")
    print(f"  Number of nodes constrained: {len(fixed_indices_delay)}")
    print(f"  Constrained node indices:    {fixed_indices_delay}\n")
    
    ga_model_delay = GeneticAlgorithm(
        pop_amount=400,
        num_generations=N_generations_ga,
        cx_prob=0.8, 
        mut_prob=0.50, 
        bound=1.5,
        num_feature=t_delay.num_decision_nodes,
        utility=u_delay, 
        fixed_values=fixed_values_delay,
        fixed_indices=fixed_indices_delay,
        print_progress=True
    )
    
    gs_model_delay = GradientSearch(
        var_nums=t_delay.num_decision_nodes,
        utility=u_delay, 
        accuracy=5.e-7,
        iterations=N_iters_gs,
        fixed_values=fixed_values_delay,
        fixed_indices=fixed_indices_delay,
        print_progress=True
    )
    
    print("Running Genetic Algorithm (delayed)...")
    final_pop_delay, fitness_delay = ga_model_delay.run()
    sort_pop_delay = final_pop_delay[np.argsort(fitness_delay)][::-1]
    
    print("Running Gradient Search (delayed)...")
    m_delayed, u_delayed = gs_model_delay.run(initial_point_list=sort_pop_delay, topk=1)
    
    for idx in fixed_indices_delay:
        if abs(m_delayed[idx]) > 1e-10:
            print(f" Warning: Constrained node {idx} not zero: m_delayed[{idx}]={m_delayed[idx]:.10f}")
    
    print(f"\n Delayed scenario complete:")
    print(f"  First-period mitigation:  {m_delayed[0]:.6f} (constrained to 0)")
    print(f"  Carbon price:             ${c_delay.price(0, m_delayed[0], 0):.2f} per ton")
    print(f"  Utility:                  {u_delayed:.10f}\n")
    
    # Save delayed scenario output
    output_prefix_delay = f'{out_folder}/delayed/{name}_delay{delay_year}yr_delayed'

    co_delay = ClimateOutput(u_delay)
    co_delay.calculate_output(m_delayed)
    co_delay.save_output(m_delayed, prefix=output_prefix_delay)
    
    # Save pickle
    p_delay = {
        'df.d_rcomb': df_delay.d_rcomb,
        'm_opt': m_delayed,
        'u_opt': u_delayed,
        'delay_action': True,
        'delay_years': delay_year,
        'decision_times': t_delay.decision_times,
        'parameters': dict(zip(header, model_params))
    }
    
    pickle_path_delay = os.path.join(DATA_DIR, out_folder, 'delayed',
                                      f'{name}_delay{delay_year}yr_delayed_log.pickle')
    with open(pickle_path_delay, 'wb') as handle:
        pickle.dump(p_delay, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Delayed scenario saved: {pickle_path_delay}\n")

    
    print("CONSTRAINT ANALYSIS \n")

    print(f"Running constraint analysis ({baseline_mode})...\n")
    
    ca = ConstraintAnalysis(u_delay, u_baseline, m_delayed, m_optimal)
    analysis_prefix = f'{out_folder}/analysis/{name}_delay{delay_year}yr_analysis'
    ca.save_output(prefix=analysis_prefix)

    
    print(f"COMPARISON SUMMARY (DELAY={delay_year} YEARS)\n")
    print(f"\nOptimization Results:")
    print(f"  Optimal first-period mitigation:   {m_optimal[0]:.6f}")
    print(f"  Delayed first-period mitigation:   {m_delayed[0]:.6f}")
    print(f"  Mitigation foregone:               {m_optimal[0] - m_delayed[0]:.6f}")
    
    print(f"\nUtility Comparison:")
    print(f"  Optimal utility:                   {u_optimal:.10f}")
    print(f"  Delayed utility:                   {u_delayed:.10f}")
    print(f"  Utility loss:                      {ca.con_cost:.10f}")
    print(f"  Relative loss:                     {(ca.con_cost/u_optimal)*100:.4f}%")
    
    print(f"\nEconomic Impacts:")
    print(f"  Consumption compensation (abs):    {ca.delta_c:.6f}")
    print(f"  Compensation (% of year 0 cons):   {ca.delta_c_pct:.4f}%")
    print(f"  Compensation (billions $):         ${ca.delta_c_billions:.2f}B")
    print(f"  5-year DWL (% of year 0 cons):     {ca.delta_c_5yr_pct:.4f}%")
    print(f"  5-year DWL annual flow ($B):       ${ca.delta_c_5yr_billions:.2f}B")
    print(f"  Emission reduction foregone:       {ca.delta_emission_gton:.4f} Gt CO2")
    
    if ca.deadweight is not None:
        print(f"\nDeadweight Analysis:")
        print(f"  Deadweight cost:               ${ca.deadweight:.2f} per ton CO2")

    print(f"\n\nRunning time-aligned constraint analysis (aligned tree structure)...\n")

    ca_align = ConstraintAnalysis(u_delay, u_delay, m_delayed, m_aligned)
    analysis_prefix_align = f'{out_folder}/analysis/{name}_delay{delay_year}yr_analysis_time_aligned'
    ca_align.save_output(prefix=analysis_prefix_align)

    print(f"COMPARISON SUMMARY TIME-ALIGNED (DELAY={delay_year} YEARS)\n")
    print(f"\nOptimization Results:")
    print(f"  Time-aligned first-period mitigation: {m_aligned[0]:.6f}")
    print(f"  Delayed first-period mitigation:      {m_delayed[0]:.6f}")
    print(f"  Mitigation foregone:                  {m_aligned[0] - m_delayed[0]:.6f}")
    
    print(f"\nUtility Comparison:")
    print(f"  Time-aligned utility:              {u_aligned:.10f}")
    print(f"  Delayed utility:                   {u_delayed:.10f}")
    print(f"  Utility loss:                      {ca_align.con_cost:.10f}")
    print(f"  Relative loss:                     {(ca_align.con_cost/u_aligned)*100:.4f}%")

    print(f"\nEconomic Impacts:")
    print(f"  Consumption compensation (abs):    {ca_align.delta_c:.6f}")
    print(f"  Compensation (% of year 0 cons):   {ca_align.delta_c_pct:.4f}%")
    print(f"  Compensation (billions $):         ${ca_align.delta_c_billions:.2f}B")
    print(f"  5-year DWL (% of year 0 cons):     {ca_align.delta_c_5yr_pct:.4f}%")
    print(f"  5-year DWL annual flow ($B):       ${ca_align.delta_c_5yr_billions:.2f}B")
    print(f"  Emission reduction foregone:       {ca_align.delta_emission_gton:.4f} Gt CO2")



def main():
    run_index, task_id, delay_year, out_folder = get_cluster_config()
    
    setup_cluster_directories(out_folder)

    header, indices, data = load_parameter_combination(None)
    
    if run_index >= len(data):
        print(f"ERROR: Run index {run_index} exceeds available combinations ({len(data)})")
        sys.exit(1)
    
    name = indices[run_index][1]
    
    print(f"Running: {name} (run index {run_index})")
    print(f"DELAY YEAR: {delay_year}")
    print(f"TASK: {task_id}\n")
    
    print(f"\nExecution Configuration:")
    print(f"  Test mode:       {test_mode}")
    print(f"  Import damages:  {import_damages}")
    
    try:
        run_delayed_analysis(run_index, delay_year, name, header, data, 
                           out_folder, test_mode, import_damages)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"TASK COMPLETE: {name} (run {run_index}, delay {delay_year} years)")
    print(f"Task ID: {task_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

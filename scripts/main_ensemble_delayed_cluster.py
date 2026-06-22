#!/usr/bin/env python
"""
EZClimate Ensemble Delayed Action Cluster Script - SGE Array Job Version

This script combines Gaussian parameter exploration with delayed action
analysis on a cluster environment. It runs BOTH optimal and 
delayed action scenarios for each parameter sample, comparing them using the 
ConstraintAnalysis class.

The workflow:
1. Generate/load Gaussian parameter samples (RA, EIS, PRTP, tech_chg, tech_scale)
2. SGE_TASK_ID maps to unique (sample_index, delay_year) combinations
3. Each task runs optimal and delayed scenarios for one (sample, delay) pair
4. Use ConstraintAnalysis to calculate deadweight costs

Mapping:
    For N_SAMPLES samples and delay_years=[5,10,15], you get N_SAMPLES*3 combinations:
    Task 1  -> (sample=0, delay_year=5)
    Task 2  -> (sample=0, delay_year=10)
    Task 3  -> (sample=0, delay_year=15)
    Task 4  -> (sample=1, delay_year=5)
    Task 5  -> (sample=1, delay_year=10)
    ...

Usage:
    # Configure parameters below, then submit array job.
    # Samples will be generated automatically if they don't exist.
    # For N_SAMPLES=100 and delay_years=[5,10,15] -> 300 tasks:
    grid_run --grid_mem=100G --grid_submit=batch --grid_array=1-300 \\
             --grid_ncpus=4 bash scripts/run_ensemble_delayed_array_job.sh

Environment Variables Expected:
    SGE_TASK_ID: Integer from 1 to (N_SAMPLES * len(delay_years))
    OUTPUT_FOLDER: Name of output folder in data/ - optional override
    BASELINE_NUM: SSP baseline scenario (1-5) - optional override

Configuration:
    - Edit `N_SAMPLES` below to set number of Gaussian samples
    - Edit `delay_years` list below to set which delay years to test (e.g., [5, 10, 15])
    - Edit parameter ranges (ubs, lbs) to customize truncated Gaussian support
    - Edit `baseline_num` for SSP scenario selection

Author: Theo Moers
"""

import os
import sys
import pprint
import numpy as np
import csv
import fcntl
import time

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
from src.optimization import GeneticAlgorithm, GradientSearch
from src.gen_samples import generate_gaussian_samples
from src.config import (
    DEFAULT_BASE_YEAR,
    DEFAULT_CALENDAR_YEARS,
    DEFAULT_DECISION_TIMES,
    GAUSSIAN_PRIOR_SET_NAME,
    PARAMETER_PRIOR_DIMS,
    PARAMETER_PRIOR_LOWER_BOUNDS,
    PARAMETER_PRIOR_MEANS,
    PARAMETER_PRIOR_NAMES,
    PARAMETER_PRIOR_STDS,
    PARAMETER_PRIOR_UPPER_BOUNDS,
    RUN0_FIXED_PARAMETERS,
)


N_SAMPLES = 1000

delay_years = [5, 10, 15]

DIMS = PARAMETER_PRIOR_DIMS
ubs = PARAMETER_PRIOR_UPPER_BOUNDS
lbs = PARAMETER_PRIOR_LOWER_BOUNDS
param_names = PARAMETER_PRIOR_NAMES
# Risk Aversion, elasticity of intertemporal substitution, rate of exogeneous technological development, rate of endogeneous technological development, pure rate of time preference, backstop premium, consumption growth rate
# EIS and PRTP supports follow the Bauer & Rudebusch (2022) term-structure
# interpolation points archived in aux_notebooks/archive/Term-Structure-Interpolation.ipynb.

# Fixed parameters (not sampled): use research_runs.csv row 0 unless a
# robustness runner explicitly overrides one of these globals.
baseline_num = RUN0_FIXED_PARAMETERS["baseline_num"]
dam_func = RUN0_FIXED_PARAMETERS["dam_func"]
tip_on = RUN0_FIXED_PARAMETERS["tip_on"]
d_unc = RUN0_FIXED_PARAMETERS["d_unc"]
t_unc = RUN0_FIXED_PARAMETERS["t_unc"]
no_free_lunch = RUN0_FIXED_PARAMETERS["no_free_lunch"]

output_folder = "ensemble-cb-ir-l-bs-g"

test_mode = os.environ.get('TEST_MODE', '0').lower() in ('1', 'true', 'yes')
import_damages = os.environ.get('IMPORT_DAMAGES', '0').lower() in ('1', 'true', 'yes')

DATA_DIR = os.path.join(str(PROJECT_ROOT), "data", "new_outputs")

START_YEAR = DEFAULT_BASE_YEAR
COMMON_YEARS = sorted(set(DEFAULT_CALENDAR_YEARS + [START_YEAR + delay for delay in delay_years]))


def damage_cache_tag(decision_times, prob_scale=1.0, prefix=''):
    """Suffix damage files by tree structure so cluster tasks cannot collide."""

    dt_tag = '-'.join(str(int(x)) for x in decision_times)
    prob_tag = f"{float(prob_scale):g}".replace('.', 'p')
    return f"{prefix or ''}_DT{dt_tag}_PS{prob_tag}"


def calculate_period_climate_metrics(m, tree, damage, climate, emit_baseline):
    """
    Calculate temperature, concentration, and damage for each period.
    
    Parameters
    ----------
    m : ndarray
        Mitigation array
    tree : TreeModel
        Tree model
    damage : BPWDamage
        Damage model
    climate : BPWClimate
        Climate model
    emit_baseline : BPWEmissionBaseline
        Emission baseline model
    
    Returns
    -------
    tuple of ndarrays
        (exp_temp, exp_conc, exp_dam) - expected values per period
    """
    periods = tree.num_periods
    
    T_node = np.zeros(len(m))
    conc_node = np.zeros(len(m))
    dam_node = np.zeros(len(m))
    
    exp_temp = np.zeros(periods)
    exp_conc = np.zeros(periods)
    exp_dam = np.zeros(periods)
    
    for period in range(periods):
        nodes = tree.get_nodes_in_period(period)
        
        for node in range(nodes[0], nodes[1]+1):
            # Calculate damage
            dam_node[node] = damage._damage_function_node(m, node)
            
            # Calculate concentration
            conc_node[node] = climate.get_conc_at_node(m, node)
            
            # Calculate temperature
            mit_emit, _ = emit_baseline.get_mitigated_baseline(m, node=node, baseline='cumemit')
            T_node[node] = climate.TCRE_BEST_ESTIMATE * mit_emit[-1]
        
        # Take expectations over the period
        probs = tree.get_probs_in_period(period)
        exp_temp[period] = np.dot(T_node[nodes[0]:nodes[1]+1], probs)
        exp_conc[period] = np.dot(conc_node[nodes[0]:nodes[1]+1], probs)
        exp_dam[period] = np.dot(dam_node[nodes[0]:nodes[1]+1], probs)
    
    return exp_temp, exp_conc, exp_dam


def calculate_period_cumemit_metrics(m, tree, emit_baseline):
    """Calculate expected cumulative emissions for each decision period."""

    periods = tree.num_periods
    cumemit_node = np.zeros(len(m))
    exp_cumemit = np.zeros(periods)

    for period in range(periods):
        nodes = tree.get_nodes_in_period(period)

        for node in range(nodes[0], nodes[1]+1):
            mit_emit, _ = emit_baseline.get_mitigated_baseline(
                m, node=node, baseline='cumemit'
            )
            cumemit_node[node] = mit_emit[-1]

        probs = tree.get_probs_in_period(period)
        exp_cumemit[period] = np.dot(cumemit_node[nodes[0]:nodes[1]+1], probs)

    return exp_cumemit


def build_delay_frontier_metrics(delay_year, t_baseline, t_delay, co_opt, co_delay,
                                 m_delayed, exp_temp_delay, exp_conc_delay,
                                 exp_cumemit_delay, common_years,
                                 exp_cumemit_opt=None, welfare_loss=np.nan):
    """Build raw re-entry metrics for delay-frontier outputs."""

    reentry_year = int(t_delay.base_year + delay_year)

    price_opt_mapped = map_to_calendar_years(
        t_baseline, co_opt.expected_period_price, common_years
    )
    price_delay_mapped = map_to_calendar_years(
        t_delay, co_delay.expected_period_price, common_years
    )
    temp_delay_mapped = map_to_calendar_years(t_delay, exp_temp_delay, common_years)
    conc_delay_mapped = map_to_calendar_years(t_delay, exp_conc_delay, common_years)
    cumemit_delay_mapped = map_to_calendar_years(t_delay, exp_cumemit_delay, common_years)
    cumemit_opt_mapped = (
        map_to_calendar_years(t_baseline, exp_cumemit_opt, common_years)
        if exp_cumemit_opt is not None else None
    )
    m_delay_mapped = map_to_calendar_years(
        t_delay, co_delay.expected_period_mitigation, common_years
    )

    year_index = common_years.index(reentry_year) if reentry_year in common_years else None

    baseline_price = price_opt_mapped[year_index] if year_index is not None else np.nan
    delayed_price = price_delay_mapped[year_index] if year_index is not None else np.nan
    reentry_price_increase = delayed_price - baseline_price
    reentry_pct = (
        reentry_price_increase / baseline_price * 100.0
        if baseline_price and not np.isnan(baseline_price) else np.nan
    )
    annualized = (
        ((delayed_price / baseline_price) ** (1.0 / delay_year) - 1.0) * 100.0
        if delay_year > 0 and baseline_price and delayed_price
        and not np.isnan(baseline_price) and not np.isnan(delayed_price)
        else np.nan
    )

    post_delay_mask = np.asarray(common_years) >= reentry_year
    post_delay_m = m_delay_mapped[post_delay_mask]
    post_delay_m = post_delay_m[~np.isnan(post_delay_m)]
    peak_post_delay_mitigation = np.max(post_delay_m) if len(post_delay_m) else np.nan
    delayed_cumemit = (
        cumemit_delay_mapped[year_index]
        if year_index is not None and not np.isnan(cumemit_delay_mapped[year_index])
        else np.nan
    )
    baseline_cumemit = (
        cumemit_opt_mapped[year_index]
        if cumemit_opt_mapped is not None and year_index is not None
        and not np.isnan(cumemit_opt_mapped[year_index])
        else np.nan
    )
    extra_cumemit = (
        delayed_cumemit - baseline_cumemit
        if not np.isnan(delayed_cumemit) and not np.isnan(baseline_cumemit)
        else np.nan
    )

    return {
        'reentry_year': reentry_year,
        'baseline_reentry_price': float(baseline_price) if not np.isnan(baseline_price) else np.nan,
        'delayed_reentry_price': float(delayed_price) if not np.isnan(delayed_price) else np.nan,
        'reentry_price_increase': (
            float(reentry_price_increase) if not np.isnan(reentry_price_increase) else np.nan
        ),
        'reentry_price_pct_increase': float(reentry_pct) if not np.isnan(reentry_pct) else np.nan,
        'reentry_price_annualized_increase': float(annualized) if not np.isnan(annualized) else np.nan,
        'baseline_cumulative_emissions_at_reentry': (
            float(baseline_cumemit) if not np.isnan(baseline_cumemit) else np.nan
        ),
        'cumulative_emissions_at_reentry': float(delayed_cumemit) if not np.isnan(delayed_cumemit) else np.nan,
        'extra_cumulative_emissions': float(extra_cumemit) if not np.isnan(extra_cumemit) else np.nan,
        'ppm_at_reentry': (
            float(conc_delay_mapped[year_index])
            if year_index is not None and not np.isnan(conc_delay_mapped[year_index])
            else np.nan
        ),
        'temperature_at_reentry': (
            float(temp_delay_mapped[year_index])
            if year_index is not None and not np.isnan(temp_delay_mapped[year_index])
            else np.nan
        ),
        'peak_post_delay_mitigation': float(peak_post_delay_mitigation)
        if not np.isnan(peak_post_delay_mitigation) else np.nan,
        'peak_catchup_mitigation': float(peak_post_delay_mitigation)
        if not np.isnan(peak_post_delay_mitigation) else np.nan,
        'welfare_loss': float(welfare_loss) if not np.isnan(welfare_loss) else np.nan,
    }


def map_to_calendar_years(tree, period_values, target_years=COMMON_YEARS, start_year=None):
    """
    Maps period-indexed values to calendar years on a common grid.
    
    Returns NaN for years not in this tree's decision times, or interpolates
    for years between decision points.
    
    Parameters
    ----------
    tree : TreeModel
        The tree model with decision_times attribute
    period_values : array-like
        Values indexed by period (length = tree.num_periods)
    target_years : list of int
        Calendar years to map to (default: COMMON_YEARS)
    start_year : int
        Starting calendar year. Defaults to tree.base_year.
    
    Returns
    -------
    np.ndarray
        Values mapped to target_years, with NaN for missing data
    """
    if start_year is None:
        start_year = tree.base_year

    # Convert tree.decision_times to calendar years
    tree_years = [start_year + dt for dt in tree.decision_times]
    
    # period_values has length = num_periods, but tree_years has length = num_periods + 1
    # We need to ensure we don't index beyond period_values bounds
    num_periods = len(period_values)
    
    result = np.full(len(target_years), np.nan)
    
    for i, target_year in enumerate(target_years):
        if target_year in tree_years[:num_periods]:
            # Exact match - use the period value (only check first num_periods years)
            period_idx = tree_years.index(target_year)
            if period_idx < num_periods:
                result[i] = period_values[period_idx]
        elif target_year < tree_years[0] or target_year > tree_years[num_periods-1]:
            # Outside range of available data - keep as NaN
            continue
        else:
            # Between decision times - linear interpolation
            for j in range(num_periods - 1):
                if tree_years[j] < target_year < tree_years[j+1]:
                    weight = (target_year - tree_years[j]) / (tree_years[j+1] - tree_years[j])
                    result[i] = period_values[j] + weight * (period_values[j+1] - period_values[j])
                    break
    
    return result


def acquire_file_lock(file_obj, timeout=120.0, poll_interval=0.25):
    """Acquire an exclusive lock without allowing a task to wait forever."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_interval)


def append_results_to_csv(results_dict, csv_path, max_retries=10, retry_delay=1.0,
                          lock_timeout=120.0):
    # ConstraintAnalysis results are written in a joint CSV   
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    for attempt in range(max_retries):
        try:
            with open(csv_path, 'a', newline='') as f:
                if not acquire_file_lock(f, timeout=lock_timeout):
                    print(f"ERROR: Timed out waiting for CSV lock: {csv_path}")
                    return False
                
                try:
                    if os.path.getsize(csv_path) == 0:
                        writer = csv.DictWriter(f, fieldnames=results_dict.keys())
                        writer.writeheader()
                    else:
                        writer = csv.DictWriter(f, fieldnames=results_dict.keys())
                    
                    writer.writerow(results_dict)
                    f.flush()
                    
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
            
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"Warning: Failed to write to CSV (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"ERROR: Failed to write to CSV after {max_retries} attempts: {e}")
                return False
    
    return False


def format_optional(value, fmt, na_value="NA", prefix="", suffix=""):
    """Format optional numeric diagnostics without failing on missing values."""
    if value is None:
        return na_value
    try:
        if np.isnan(value):
            return na_value
    except TypeError:
        pass
    return f"{prefix}{format(value, fmt)}{suffix}"


def append_rows_to_csv(rows, csv_path, max_retries=10, retry_delay=1.0,
                       lock_timeout=120.0):
    """Append a list of dictionaries to a CSV using the same cluster-safe lock."""

    if not rows:
        return True

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(rows[0].keys())

    for attempt in range(max_retries):
        try:
            with open(csv_path, 'a', newline='') as f:
                if not acquire_file_lock(f, timeout=lock_timeout):
                    print(f"ERROR: Timed out waiting for CSV lock: {csv_path}")
                    return False

                try:
                    if os.path.getsize(csv_path) == 0:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                    else:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)

                    writer.writerows(rows)
                    f.flush()

                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True

        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"Warning: Failed to write rows to CSV (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"ERROR: Failed to write rows to CSV after {max_retries} attempts: {e}")
                return False

    return False


def build_node_price_rows(sample_id, delay_year, task_id, scenario, tree, climate_output,
                          mitigation, run_type, tree_spec, decision_times_label,
                          params, comparison_type, output_metadata=None):
    """Build long-form node price output for notebook posterior analysis."""

    metadata = dict(output_metadata or {})
    rows = []
    for node in range(tree.num_decision_nodes):
        period = tree.get_period(node)
        row = {
            'sample_index': sample_id,
            'delay_year': delay_year,
            'task_id': task_id,
            'run_type': run_type,
            'comparison_type': comparison_type,
            'tree_spec': tree_spec,
            'decision_times': decision_times_label,
            'scenario': scenario,
            'node': node,
            'period': period,
            'calendar_year': int(tree.base_year + tree.decision_times[period]),
            'node_probability': float(tree.node_prob[node]),
            'price': float(climate_output.prices[node]),
            'mitigation': float(mitigation[node]),
            'average_mitigation': float(climate_output.ave_mitigations[node]),
            'ghg_level': float(climate_output.ghg_levels[node]),
            'ra': float(params['ra']),
            'eis': float(params['eis']),
            'pref': float(params['pref']),
            'tech_chg': float(params['tech_chg']),
            'tech_scale': float(params['tech_scale']),
            'bs_premium': float(params['bs_premium']),
            'growth': float(params['growth']),
        }
        row.update(metadata)
        rows.append(row)
    return rows


class ZeroDamage(BPWDamage):
    """Damage object for Shapley coalitions with climate damages switched off."""

    def damage_simulation(self, filename="zero_damages", save_simulation=True,
                          dam_func=0, tip_on=True, d_unc=1, t_unc=1):
        self.d = np.zeros((self.dnum, self.tree.num_final_states, self.tree.num_periods))
        self.d_rcomb = self.d
        print("Zero climate damages active; skipped damage simulation.")

    def import_damages(self, file_name="zero_damages"):
        self.damage_simulation(filename=file_name, save_simulation=False)

    def damage_function(self, m, period, is_last=False):
        return np.zeros(self.tree.get_num_nodes_period(period))

    def _damage_function_node(self, m, node, is_last=False):
        return 0.0


def get_sample_filename():
    return os.path.join(DATA_DIR, f'Gaussian_samples_N{N_SAMPLES}_DIMS{DIMS}_{GAUSSIAN_PRIOR_SET_NAME}_ensemble_delayed.csv')


def generate_gaussian_ensemble_samples():
    samp_fname = get_sample_filename()
    means = PARAMETER_PRIOR_MEANS
    stds = PARAMETER_PRIOR_STDS
    
    print(f"\nGenerating {N_SAMPLES} bounded Gaussian samples...")
    print(f"Parameter space dimension: {DIMS}")
    print(f"Parameter support, Gaussian mode/main-spec value, and standard deviation:")
    for i, name in enumerate(param_names):
        print(f"  {name}: [{lbs[i]}, {ubs[i]}], mode={means[i]}, std={stds[i]}")
    
    generate_gaussian_samples(N_SAMPLES, DIMS, lbs, ubs, means=means, stds=stds,
                              save_file=True, filename=samp_fname)
    
    print(f"Samples saved to: {samp_fname}\n")


def load_or_generate_gaussian_samples():
    samp_fname = get_sample_filename()
    
    if not os.path.exists(samp_fname):
        print(f"\nSample file not found, generating new samples...")
        generate_gaussian_ensemble_samples()
    else:
        print(f"\nSample file found: {samp_fname}")
    
    param_vals = np.atleast_2d(np.loadtxt(samp_fname, delimiter=','))
    print(f"Loaded {len(param_vals)} parameter samples")
    
    return param_vals


def get_cluster_config():    
    sge_task_id = os.environ.get('SGE_TASK_ID') # Get SGE task ID (1-indexed)
    if sge_task_id is None:
        print("ERROR: SGE_TASK_ID environment variable not found!")
        print("This script is designed to run as part of an SGE array job.")
        sys.exit(1)
    
    try:
        task_id = int(sge_task_id)
    except ValueError:
        print(f"ERROR: Invalid SGE_TASK_ID: {sge_task_id}")
        sys.exit(1)
    
    num_delays = len(delay_years)
    total_combinations = N_SAMPLES * num_delays
    
    task_index = task_id - 1
    
    if task_index >= total_combinations:
        print(f"Error: Task ID {task_id} exceeds total combinations ({total_combinations})\n")
        print(f"N_SAMPLES = {N_SAMPLES}")
        print(f"delay_years = {delay_years} (length {num_delays})")
        print(f"Expected task range: 1-{total_combinations}")
        sys.exit(1)
    
    # We iterate through delays for each sample:
    # task_index = sample_idx * num_delays + delay_idx
    sample_index = task_index // num_delays
    delay_idx = task_index % num_delays
    delay_year = delay_years[delay_idx]

    out_folder = os.environ.get('OUTPUT_FOLDER', output_folder)
    baseline = int(os.environ.get('BASELINE_NUM', baseline_num))
    sge_task_first = os.environ.get('SGE_TASK_FIRST', 'Unknown')
    sge_task_last = os.environ.get('SGE_TASK_LAST', 'Unknown')
    job_id = os.environ.get('JOB_ID', 'Unknown')
    
    print(f"\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {total_combinations}")
    print(f"  Mapping: sample={sample_index}, delay_year={delay_year}")
    print(f"  Array range: {sge_task_first} to {sge_task_last}")
    print(f"  Hostname: {os.environ.get('HOSTNAME', 'Unknown')}")
    print(f"  Output folder: {out_folder}")
    print(f"  Baseline (SSP): {baseline}")
    print(f"\nConfiguration:")
    print(f"  Total samples: {N_SAMPLES}")
    print(f"  Delay years: {delay_years}")
    
    return sample_index, task_id, delay_year, out_folder, baseline


def setup_cluster_directories(out_folder):

    directories = [
        os.path.join(DATA_DIR, out_folder),
        os.path.join(DATA_DIR, out_folder, 'analysis'),
        os.path.join(DATA_DIR, out_folder, 'logs'),
        os.path.join(DATA_DIR, out_folder, 'samples')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("\nCreated directory structure:")
    for d in directories:
        print(f"  {d}")
    
    return directories


def run_ensemble_delayed_analysis(sample_index, delay_year, param_vals,
                                  out_folder, baseline, test_mode, import_damages,
                                  run_type='ensemble', tree_spec='default',
                                  comparison_type='same_grid',
                                  decision_times_baseline=None,
                                  decision_times_delay=None,
                                  prob_scale_baseline=1.0,
                                  prob_scale_delay=1.0,
                                  sample_label=None,
                                  common_years=None,
                                  delay_periods=1,
                                  period_len=5.0,
                                  emissions_time_step=None,
                                  damage_file_tag='',
                                  output_metadata=None,
                                  zero_climate_damages=False):
    
    ra, eis, tech_chg, tech_scale, pref, bs_premium, growth = param_vals[sample_index]
    sample_id = sample_label if sample_label is not None else sample_index
    task_id = os.environ.get('SGE_TASK_ID', 'unknown')
    
    name = f'sample{sample_index:04d}'
    
    print(f"\nSample {sample_id} | Delay Year: {delay_year} | Tree Spec: {tree_spec}\n")
    
    print('\n**Model Parameters:')
    model_params = {
        'ra': ra,
        'eis': eis,
        'pref': pref,
        'growth': growth,
        'tech_chg': tech_chg,
        'tech_scale': tech_scale,
        'dam_func': dam_func,
        'baseline_num': baseline,
        'tip_on': tip_on,
        'bs_premium': bs_premium,
        'd_unc': d_unc,
        't_unc': t_unc,
        'no_free_lunch': no_free_lunch,
        'period_len': period_len,
        'emissions_time_step': emissions_time_step,
        'damage_file_tag': damage_file_tag,
        'zero_climate_damages': zero_climate_damages,
    }
    pprint.pprint(model_params)
    
    if test_mode:
        print("\n***RUNNING IN TEST MODE***")
        N_generations_ga = 2
        N_iters_gs = 2
    else:
        print("\n***RUNNING IN FULL MODE***")
        N_generations_ga = 200
        N_iters_gs = 150
    
    print("\nInitializing model components...")
    
    if decision_times_delay is None:
        decision_times_delay = DEFAULT_DECISION_TIMES.copy()
        if delay_year > 0:
            decision_times_delay[1] = delay_year
    else:
        decision_times_delay = list(decision_times_delay)

    if decision_times_baseline is None:
        # Same-grid comparison: the unconstrained comparator and delayed run
        # use identical decision times, isolating the mitigation constraint.
        decision_times_baseline = decision_times_delay.copy()
    else:
        decision_times_baseline = list(decision_times_baseline)

    t_baseline = TreeModel(decision_times=decision_times_baseline,
                  prob_scale=prob_scale_baseline)

    t_delay = TreeModel(decision_times=decision_times_delay,
                  prob_scale=prob_scale_delay)

    if common_years is None:
        common_years = sorted(set(COMMON_YEARS
                                  + list(t_baseline.calendar_years)
                                  + list(t_delay.calendar_years)))

    decision_times_label = '|'.join(str(int(x)) for x in decision_times_baseline)
    delay_decision_times_label = '|'.join(str(int(x)) for x in decision_times_delay)
    baseline_damage_file_tag = damage_cache_tag(
        decision_times_baseline, prob_scale_baseline, damage_file_tag
    )
    delay_damage_file_tag = damage_cache_tag(
        decision_times_delay, prob_scale_delay, damage_file_tag
    )
    
    # Emission baseline model
    baseline_emission_model_baseline = BPWEmissionBaseline(tree=t_baseline,
                                                  baseline_num=baseline,
                                                  emissions_time_step=emissions_time_step)
    baseline_emission_model_baseline.baseline_emission_setup()

    baseline_emission_model_delay = BPWEmissionBaseline(tree=t_delay,
                                                  baseline_num=baseline,
                                                  emissions_time_step=emissions_time_step)
    baseline_emission_model_delay.baseline_emission_setup()
    
    # Climate class
    draws = 3 * 10**6
    climate_baseline = BPWClimate(
        t_baseline, baseline_emission_model_baseline, draws=draws
    )

    climate_delay = BPWClimate(
        t_delay, baseline_emission_model_delay, draws=draws
    )

    # Cost class
    emit_at_0_baseline = np.interp(2030, baseline_emission_model_baseline.times,
                          baseline_emission_model_baseline.baseline_gtco2)
    c_baseline = BPWCost(t_baseline, emit_at_0=emit_at_0_baseline,
                baseline_num=baseline, tech_const=tech_chg,
                tech_scale=tech_scale, cons_at_0=86252.0, # 2025 estimated from https://data.worldbank.org/indicator/NE.CON.TOTL.CD
                backstop_premium=bs_premium, no_free_lunch=no_free_lunch)
    
    emit_at_0_delay = np.interp(2030, baseline_emission_model_delay.times,
                          baseline_emission_model_delay.baseline_gtco2)
    c_delay = BPWCost(t_delay, emit_at_0=emit_at_0_delay,
                baseline_num=baseline, tech_const=tech_chg,
                tech_scale=tech_scale, cons_at_0=86252.0, # 2025 estimatedfrom https://data.worldbank.org/indicator/NE.CON.TOTL.CD
                backstop_premium=bs_premium, no_free_lunch=no_free_lunch)
    
    # Damage class
    d_m = 0.1
    mitigation_constants = np.arange(0, 1 + d_m, d_m)[::-1]
    damage_class = ZeroDamage if zero_climate_damages else BPWDamage
    df_baseline = damage_class(tree=t_baseline, emit_baseline=baseline_emission_model_baseline,
                   climate=climate_baseline, mitigation_constants=mitigation_constants,
                   draws=draws)

    df_delay = damage_class(tree=t_delay, emit_baseline=baseline_emission_model_delay,
                   climate=climate_delay, mitigation_constants=mitigation_constants,
                   draws=draws)


    damsim_filename_baseline = ''.join(["simulated_damages_df", str(dam_func),
                               "_TP", str(tip_on), "_SSP", str(baseline),
                               "_BY", str(t_baseline.base_year),
                               "_dunc", str(d_unc), "_tunc", str(t_unc),
                               baseline_damage_file_tag])
    
    damsim_filename_delay = ''.join(["simulated_damages_df", str(dam_func),
                               "_TP", str(tip_on), "_SSP", str(baseline),
                               "_BY", str(t_delay.base_year),
                               "_dunc", str(d_unc), "_tunc", str(t_unc),
                               delay_damage_file_tag])
    
    print(f"Damage simulation: {damsim_filename_baseline}")
    
    if zero_climate_damages:
        df_baseline.damage_simulation(filename=damsim_filename_baseline, save_simulation=False,
                             dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                             t_unc=t_unc)
        df_delay.damage_simulation(filename=damsim_filename_delay, save_simulation=False,
                             dam_func=dam_func, tip_on=tip_on, d_unc=d_unc,
                             t_unc=t_unc)
    elif import_damages:
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

    u_baseline = EZUtility(tree=t_baseline, damage=df_baseline, cost=c_baseline, period_len=period_len, eis=eis, ra=ra,
                  time_pref=pref, cons_growth=growth)
    
    u_delay = EZUtility(tree=t_delay, damage=df_delay, cost=c_delay, period_len=period_len, eis=eis, ra=ra,
                  time_pref=pref, cons_growth=growth)

    print("Model components initialized\n")
    
    
    print("\nRUNNING OPTIMAL (UNCONSTRAINED) SCENARIO\n")
    
    # no constraints for optimal scenario
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
    
    # Calculate climate output for timeseries (but don't save individual files)
    co_opt = ClimateOutput(u_baseline)
    co_opt.calculate_output(m_optimal)
    
    print(f"\nRUNNING DELAYED ACTION SCENARIO (DELAY={delay_year} YEARS)\n")
    
    if delay_periods > 0:
        fixed_indices_delay = get_delay_nodes(t_delay, delay_periods)
        fixed_values_delay = np.zeros(len(fixed_indices_delay))
    else:
        fixed_indices_delay = None
        fixed_values_delay = None
    
    print(f"Constraint configuration:")
    print(f"  Total decision nodes:        {t_delay.num_decision_nodes}")
    print(f"  Number of nodes constrained: {len(fixed_indices_delay) if fixed_indices_delay is not None else 0}")
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
    
    if fixed_indices_delay is not None:
        for idx in fixed_indices_delay:
            if abs(m_delayed[idx]) > 1e-10:
                print(f"Warning: Constrained node {idx} not zero: m_delayed[{idx}]={m_delayed[idx]:.10f}")
    
    print(f"\nDelayed scenario complete:")
    print(f"  First-period mitigation:  {m_delayed[0]:.6f} (constrained to 0)")
    print(f"  Carbon price:             ${c_delay.price(0, m_delayed[0], 0):.2f} per ton")
    print(f"  Utility:                  {u_delayed:.10f}\n")
    
    # Calculate climate output for timeseries (but don't save individual files)
    co_delay = ClimateOutput(u_delay)
    co_delay.calculate_output(m_delayed)
    
    print("\nCONSTRAINT ANALYSIS (COMPARING OPTIMAL VS DELAYED)\n")
    
    ca = ConstraintAnalysis(u_delay, u_baseline, m_delayed, m_optimal)

    exp_temp_opt, exp_conc_opt, exp_dam_opt = calculate_period_climate_metrics(
        m_optimal, t_baseline, df_baseline, climate_baseline, baseline_emission_model_baseline
    )

    exp_temp_delay, exp_conc_delay, exp_dam_delay = calculate_period_climate_metrics(
        m_delayed, t_delay, df_delay, climate_delay, baseline_emission_model_delay
    )

    exp_cumemit_delay = calculate_period_cumemit_metrics(
        m_delayed, t_delay, baseline_emission_model_delay
    )
    exp_cumemit_opt = calculate_period_cumemit_metrics(
        m_optimal, t_baseline, baseline_emission_model_baseline
    )

    frontier_metrics = build_delay_frontier_metrics(
        delay_year, t_baseline, t_delay, co_opt, co_delay, m_delayed,
        exp_temp_delay, exp_conc_delay, exp_cumemit_delay, common_years,
        exp_cumemit_opt=exp_cumemit_opt, welfare_loss=ca.con_cost
    )
    
    print(f"\nCOMPARISON SUMMARY (SAMPLE={sample_index}, DELAY={delay_year})\n")
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
    print(f"  Consumption compensation (abs):    {format_optional(ca.delta_c, '.6f')}")
    print(f"  Compensation (% of year 0 cons):   {format_optional(ca.delta_c_pct, '.4f', suffix='%')}")
    print(f"  Compensation (billions $):         {format_optional(ca.delta_c_billions, '.2f', prefix='$', suffix='B')}")
    print(f"  5-year DWL (% of year 0 cons):     {format_optional(ca.delta_c_5yr_pct, '.4f', suffix='%')}")
    print(f"  5-year DWL annual flow ($B):       {format_optional(ca.delta_c_5yr_billions, '.2f', prefix='$', suffix='B')}")
    print(f"  Emission reduction foregone:       {ca.delta_emission_gton:.4f} Gt CO2")
    
    if ca.deadweight is not None:
        print(f"\nDeadweight Analysis:")
        print(f"  Deadweight cost:                   ${ca.deadweight:.2f} per ton CO2")
    
    results_dict = {
        # Run identifiers
        'sample_index': sample_id,
        'delay_year': delay_year,
        'task_id': task_id,
        'run_type': run_type,
        'comparison_type': comparison_type,
        'tree_spec': tree_spec,
        'decision_times_optimal': decision_times_label,
        'decision_times_delayed': delay_decision_times_label,
        
        # Parameter values (from Gaussian sampling)
        'ra': float(ra),
        'eis': float(eis),
        'pref': float(pref),
        'tech_chg': float(tech_chg),
        'tech_scale': float(tech_scale),
        'bs_premium': float(bs_premium),
        'growth': float(growth),
        
        # Fixed parameters
        'baseline_num': int(baseline),
        'dam_func': int(dam_func),
        'tip_on': int(tip_on),
        'd_unc': int(d_unc),
        't_unc': int(t_unc),
        'no_free_lunch': bool(no_free_lunch),
        'period_len': float(period_len),
        'emissions_time_step': (
            float(emissions_time_step) if emissions_time_step is not None else np.nan
        ),
        'damage_file_tag': damage_file_tag,
        
        # Mitigation levels
        'm_optimal_period0': float(m_optimal[0]),
        'm_delayed_period0': float(m_delayed[0]),
        'mitigation_foregone': float(m_optimal[0] - m_delayed[0]),
        
        # Utility metrics
        'u_optimal': float(u_optimal),
        'u_delayed': float(u_delayed),
        'utility_loss': float(ca.con_cost),
        'utility_loss_pct': float((ca.con_cost/u_optimal)*100) if u_optimal != 0 else np.nan,
        
        # Economic impacts
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
        'year0_cons_delayed': float(ca.year0_cons_delayed),
        'delta_emission_gton': float(ca.delta_emission_gton),
        'deadweight_per_ton': float(ca.deadweight) if ca.deadweight is not None else np.nan,
        
        # Carbon prices
        'carbon_price_delayed': float(c_delay.price(0, m_delayed[0], 0)),
        'carbon_price_optimal': float(c_baseline.price(0, m_optimal[0], 0)),
    }
    results_dict.update(output_metadata or {})
    results_dict.update(frontier_metrics)
    
    csv_path = os.path.join(DATA_DIR, out_folder, 'analysis', f'{out_folder}_consolidated_results.csv')
    
    print(f"Appending results to: {csv_path}")
    success = append_results_to_csv(results_dict, csv_path)
    
    if success:
        print(f"Successfully appended results to consolidated CSV\n")
    else:
        print(f"Warning: Could not append to consolidated CSV (individual files still saved)\n")
    
    # Build timeseries data on common temporal grid
    print("\nCalculating climate metrics for timeseries...")
    
    print("Mapping timeseries data to common temporal grid...")
    print(f"  Common years: {common_years}")
    
    # Map optimal scenario to common grid
    m_opt_mapped = map_to_calendar_years(t_baseline, co_opt.expected_period_mitigation, common_years)
    T_opt_mapped = map_to_calendar_years(t_baseline, exp_temp_opt, common_years)
    conc_opt_mapped = map_to_calendar_years(t_baseline, exp_conc_opt, common_years)
    dam_opt_mapped = map_to_calendar_years(t_baseline, exp_dam_opt, common_years)
    price_opt_mapped = map_to_calendar_years(t_baseline, co_opt.expected_period_price, common_years)
    
    # Map delayed scenario to common grid
    m_delay_mapped = map_to_calendar_years(t_delay, co_delay.expected_period_mitigation, common_years)
    T_delay_mapped = map_to_calendar_years(t_delay, exp_temp_delay, common_years)
    conc_delay_mapped = map_to_calendar_years(t_delay, exp_conc_delay, common_years)
    dam_delay_mapped = map_to_calendar_years(t_delay, exp_dam_delay, common_years)
    price_delay_mapped = map_to_calendar_years(t_delay, co_delay.expected_period_price, common_years)
    
    # Build timeseries dictionary
    timeseries_dict = {
        # Run identifiers
        'sample_index': sample_id,
        'delay_year': delay_year,
        'task_id': task_id,
        'run_type': run_type,
        'comparison_type': comparison_type,
        'tree_spec': tree_spec,
        'decision_times_optimal': decision_times_label,
        'decision_times_delayed': delay_decision_times_label,
        
        # Parameter values (from Gaussian sampling)
        'ra': float(ra),
        'eis': float(eis),
        'pref': float(pref),
        'tech_chg': float(tech_chg),
        'tech_scale': float(tech_scale),
        'bs_premium': float(bs_premium),
        'growth': float(growth),
        
        # Fixed parameters
        'baseline_num': int(baseline),
        'dam_func': int(dam_func),
        'tip_on': int(tip_on),
        'd_unc': int(d_unc),
        't_unc': int(t_unc),
        'no_free_lunch': bool(no_free_lunch),
        'period_len': float(period_len),
        'emissions_time_step': (
            float(emissions_time_step) if emissions_time_step is not None else np.nan
        ),
        'damage_file_tag': damage_file_tag,
        
        # Summary metrics
        'u_optimal': float(u_optimal),
        'u_delayed': float(u_delayed),
        'utility_loss': float(ca.con_cost),
    }
    timeseries_dict.update(output_metadata or {})
    
    # Add timeseries organized by variable type (mitigation, temperature, concentration, damage, price)
    # For each variable, add optimal years first, then delayed years
    
    # Mitigation timeseries
    for i, year in enumerate(common_years):
        timeseries_dict[f'm_opt_{year}'] = float(m_opt_mapped[i]) if not np.isnan(m_opt_mapped[i]) else np.nan
    for i, year in enumerate(common_years):
        timeseries_dict[f'm_delay_{year}'] = float(m_delay_mapped[i]) if not np.isnan(m_delay_mapped[i]) else np.nan
    
    # Temperature timeseries
    for i, year in enumerate(common_years):
        timeseries_dict[f'T_opt_{year}'] = float(T_opt_mapped[i]) if not np.isnan(T_opt_mapped[i]) else np.nan
    for i, year in enumerate(common_years):
        timeseries_dict[f'T_delay_{year}'] = float(T_delay_mapped[i]) if not np.isnan(T_delay_mapped[i]) else np.nan
    
    # Concentration timeseries
    for i, year in enumerate(common_years):
        timeseries_dict[f'conc_opt_{year}'] = float(conc_opt_mapped[i]) if not np.isnan(conc_opt_mapped[i]) else np.nan
    for i, year in enumerate(common_years):
        timeseries_dict[f'conc_delay_{year}'] = float(conc_delay_mapped[i]) if not np.isnan(conc_delay_mapped[i]) else np.nan
    
    # Damage timeseries
    for i, year in enumerate(common_years):
        timeseries_dict[f'dam_opt_{year}'] = float(dam_opt_mapped[i]) if not np.isnan(dam_opt_mapped[i]) else np.nan
    for i, year in enumerate(common_years):
        timeseries_dict[f'dam_delay_{year}'] = float(dam_delay_mapped[i]) if not np.isnan(dam_delay_mapped[i]) else np.nan
    
    # Carbon price timeseries
    for i, year in enumerate(common_years):
        timeseries_dict[f'price_opt_{year}'] = float(price_opt_mapped[i]) if not np.isnan(price_opt_mapped[i]) else np.nan
    for i, year in enumerate(common_years):
        timeseries_dict[f'price_delay_{year}'] = float(price_delay_mapped[i]) if not np.isnan(price_delay_mapped[i]) else np.nan
    
    # Save timeseries to consolidated CSV
    timeseries_csv_path = os.path.join(DATA_DIR, out_folder, 'analysis', f'{out_folder}_consolidated_timeseries.csv')
    
    print(f"Appending timeseries to: {timeseries_csv_path}")
    success_ts = append_results_to_csv(timeseries_dict, timeseries_csv_path)
    
    if success_ts:
        print(f"Successfully appended timeseries to consolidated CSV\n")
    else:
        print(f"Warning: Could not append timeseries to consolidated CSV\n")

    node_price_rows = []
    node_price_rows.extend(build_node_price_rows(
        sample_id, delay_year, task_id, 'optimal', t_baseline, co_opt,
        m_optimal, run_type, tree_spec, decision_times_label, model_params,
        comparison_type, output_metadata=output_metadata
    ))
    node_price_rows.extend(build_node_price_rows(
        sample_id, delay_year, task_id, 'delayed', t_delay, co_delay,
        m_delayed, run_type, tree_spec, delay_decision_times_label, model_params,
        comparison_type, output_metadata=output_metadata
    ))

    node_prices_csv_path = os.path.join(DATA_DIR, out_folder, 'analysis',
                                        f'{out_folder}_node_prices.csv')
    print(f"Appending node prices to: {node_prices_csv_path}")
    success_nodes = append_rows_to_csv(node_price_rows, node_prices_csv_path)

    if success_nodes:
        print(f"Successfully appended node prices to consolidated CSV\n")
    else:
        print(f"Warning: Could not append node prices to consolidated CSV\n")


def main():    
    print("\nEZClimate ENSEMBLE DELAYED ACTION ANALYSIS - CLUSTER ARRAY JOB\n")

    sample_index, task_id, delay_year, out_folder, baseline = get_cluster_config()

    setup_cluster_directories(out_folder)
    
    param_vals = load_or_generate_gaussian_samples()
    
    if sample_index >= len(param_vals):
        print(f"ERROR: Sample index {sample_index} exceeds available samples ({len(param_vals)})")
        sys.exit(1)
    
    if task_id == 1:
        samples_copy = os.path.join(DATA_DIR, out_folder, 'samples', 
                                    f'Gaussian_samples_N{N_SAMPLES}_DIMS{DIMS}_{GAUSSIAN_PRIOR_SET_NAME}.csv')
        np.savetxt(samples_copy, param_vals, delimiter=',')
        print(f"\nSaved copy of samples to: {samples_copy}")
    

    print(f"RUNNING: Sample {sample_index}/{len(param_vals)-1}")
    print(f"DELAY YEAR: {delay_year}")
    print(f"TASK: {task_id}\n")
    
    print(f"\nExecution Configuration:")
    print(f"  Test mode:       {test_mode}")
    print(f"  Import damages:  {import_damages}")
    print(f"  Baseline (SSP):  {baseline}")
    
    try:
        run_ensemble_delayed_analysis(sample_index, delay_year, param_vals,
                                     out_folder, baseline, test_mode, import_damages)
    except Exception as e:
        print(f"ERROR running sample {sample_index} with delay {delay_year}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\nTASK COMPLETE: Sample {sample_index} (delay {delay_year})")
    print(f"Task ID: {task_id}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

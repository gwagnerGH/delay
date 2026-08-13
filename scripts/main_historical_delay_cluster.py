#!/usr/bin/env python
"""
Historical-delay deterministic runner.

This runner starts the model in 1975, uses observed historical total CO2
emissions through 2025 spliced to SSP2 afterwards, and compares an unconstrained
1975 optimum against a path constrained to zero mitigation before 2025.

Unlike the ensemble delayed runner, every task is a deterministic parameter
specification. Set HISTORICAL_PARAMETER_SOURCE=mean for run-0 parameters or
HISTORICAL_PARAMETER_SOURCE=robustness for deterministic robustness rows.
"""

import csv
import fcntl
import hashlib
import os
import pprint
import sys
import time

import numpy as np

try:
    from _project_paths import PROJECT_ROOT, configure_paths
except ModuleNotFoundError:
    from scripts._project_paths import PROJECT_ROOT, configure_paths
configure_paths()

from src.analysis.climate_output import ClimateOutput
from src.analysis.delayed_action import ConstraintAnalysis, get_delay_nodes
from src.climate import BPWClimate
from src.config import (
    GAUSSIAN_PRIOR_SET_NAME,
    PARAMETER_PRIOR_DIMS,
    PARAMETER_PRIOR_INDEX,
    PARAMETER_PRIOR_LOWER_BOUNDS,
    PARAMETER_PRIOR_UPPER_BOUNDS,
    RUN0_FIXED_PARAMETERS,
    RUN0_PARAMETER_VALUES,
)
from src.cost import BPWCost
from src.damage import BPWDamage
from src.emit_baseline import BPWEmissionBaseline
from src.optimization import (
    CandidateScreenedLBFGSB,
    GeneticAlgorithm,
    GradientSearch,
)
from src.adjoint_objective import EZAdjointObjective
from src.tree import TreeModel
from src.utility import EZUtility
from main_ensemble_delayed_cluster import ga_adjoint_warm_starts, solve_lbfgsb_policy


DATA_DIR = os.path.join(str(PROJECT_ROOT), "data", "new_outputs")

BASE_YEAR = 1975
HISTORICAL_DECISION_TIMES = [0, 15, 50, 55, 85, 125, 175, 225, 275]
HISTORICAL_CALENDAR_YEARS = [BASE_YEAR + year for year in HISTORICAL_DECISION_TIMES]
DELAY_PERIODS = 2
DELAY_YEAR = 50
PERIOD_LEN = 5.0
BASELINE_NUM = 2
BASELINE_SOURCE = "historical_splice"
EMISSIONS_TIME_STEP = 1
EMIT_AT_0_YEAR = 1975
CONS_AT_0 = 4544.81873304222  # billions current USD, World Bank NE.CON.TOTL.CD, WLD, 1975

DEFAULT_OUTPUT_FOLDER = "historical-delay-BY1975-SSP2-cons1975-v1"
DEFAULT_PARAMETER_SPECS = [
    "low_eis",
    "high_ra",
    "low_ra",
    "no_endogenous_learning",
]

test_mode = os.environ.get("TEST_MODE", "0").lower() in ("1", "true", "yes")
import_damages = os.environ.get("IMPORT_DAMAGES", "0").lower() in ("1", "true", "yes")
require_damage_import = os.environ.get("REQUIRE_DAMAGE_IMPORT", "0").lower() in ("1", "true", "yes")
if require_damage_import and not import_damages:
    raise ValueError("REQUIRE_DAMAGE_IMPORT=1 requires IMPORT_DAMAGES=1")

dam_func = RUN0_FIXED_PARAMETERS["dam_func"]
tip_on = RUN0_FIXED_PARAMETERS["tip_on"]
d_unc = RUN0_FIXED_PARAMETERS["d_unc"]
t_unc = RUN0_FIXED_PARAMETERS["t_unc"]
no_free_lunch = RUN0_FIXED_PARAMETERS["no_free_lunch"]


def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return parsed


def env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.lower() in ("1", "true", "yes", "y")


def default_lbfgsb_workers():
    for name in ("NSLOTS", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(name)
        if value:
            parsed = int(value)
            if parsed > 0:
                return parsed
    return 1


def optimizer_mode():
    return os.environ.get("OPTIMIZER", "ga_gs").strip().lower()


def require_lbfgsb_success(diagnostics, scenario_name):
    if diagnostics.get("lbfgsb_success", False):
        return
    fields = [
        "lbfgsb_message",
        "lbfgsb_best_result_accepted",
        "lbfgsb_scipy_success",
        "success_scipy",
        "success_diagnostics",
        "gradient_mode",
        "gradient_validation_status",
        "gradient_validation_message",
        "gradient_validation_max_abs_error",
        "gradient_validation_max_rel_error",
        "projected_grad_inf_norm",
        "projected_grad_l2_norm",
        "worst_kkt_node",
        "worst_kkt_time",
        "worst_kkt_state",
        "final_utility_spread",
        "final_utility_spread_rel",
        "effective_utility_spread_tol",
        "best_final_utility",
        "second_best_final_utility",
        "median_final_utility",
        "n_near_best_final_utility",
        "best_second_solution_linf",
        "projected_gradient_max_abs",
        "projected_gradient_tol",
        "projected_gradient_pass",
        "lbfgsb_converged",
        "stationarity_failed",
        "mitigation_kink_failed",
        "perturbation_failed",
        "max_perturbation_utility_gain",
        "effective_perturbation_tol",
        "best_perturbation_kind",
        "best_perturbation_direction",
        "best_perturbation_size",
        "all_at_mitigation_kink",
        "all_active_at_mitigation_kink",
        "all_active_free_at_mitigation_kink",
        "share_active_free_at_mitigation_kink",
        "selected_start_source_groups",
        "n_selected_start_source_groups",
        "welfare_decomposition_available",
        "consumption_tree_min",
        "consumption_tree_max",
        "cost_tree_min",
        "cost_tree_max",
        "n_candidates_evaluated",
        "n_local_starts",
        "escalation_used",
        "dispersion_failed",
    ]
    details = ", ".join(
        "{}={}".format(field, diagnostics.get(field))
        for field in fields
        if field in diagnostics
    )
    raise RuntimeError(
        "L-BFGS-B {} solve failed diagnostics; aborting because GA fallback is disabled. {}".format(
            scenario_name, details
        )
    )


def stable_seed(*parts):
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def historical_lbfgsb_policy(utility, num_nodes, scenario_name, warm_starts=None,
                             fixed_indices=None, fixed_values=None,
                             gradient_mode="finite_difference", seed_parts=()):
    # Adjoint production runs use the same removal-aware active-set solve and
    # fail-closed diagnostics as every other paper experiment.
    if gradient_mode == "adjoint":
        return solve_lbfgsb_policy(
            utility,
            num_nodes,
            scenario_name,
            warm_starts=warm_starts,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            seed_parts=seed_parts or (scenario_name,),
            print_progress=True,
            gradient_mode=gradient_mode,
        )

    lower = np.zeros(num_nodes, dtype=float)
    upper = np.full(num_nodes, 1.5, dtype=float)
    if fixed_indices is not None:
        fixed_indices = np.asarray(fixed_indices, dtype=int)
        fixed_values = np.asarray(fixed_values, dtype=float).flatten()
        lower[fixed_indices] = fixed_values
        upper[fixed_indices] = fixed_values
    seed = stable_seed(os.environ.get("RANDOM_SEED_BASE", "20250706"), scenario_name)
    objective_with_gradient = None
    optimizer_name = "lbfgsb_multistart"
    if gradient_mode == "adjoint":
        objective_with_gradient = EZAdjointObjective(utility)
        optimizer_name = "adjoint_lbfgsb"
    optimizer = CandidateScreenedLBFGSB(
        utility=utility,
        lower_bounds=lower,
        upper_bounds=upper,
        objective_with_gradient=objective_with_gradient,
        gradient_mode=gradient_mode,
        optimizer_name=optimizer_name,
        warm_starts=warm_starts or [],
        n_candidates=env_int("N_CANDIDATES", 256),
        n_local_starts=env_int("N_LOCAL_STARTS", 8),
        max_candidates=env_int("MAX_CANDIDATES", 1024),
        max_local_starts=env_int("MAX_LOCAL_STARTS", 16),
        maxiter=env_int("LBFGSB_MAXITER", 150),
        ftol=env_float("LBFGSB_FTOL", 1e-7),
        gtol=env_float("LBFGSB_GTOL", 1e-5),
        utility_spread_tol=env_float("UTILITY_SPREAD_TOL", 1e-7),
        utility_spread_rel_tol=env_float("UTILITY_SPREAD_REL_TOL", 1e-3),
        escalate_on_dispersion=env_bool("ESCALATE_ON_DISPERSION", True),
        start_design=os.environ.get("START_DESIGN", "sobol"),
        seed=seed,
        print_progress=True,
        scenario_name=scenario_name,
        candidate_progress_every=env_int("LBFGSB_CANDIDATE_PROGRESS_EVERY", 25),
        callback_progress_every=env_int("LBFGSB_CALLBACK_PROGRESS_EVERY", 10),
        n_workers=env_int("LBFGSB_N_WORKERS", default_lbfgsb_workers()),
        screening_workers=env_int(
            "LBFGSB_SCREENING_WORKERS",
            env_int("LBFGSB_N_WORKERS", default_lbfgsb_workers()),
        ),
        gradient_workers=env_int(
            "LBFGSB_GRADIENT_WORKERS",
            env_int("LBFGSB_N_WORKERS", default_lbfgsb_workers()),
        ),
        local_start_workers=env_int("LBFGSB_LOCAL_START_WORKERS", 1),
        finite_diff_step=env_float("LBFGSB_FINITE_DIFF_STEP", 1e-8),
        warm_start_perturbations=env_int("LBFGSB_WARM_START_PERTURBATIONS", 16),
        warm_start_perturbation_scale=env_float(
            "LBFGSB_WARM_START_PERTURBATION_SCALE", 0.05
        ),
        structured_start_count=env_int("LBFGSB_STRUCTURED_STARTS", 64),
        near_full_mitigation=env_float("LBFGSB_NEAR_FULL_MITIGATION", 0.98),
        start_boundary_epsilon=env_float("LBFGSB_START_BOUNDARY_EPSILON", 1e-6),
        preserve_diverse_starts=env_bool("LBFGSB_PRESERVE_DIVERSE_STARTS", True),
        min_diverse_start_groups=env_int("LBFGSB_MIN_DIVERSE_START_GROUPS", 4),
        perturbation_check=env_bool("LBFGSB_PERTURBATION_CHECK", True),
        perturbation_step=env_float("LBFGSB_PERTURBATION_STEP", 0.01),
        perturbation_tol=env_float("LBFGSB_PERTURBATION_TOL", 1e-7),
        perturbation_rel_tol=env_float("LBFGSB_PERTURBATION_REL_TOL", 1e-6),
        perturbation_block_count=env_int("LBFGSB_PERTURBATION_BLOCKS", 8),
        local_start_max_utility_gap=env_float("LBFGSB_LOCAL_START_MAX_UTILITY_GAP", np.inf),
        local_start_max_relative_utility_gap=env_float(
            "LBFGSB_LOCAL_START_MAX_RELATIVE_UTILITY_GAP", 0.25
        ),
        min_local_starts_after_filter=env_int("LBFGSB_MIN_LOCAL_STARTS_AFTER_FILTER", 2),
        gradient_progress_every=env_int("LBFGSB_GRADIENT_PROGRESS_EVERY", 1),
        kkt_check=env_bool("LBFGSB_KKT_CHECK", True),
        projected_gradient_tol=env_float("LBFGSB_PROJECTED_GRADIENT_TOL", 1e-5),
        validate_gradient=env_bool(
            "ADJOINT_VALIDATE_GRADIENT", gradient_mode == "adjoint"
        ),
        gradient_validation_directions=env_int("ADJOINT_VALIDATION_DIRECTIONS", 4),
        gradient_validation_abs_tol=env_float("ADJOINT_VALIDATION_ABS_TOL", 1e-5),
        gradient_validation_rel_tol=env_float("ADJOINT_VALIDATION_REL_TOL", 1e-3),
    )
    return optimizer.run()


def setup_cluster_directories(out_folder):
    for subdir in ("analysis", "samples"):
        os.makedirs(os.path.join(DATA_DIR, out_folder, subdir), exist_ok=True)


def damage_cache_tag(decision_times, prob_scale=1.0, prefix=""):
    dt_tag = "-".join(str(int(x)) for x in decision_times)
    prob_tag = f"{float(prob_scale):g}".replace(".", "p")
    return f"{prefix or ''}_DT{dt_tag}_PS{prob_tag}"


def parse_parameter_specs():
    env_value = os.environ.get("HISTORICAL_PARAMETER_SPECS")
    if not env_value:
        return list(DEFAULT_PARAMETER_SPECS)
    return [value.strip().lower() for value in env_value.split(",") if value.strip()]


def deterministic_parameter_rows():
    source = os.environ.get("HISTORICAL_PARAMETER_SOURCE", "mean").strip().lower()
    if source == "mean":
        return np.atleast_2d(RUN0_PARAMETER_VALUES), ["run0_params"], source

    if source not in ("robustness", "spec", "specs"):
        raise ValueError(
            "HISTORICAL_PARAMETER_SOURCE must be 'mean' or 'robustness' "
            f"(got {source!r})."
        )

    values = []
    labels = []
    for spec in parse_parameter_specs():
        row = RUN0_PARAMETER_VALUES.copy()
        if spec == "low_eis":
            row[PARAMETER_PRIOR_INDEX["EIS"]] = PARAMETER_PRIOR_LOWER_BOUNDS[PARAMETER_PRIOR_INDEX["EIS"]]
            label = spec
        elif spec == "high_eis":
            row[PARAMETER_PRIOR_INDEX["EIS"]] = PARAMETER_PRIOR_UPPER_BOUNDS[PARAMETER_PRIOR_INDEX["EIS"]]
            label = spec
        elif spec == "high_ra":
            row[PARAMETER_PRIOR_INDEX["RA"]] = PARAMETER_PRIOR_UPPER_BOUNDS[PARAMETER_PRIOR_INDEX["RA"]]
            label = spec
        elif spec == "low_ra":
            row[PARAMETER_PRIOR_INDEX["RA"]] = PARAMETER_PRIOR_LOWER_BOUNDS[PARAMETER_PRIOR_INDEX["RA"]]
            label = spec
        elif spec in ("no_endogenous_learning", "no_learning"):
            row[PARAMETER_PRIOR_INDEX["tech_scale"]] = 0.0
            label = "no_endogenous_learning"
        else:
            raise ValueError(
                f"Unknown HISTORICAL_PARAMETER_SPECS entry {spec!r}. "
                f"Known values: {DEFAULT_PARAMETER_SPECS}"
            )
        values.append(row)
        labels.append(label)

    return np.asarray(values, dtype=float), labels, source


def get_cluster_config(param_vals, labels, source):
    sge_task_id = os.environ.get("SGE_TASK_ID")
    if sge_task_id in (None, "", "undefined"):
        if len(param_vals) == 1:
            task_id = 1
            print("No valid SGE_TASK_ID found; treating single-row batch job as task 1.")
        else:
            print("ERROR: SGE_TASK_ID environment variable not found!")
            print("Use an SGE array job for multiple historical parameter rows.")
            sys.exit(1)
    else:
        try:
            task_id = int(sge_task_id)
        except ValueError:
            print(f"ERROR: Invalid SGE_TASK_ID: {sge_task_id}")
            sys.exit(1)

    task_index = task_id - 1
    if task_index >= len(param_vals):
        print(f"ERROR: Task ID {task_id} exceeds deterministic parameter count ({len(param_vals)})")
        print(f"HISTORICAL_PARAMETER_SOURCE = {source}")
        print(f"Parameter labels = {labels}")
        print(f"Expected task range: 1-{len(param_vals)}")
        sys.exit(1)

    out_folder = os.environ.get("OUTPUT_FOLDER", DEFAULT_OUTPUT_FOLDER)
    job_id = os.environ.get("JOB_ID", "Unknown")

    print("\nSGE Array Job Configuration:")
    print(f"  Job ID: {job_id}")
    print(f"  Task ID: {task_id} of {len(param_vals)}")
    print(f"  Parameter source: {source}")
    print(f"  Parameter label: {labels[task_index]}")
    print(f"  Output folder: {out_folder}")
    print(f"  Baseline: SSP{BASELINE_NUM} after 2025")
    print(f"  Historical years: {HISTORICAL_CALENDAR_YEARS}")

    return task_index, task_id, out_folder


def acquire_file_lock(file_obj, timeout=120.0, poll_interval=0.25):
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_interval)


def append_results_to_csv(results_dict, csv_path, max_retries=10, retry_delay=1.0):
    """Append result rows by field name even when diagnostics differ by branch."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    for attempt in range(max_retries):
        try:
            with open(csv_path, "a+", newline="") as f:
                if not acquire_file_lock(f):
                    print(f"ERROR: Timed out waiting for CSV lock: {csv_path}")
                    return False
                try:
                    if os.path.getsize(csv_path) == 0:
                        writer = csv.DictWriter(f, fieldnames=results_dict.keys())
                        writer.writeheader()
                    else:
                        f.seek(0)
                        reader = csv.DictReader(f)
                        fieldnames = list(reader.fieldnames or [])
                        missing_fields = [
                            key for key in results_dict if key not in fieldnames
                        ]
                        if missing_fields:
                            existing_rows = list(reader)
                            fieldnames.extend(missing_fields)
                            f.seek(0)
                            f.truncate()
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(existing_rows)
                        else:
                            f.seek(0, os.SEEK_END)
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({
                        key: results_dict.get(key, "") for key in writer.fieldnames
                    })
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"Warning: CSV write failed ({attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                print(f"ERROR: CSV write failed after {max_retries} attempts: {e}")
                return False
    return False

def append_rows_to_csv(rows, csv_path):
    """Append output rows by field name even when optional columns vary."""
    if not rows:
        return True
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    incoming_fieldnames = []
    for row in rows:
        incoming_fieldnames.extend(
            key for key in row if key not in incoming_fieldnames
        )
    with open(csv_path, "a+", newline="") as f:
        if not acquire_file_lock(f):
            print(f"ERROR: Timed out waiting for CSV lock: {csv_path}")
            return False
        try:
            if os.path.getsize(csv_path) == 0:
                writer = csv.DictWriter(f, fieldnames=incoming_fieldnames)
                writer.writeheader()
            else:
                f.seek(0)
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                missing_fields = [
                    key for key in incoming_fieldnames if key not in fieldnames
                ]
                if missing_fields:
                    existing_rows = list(reader)
                    fieldnames.extend(missing_fields)
                    f.seek(0)
                    f.truncate()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
                else:
                    f.seek(0, os.SEEK_END)
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows([
                {key: row.get(key, "") for key in writer.fieldnames}
                for row in rows
            ])
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return True

def map_to_calendar_years(tree, period_values, target_years):
    tree_years = [int(tree.base_year + dt) for dt in tree.decision_times]
    num_periods = len(period_values)
    result = np.full(len(target_years), np.nan)
    for i, target_year in enumerate(target_years):
        if target_year in tree_years[:num_periods]:
            period_idx = tree_years.index(target_year)
            result[i] = period_values[period_idx]
        elif target_year < tree_years[0] or target_year > tree_years[num_periods - 1]:
            continue
        else:
            for j in range(num_periods - 1):
                if tree_years[j] < target_year < tree_years[j + 1]:
                    weight = (target_year - tree_years[j]) / (tree_years[j + 1] - tree_years[j])
                    result[i] = period_values[j] + weight * (period_values[j + 1] - period_values[j])
                    break
    return result


def calculate_period_climate_metrics(m, tree, damage, climate, emit_baseline):
    periods = tree.num_periods
    exp_temp = np.zeros(periods)
    exp_conc = np.zeros(periods)
    exp_dam = np.zeros(periods)

    for period in range(periods):
        nodes = tree.get_nodes_in_period(period)
        probs = tree.get_probs_in_period(period)
        temp_node = []
        conc_node = []
        dam_node = []
        for node in range(nodes[0], nodes[1] + 1):
            dam_node.append(damage._damage_function_node(m, node))
            conc_node.append(climate.get_conc_at_node(m, node))
            mit_emit, _ = emit_baseline.get_mitigated_baseline(m, node=node, baseline="cumemit")
            temp_node.append(climate.TCRE_BEST_ESTIMATE * mit_emit[-1])
        exp_temp[period] = np.dot(temp_node, probs)
        exp_conc[period] = np.dot(conc_node, probs)
        exp_dam[period] = np.dot(dam_node, probs)

    return exp_temp, exp_conc, exp_dam


def calculate_period_cumemit_metrics(m, tree, emit_baseline):
    periods = tree.num_periods
    exp_cumemit = np.zeros(periods)
    for period in range(periods):
        nodes = tree.get_nodes_in_period(period)
        probs = tree.get_probs_in_period(period)
        values = []
        for node in range(nodes[0], nodes[1] + 1):
            mit_emit, _ = emit_baseline.get_mitigated_baseline(m, node=node, baseline="cumemit")
            values.append(mit_emit[-1])
        exp_cumemit[period] = np.dot(values, probs)
    return exp_cumemit


def build_node_price_rows(sample_id, task_id, scenario, tree, climate_output,
                          mitigation, params, metadata):
    rows = []
    for node in range(tree.num_decision_nodes):
        period = tree.get_period(node)
        row = {
            "sample_index": sample_id,
            "task_id": task_id,
            "run_type": "historical_delay",
            "comparison_type": "historical_observed_vs_1975_optimal",
            "tree_spec": "historical_1975",
            "decision_times": "|".join(str(int(x)) for x in tree.decision_times),
            "scenario": scenario,
            "node": node,
            "period": period,
            "calendar_year": int(tree.base_year + tree.decision_times[period]),
            "node_probability": float(tree.node_prob[node]),
            "price": float(climate_output.prices[node]),
            "mitigation": float(mitigation[node]),
            "average_mitigation": float(climate_output.ave_mitigations[node]),
            "ghg_level": float(climate_output.ghg_levels[node]),
            "ra": float(params["ra"]),
            "eis": float(params["eis"]),
            "pref": float(params["pref"]),
            "tech_chg": float(params["tech_chg"]),
            "tech_scale": float(params["tech_scale"]),
            "bs_premium": float(params["bs_premium"]),
            "growth": float(params["growth"]),
        }
        row.update(metadata)
        rows.append(row)
    return rows


def run_historical_delay(sample_index, task_id, param_vals, labels, source, out_folder):
    ra, eis, tech_chg, tech_scale, pref, bs_premium, growth = param_vals[sample_index]
    sample_label = labels[sample_index]

    if test_mode:
        n_generations_ga = 2
        n_iters_gs = 2
    else:
        n_generations_ga = env_int("N_GENERATIONS_GA", 200)
        n_iters_gs = env_int("N_ITERS_GS", 150)
    n_topk_gs = env_int("N_TOPK_GS", 8)
    print(
        "Optimization settings: "
        f"GA generations={n_generations_ga}, "
        f"gradient iterations={n_iters_gs}, "
        f"gradient topk={n_topk_gs}"
    )

    model_params = {
        "ra": ra,
        "eis": eis,
        "pref": pref,
        "growth": growth,
        "tech_chg": tech_chg,
        "tech_scale": tech_scale,
        "dam_func": dam_func,
        "baseline_num": BASELINE_NUM,
        "baseline_source": BASELINE_SOURCE,
        "tip_on": tip_on,
        "bs_premium": bs_premium,
        "d_unc": d_unc,
        "t_unc": t_unc,
        "no_free_lunch": no_free_lunch,
        "base_year": BASE_YEAR,
        "cons_at_0": CONS_AT_0,
        "emit_at_0_year": EMIT_AT_0_YEAR,
        "period_len": PERIOD_LEN,
    }
    print("\nModel parameters:")
    pprint.pprint(model_params)

    t_optimal = TreeModel(
        decision_times=HISTORICAL_DECISION_TIMES.copy(),
        prob_scale=1.0,
        base_year=BASE_YEAR,
    )
    t_delayed = TreeModel(
        decision_times=HISTORICAL_DECISION_TIMES.copy(),
        prob_scale=1.0,
        base_year=BASE_YEAR,
    )

    emit_optimal = BPWEmissionBaseline(
        tree=t_optimal,
        baseline_num=BASELINE_NUM,
        emissions_time_step=EMISSIONS_TIME_STEP,
        baseline_source=BASELINE_SOURCE,
    )
    emit_optimal.baseline_emission_setup()
    emit_delayed = BPWEmissionBaseline(
        tree=t_delayed,
        baseline_num=BASELINE_NUM,
        emissions_time_step=EMISSIONS_TIME_STEP,
        baseline_source=BASELINE_SOURCE,
    )
    emit_delayed.baseline_emission_setup()

    draws = int(os.environ.get(
        "HISTORICAL_DRAWS",
        "1000" if test_mode else str(3 * 10**6),
    ))
    climate_optimal = BPWClimate(t_optimal, emit_optimal, draws=draws)
    climate_delayed = BPWClimate(t_delayed, emit_delayed, draws=draws)

    emit_at_0_optimal = np.interp(EMIT_AT_0_YEAR, emit_optimal.times, emit_optimal.baseline_gtco2)
    emit_at_0_delayed = np.interp(EMIT_AT_0_YEAR, emit_delayed.times, emit_delayed.baseline_gtco2)
    cost_optimal = BPWCost(
        t_optimal,
        emit_at_0=emit_at_0_optimal,
        baseline_num=BASELINE_NUM,
        tech_const=tech_chg,
        tech_scale=tech_scale,
        cons_at_0=CONS_AT_0,
        backstop_premium=bs_premium,
        no_free_lunch=no_free_lunch,
    )
    cost_delayed = BPWCost(
        t_delayed,
        emit_at_0=emit_at_0_delayed,
        baseline_num=BASELINE_NUM,
        tech_const=tech_chg,
        tech_scale=tech_scale,
        cons_at_0=CONS_AT_0,
        backstop_premium=bs_premium,
        no_free_lunch=no_free_lunch,
    )

    mitigation_constants = np.arange(0, 1.1, 0.1)[::-1]
    damage_optimal = BPWDamage(
        tree=t_optimal,
        emit_baseline=emit_optimal,
        climate=climate_optimal,
        mitigation_constants=mitigation_constants,
        draws=draws,
    )
    damage_delayed = BPWDamage(
        tree=t_delayed,
        emit_baseline=emit_delayed,
        climate=climate_delayed,
        mitigation_constants=mitigation_constants,
        draws=draws,
    )

    damage_tag = damage_cache_tag(HISTORICAL_DECISION_TIMES, 1.0, "_HIST1975_GRID1")
    damsim_filename = "".join([
        "simulated_damages_df", str(dam_func),
        "_TP", str(tip_on), "_SSP", str(BASELINE_NUM),
        "_BY", str(BASE_YEAR),
        "_dunc", str(d_unc), "_tunc", str(t_unc),
        damage_tag,
    ])
    print(f"\nDamage simulation: {damsim_filename}")
    if import_damages:
        try:
            damage_optimal.import_damages(file_name=damsim_filename)
            damage_delayed.import_damages(file_name=damsim_filename)
        except Exception as exc:
            if require_damage_import:
                raise RuntimeError(
                    f"Required historical damage import failed for {damsim_filename!r}"
                ) from exc
            print(f"Warning: could not import damages ({exc}); running simulation.")
            damage_optimal.damage_simulation(
                filename=damsim_filename, save_simulation=True,
                dam_func=dam_func, tip_on=tip_on, d_unc=d_unc, t_unc=t_unc,
            )
            damage_delayed.import_damages(file_name=damsim_filename)
    else:
        damage_optimal.damage_simulation(
            filename=damsim_filename, save_simulation=True,
            dam_func=dam_func, tip_on=tip_on, d_unc=d_unc, t_unc=t_unc,
        )
        damage_delayed.import_damages(file_name=damsim_filename)

    utility_optimal = EZUtility(
        tree=t_optimal,
        damage=damage_optimal,
        cost=cost_optimal,
        period_len=PERIOD_LEN,
        eis=eis,
        ra=ra,
        time_pref=pref,
        cons_growth=growth,
    )
    utility_delayed = EZUtility(
        tree=t_delayed,
        damage=damage_delayed,
        cost=cost_delayed,
        period_len=PERIOD_LEN,
        eis=eis,
        ra=ra,
        time_pref=pref,
        cons_growth=growth,
    )

    print("\nRunning unconstrained 1975 optimum...")
    print("Shared optimal cache is disabled permanently; computing comparator locally.")
    historical_solver_diagnostics = {"optimizer": optimizer_mode()}
    if optimizer_mode() in ("lbfgsb_multistart", "adjoint_lbfgsb", "ga_adjoint_lbfgsb"):
        gradient_mode = (
            "adjoint" if optimizer_mode() in ("adjoint_lbfgsb", "ga_adjoint_lbfgsb")
            else "finite_difference"
        )
        optimal_warm_starts = []
        if optimizer_mode() == "ga_adjoint_lbfgsb":
            optimal_warm_starts, ga_diag_opt = ga_adjoint_warm_starts(
                utility_optimal, t_optimal.num_decision_nodes, "historical_optimal",
                (sample_label,),
            )
            historical_solver_diagnostics.update({f"optimal_{k}": v for k, v in ga_diag_opt.items()})
        m_optimal, u_optimal, diag_opt = historical_lbfgsb_policy(
            utility_optimal,
            t_optimal.num_decision_nodes,
            "historical_optimal",
            warm_starts=optimal_warm_starts,
            gradient_mode=gradient_mode,
        )
        historical_solver_diagnostics.update({f"optimal_{k}": v for k, v in diag_opt.items()})
        if not diag_opt.get("lbfgsb_success", False):
            require_lbfgsb_success(diag_opt, "historical optimal")
    else:
        ga_opt = GeneticAlgorithm(
            pop_amount=400,
            num_generations=n_generations_ga,
            cx_prob=0.8,
            mut_prob=0.50,
            bound=1.5,
            num_feature=t_optimal.num_decision_nodes,
            utility=utility_optimal,
            fixed_values=None,
            fixed_indices=None,
            print_progress=True,
        )
        gs_opt = GradientSearch(
            var_nums=t_optimal.num_decision_nodes,
            utility=utility_optimal,
            accuracy=5.e-7,
            iterations=n_iters_gs,
            fixed_values=None,
            fixed_indices=None,
            print_progress=True,
        )
        final_pop_opt, fitness_opt = ga_opt.run()
        sort_pop_opt = final_pop_opt[np.argsort(fitness_opt)][::-1]
        m_optimal, u_optimal = gs_opt.run(
            initial_point_list=sort_pop_opt,
            topk=min(n_topk_gs, len(sort_pop_opt)),
        )

    fixed_indices = get_delay_nodes(t_delayed, DELAY_PERIODS)
    fixed_values = np.zeros(len(fixed_indices))
    print("\nRunning observed-history delayed path...")
    print(f"  Constrained node indices: {fixed_indices}")
    if optimizer_mode() in ("lbfgsb_multistart", "adjoint_lbfgsb", "ga_adjoint_lbfgsb"):
        gradient_mode = (
            "adjoint" if optimizer_mode() in ("adjoint_lbfgsb", "ga_adjoint_lbfgsb")
            else "finite_difference"
        )
        warm_start_delay = np.asarray(m_optimal, dtype=float).copy()
        warm_start_delay[fixed_indices] = fixed_values
        delayed_warm_starts = [warm_start_delay]
        if optimizer_mode() == "ga_adjoint_lbfgsb":
            ga_starts, ga_diag_delay = ga_adjoint_warm_starts(
                utility_delayed, t_delayed.num_decision_nodes, "historical_delayed",
                (sample_label,), fixed_indices=fixed_indices, fixed_values=fixed_values,
            )
            delayed_warm_starts.extend(ga_starts)
            historical_solver_diagnostics.update({f"delayed_{k}": v for k, v in ga_diag_delay.items()})
        m_delayed, u_delayed, diag_delay = historical_lbfgsb_policy(
            utility_delayed,
            t_delayed.num_decision_nodes,
            "historical_delayed",
            warm_starts=delayed_warm_starts,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            gradient_mode=gradient_mode,
        )
        historical_solver_diagnostics.update({f"delayed_{k}": v for k, v in diag_delay.items()})
        if not diag_delay.get("lbfgsb_success", False):
            require_lbfgsb_success(diag_delay, "historical delayed")
    else:
        ga_delay = GeneticAlgorithm(
            pop_amount=400,
            num_generations=n_generations_ga,
            cx_prob=0.8,
            mut_prob=0.50,
            bound=1.5,
            num_feature=t_delayed.num_decision_nodes,
            utility=utility_delayed,
            fixed_values=fixed_values,
            fixed_indices=fixed_indices,
            print_progress=True,
        )
        gs_delay = GradientSearch(
            var_nums=t_delayed.num_decision_nodes,
            utility=utility_delayed,
            accuracy=5.e-7,
            iterations=n_iters_gs,
            fixed_values=fixed_values,
            fixed_indices=fixed_indices,
            print_progress=True,
        )
        final_pop_delay, fitness_delay = ga_delay.run()
        sort_pop_delay = final_pop_delay[np.argsort(fitness_delay)][::-1]
        warm_start_delay = np.asarray(m_optimal, dtype=float).copy()
        warm_start_delay[fixed_indices] = fixed_values
        gs_initial_delay = np.vstack([warm_start_delay, sort_pop_delay])
        m_delayed, u_delayed = gs_delay.run(
            initial_point_list=gs_initial_delay,
            topk=min(n_topk_gs, len(gs_initial_delay)),
        )

    for idx in fixed_indices:
        assert abs(m_delayed[idx]) < 1e-10, (
            f"Constrained node {idx} not zero: {m_delayed[idx]:.10f}"
        )

    co_opt = ClimateOutput(utility_optimal)
    co_opt.calculate_output(m_optimal)
    co_delay = ClimateOutput(utility_delayed)
    co_delay.calculate_output(m_delayed)

    ca = ConstraintAnalysis(utility_delayed, utility_optimal, m_delayed, m_optimal)

    exp_temp_opt, exp_conc_opt, exp_dam_opt = calculate_period_climate_metrics(
        m_optimal, t_optimal, damage_optimal, climate_optimal, emit_optimal
    )
    exp_temp_delay, exp_conc_delay, exp_dam_delay = calculate_period_climate_metrics(
        m_delayed, t_delayed, damage_delayed, climate_delayed, emit_delayed
    )
    exp_cumemit_opt = calculate_period_cumemit_metrics(m_optimal, t_optimal, emit_optimal)
    exp_cumemit_delay = calculate_period_cumemit_metrics(m_delayed, t_delayed, emit_delayed)

    common_years = HISTORICAL_CALENDAR_YEARS[:-1]
    decision_times_label = "|".join(str(int(x)) for x in HISTORICAL_DECISION_TIMES)
    metadata = {
        "parameter_source": source,
        "parameter_label": sample_label,
        "base_year": BASE_YEAR,
        "baseline_source": BASELINE_SOURCE,
        "historical_delay_periods": DELAY_PERIODS,
        "first_free_policy_year": 2025,
        "cons_at_0": CONS_AT_0,
        "emit_at_0_year": EMIT_AT_0_YEAR,
        "emit_at_0_optimal": float(emit_at_0_optimal),
        "emit_at_0_delayed": float(emit_at_0_delayed),
    }

    results = {
        "sample_index": sample_label,
        "task_id": task_id,
        "run_type": "historical_delay",
        "comparison_type": "historical_observed_vs_1975_optimal",
        "tree_spec": "historical_1975",
        "decision_times_optimal": decision_times_label,
        "decision_times_delayed": decision_times_label,
        "ra": float(ra),
        "eis": float(eis),
        "pref": float(pref),
        "tech_chg": float(tech_chg),
        "tech_scale": float(tech_scale),
        "bs_premium": float(bs_premium),
        "growth": float(growth),
        "baseline_num": BASELINE_NUM,
        "dam_func": int(dam_func),
        "tip_on": int(tip_on),
        "d_unc": int(d_unc),
        "t_unc": int(t_unc),
        "no_free_lunch": bool(no_free_lunch),
        "period_len": PERIOD_LEN,
        "emissions_time_step": EMISSIONS_TIME_STEP,
        "m_optimal_period0": float(m_optimal[0]),
        "m_delayed_period0": float(m_delayed[0]),
        "m_delayed_1990_low_state": float(m_delayed[1]),
        "m_delayed_1990_high_state": float(m_delayed[2]),
        "u_optimal": float(u_optimal),
        "u_delayed": float(u_delayed),
        "utility_loss": float(ca.con_cost),
        "utility_loss_pct": float((ca.con_cost / u_optimal) * 100) if u_optimal != 0 else np.nan,
        "delta_c": float(ca.delta_c) if ca.delta_c is not None else np.nan,
        "delta_c_pct": float(ca.delta_c_pct) if ca.delta_c_pct is not None else np.nan,
        "delta_c_billions": float(ca.delta_c_billions) if ca.delta_c_billions is not None else np.nan,
        "delta_c_5yr": float(ca.delta_c_5yr) if ca.delta_c_5yr is not None else np.nan,
        "delta_c_5yr_pct": float(ca.delta_c_5yr_pct) if ca.delta_c_5yr_pct is not None else np.nan,
        "delta_c_5yr_billions": float(ca.delta_c_5yr_billions) if ca.delta_c_5yr is not None else np.nan,
        "delta_c_5yr_total_billions": float(ca.delta_c_5yr_total_billions) if ca.delta_c_5yr is not None else np.nan,
        "year0_cons_delayed": float(ca.year0_cons_delayed),
        "delta_emission_gton": float(ca.delta_emission_gton),
        "deadweight_per_ton": float(ca.deadweight) if ca.deadweight is not None else np.nan,
        "carbon_price_delayed": float(cost_delayed.price(0, m_delayed[0], 0)),
        "carbon_price_optimal": float(cost_optimal.price(0, m_optimal[0], 0)),
    }
    results.update(metadata)
    results.update(historical_solver_diagnostics)

    analysis_dir = os.path.join(DATA_DIR, out_folder, "analysis")
    append_results_to_csv(
        results,
        os.path.join(analysis_dir, f"{out_folder}_consolidated_results.csv"),
    )

    mapped = {
        "m_opt": map_to_calendar_years(t_optimal, co_opt.expected_period_mitigation, common_years),
        "m_delay": map_to_calendar_years(t_delayed, co_delay.expected_period_mitigation, common_years),
        "T_opt": map_to_calendar_years(t_optimal, exp_temp_opt, common_years),
        "T_delay": map_to_calendar_years(t_delayed, exp_temp_delay, common_years),
        "conc_opt": map_to_calendar_years(t_optimal, exp_conc_opt, common_years),
        "conc_delay": map_to_calendar_years(t_delayed, exp_conc_delay, common_years),
        "dam_opt": map_to_calendar_years(t_optimal, exp_dam_opt, common_years),
        "dam_delay": map_to_calendar_years(t_delayed, exp_dam_delay, common_years),
        "price_opt": map_to_calendar_years(t_optimal, co_opt.expected_period_price, common_years),
        "price_delay": map_to_calendar_years(t_delayed, co_delay.expected_period_price, common_years),
        "cumemit_opt": map_to_calendar_years(t_optimal, exp_cumemit_opt, common_years),
        "cumemit_delay": map_to_calendar_years(t_delayed, exp_cumemit_delay, common_years),
    }
    timeseries = {
        "sample_index": sample_label,
        "task_id": task_id,
        "run_type": "historical_delay",
        "comparison_type": "historical_observed_vs_1975_optimal",
        "tree_spec": "historical_1975",
        "decision_times_optimal": decision_times_label,
        "decision_times_delayed": decision_times_label,
        "u_optimal": float(u_optimal),
        "u_delayed": float(u_delayed),
        "utility_loss": float(ca.con_cost),
    }
    timeseries.update(metadata)
    for variable, values in mapped.items():
        for i, year in enumerate(common_years):
            timeseries[f"{variable}_{year}"] = float(values[i]) if not np.isnan(values[i]) else np.nan

    append_results_to_csv(
        timeseries,
        os.path.join(analysis_dir, f"{out_folder}_consolidated_timeseries.csv"),
    )

    node_rows = []
    node_rows.extend(build_node_price_rows(sample_label, task_id, "optimal", t_optimal, co_opt, m_optimal, model_params, metadata))
    node_rows.extend(build_node_price_rows(sample_label, task_id, "delayed", t_delayed, co_delay, m_delayed, model_params, metadata))
    append_rows_to_csv(
        node_rows,
        os.path.join(analysis_dir, f"{out_folder}_node_prices.csv"),
    )

    print("\nHistorical-delay task complete.")
    print(f"  Optimal utility: {float(u_optimal):.10f}")
    print(f"  Delayed utility: {float(u_delayed):.10f}")
    print(f"  Utility loss:    {float(ca.con_cost):.10f}")


def save_parameter_metadata(param_vals, labels, source, out_folder):
    samples_path = os.path.join(
        DATA_DIR, out_folder, "samples",
        f"historical_delay_{source}_DIMS{PARAMETER_PRIOR_DIMS}_{GAUSSIAN_PRIOR_SET_NAME}.csv",
    )
    np.savetxt(samples_path, param_vals, delimiter=",")

    labels_path = os.path.join(
        DATA_DIR, out_folder, "samples",
        f"historical_delay_{source}_labels.csv",
    )
    with open(labels_path, "w") as f:
        f.write("sample_index,label\n")
        for index, label in enumerate(labels):
            f.write(f"{index},{label}\n")
    print(f"Saved deterministic parameter rows to: {samples_path}")
    print(f"Saved deterministic labels to: {labels_path}")


def main():
    print("\nEZClimate HISTORICAL DELAY ANALYSIS - DETERMINISTIC CLUSTER JOB\n")
    param_vals, labels, source = deterministic_parameter_rows()
    sample_index, task_id, out_folder = get_cluster_config(param_vals, labels, source)
    setup_cluster_directories(out_folder)

    if task_id == 1:
        save_parameter_metadata(param_vals, labels, source, out_folder)

    print("\nExecution Configuration:")
    print(f"  Test mode:       {test_mode}")
    print(f"  Import damages:  {import_damages}")
    print(f"  Parameter source:{source}")
    print(f"  Parameter label: {labels[sample_index]}")
    print(f"  Output folder:   {out_folder}")

    run_historical_delay(sample_index, task_id, param_vals, labels, source, out_folder)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in historical-delay run: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

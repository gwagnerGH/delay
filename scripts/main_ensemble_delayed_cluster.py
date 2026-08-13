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
    # For N_SAMPLES=300 and delay_years=[5,10,15] -> 900 tasks:
    grid_run --grid_mem=200G --grid_submit=batch --grid_array=1-900/50 \\
             --grid_ncpus=8 bash run_ensemble_delayed_array_job.sh

Environment Variables Expected:
    SGE_TASK_ID: Integer from 1 to (N_SAMPLES * len(delay_years))
    OUTPUT_FOLDER: Name of output folder in data/ - optional override
    BASELINE_NUM: SSP baseline scenario (1-5) - optional override
    BASELINE_ONLY: Set to 1 to export only the unconstrained benchmark baseline
    REQUIRE_DAMAGE_IMPORT: Set to 1 to forbid damage-import simulation fallback
    LBFGSB_POLICY_UPPER_BOUND: Experimental free-node L-BFGS-B cap (default 1.5)

Configuration:
    - Edit `N_SAMPLES` below to set number of Gaussian samples
    - Edit `delay_years` list below to set which delay years to test (default: [5, 10, 15])
    - Edit parameter ranges (ubs, lbs) to customize truncated Gaussian support
    - Edit `baseline_num` for SSP scenario selection

Author: Theo Moers
"""

import os
import sys
import pprint
import copy
import numpy as np
import csv
import fcntl
import time
import hashlib
import json
import subprocess

from _project_paths import PROJECT_ROOT, configure_paths
configure_paths()

from src.tree import TreeModel
from src.emit_baseline import BPWEmissionBaseline
from src.cost import BPWCost
from src.climate import BPWClimate
from src.damage import BPWDamage
from src.utility import EZUtility
from src.analysis.climate_output import ClimateOutput
from src.analysis.delayed_action import (
    FIXED_DELAY_DAMAGE_FILE_TAG,
    FIXED_DELAY_EMISSIONS_TIME_STEP,
    FIXED_DELAY_PERIOD_LEN,
    SUPPORTED_FIXED_DELAY_YEARS,
    ConstraintAnalysis,
    fixed_delay_decision_times,
    get_delay_nodes,
    get_delay_periods_for_year,
)
from src.optimization import (
    CandidateScreenedLBFGSB,
    GeneticAlgorithm,
    GradientSearch,
    prolong_policy_nearest_ancestor,
)
from src.adjoint_objective import EZAdjointObjective
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


N_SAMPLES = int(os.environ.get("N_SAMPLES", "10000"))
if N_SAMPLES <= 0:
    raise ValueError("N_SAMPLES must be positive.")

delay_years = [
    int(value.strip())
    for value in os.environ.get("DELAY_YEARS", "5,10,15").split(",")
    if value.strip()
]
if not delay_years or any(delay not in (5, 10, 15) for delay in delay_years):
    raise ValueError("DELAY_YEARS must be a nonempty subset of 5,10,15.")

DIMS = PARAMETER_PRIOR_DIMS
ubs = PARAMETER_PRIOR_UPPER_BOUNDS
lbs = PARAMETER_PRIOR_LOWER_BOUNDS
param_names = PARAMETER_PRIOR_NAMES
# Risk Aversion, elasticity of intertemporal substitution, rate of exogeneous technological development, rate of endogeneous technological development, pure rate of time preference, backstop premium, consumption growth rate
# EIS and PRTP supports follow the Bauer & Rudebusch (2022) term-structure
# interpolation points archived in aux_notebooks/archive/Term-Structure-Interpolation.ipynb.


def gaussian_eis_upper_bound():
    """Return EIS upper support configured for this ensemble."""

    eis_index = param_names.index("EIS")
    configured = os.environ.get("GAUSSIAN_EIS_UPPER_BOUND")
    upper = float(ubs[eis_index]) if configured in (None, "") else float(configured)
    lower = float(lbs[eis_index])
    native_upper = float(ubs[eis_index])
    if not lower < upper <= native_upper:
        raise ValueError(
            "GAUSSIAN_EIS_UPPER_BOUND must be greater than the configured "
            "lower support {} and no greater than {}, got {}".format(
                lower, native_upper, upper
            )
        )
    return upper


def gaussian_support_upper_bounds():
    """Return a private upper-bound array for the configured Gaussian support."""

    support_ubs = np.asarray(ubs, dtype=float).copy()
    support_ubs[param_names.index("EIS")] = gaussian_eis_upper_bound()
    return support_ubs


def gaussian_sample_seed():
    """Return the reproducible random seed for the shared Gaussian draw."""

    value = os.environ.get(
        "GAUSSIAN_SAMPLE_SEED", os.environ.get("RANDOM_SEED_BASE", "20250706")
    )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("GAUSSIAN_SAMPLE_SEED must be an integer, got {!r}".format(value)) from exc


def gaussian_preference_means():
    """Return Gaussian modes, optionally with a complete preference override."""
    names = ("GAUSSIAN_RA_MEAN", "GAUSSIAN_EIS_MEAN", "GAUSSIAN_PRTP_MEAN")
    supplied = [name for name in names if os.environ.get(name, "") != ""]
    if supplied and len(supplied) != len(names):
        missing = [name for name in names if name not in supplied]
        raise ValueError(
            "Specify GAUSSIAN_RA_MEAN, GAUSSIAN_EIS_MEAN, and "
            "GAUSSIAN_PRTP_MEAN together; missing {}".format(", ".join(missing))
        )
    means = np.asarray(PARAMETER_PRIOR_MEANS, dtype=float).copy()
    if supplied:
        means[param_names.index("RA")] = float(os.environ["GAUSSIAN_RA_MEAN"])
        means[param_names.index("EIS")] = float(os.environ["GAUSSIAN_EIS_MEAN"])
        means[param_names.index("PRTP")] = float(os.environ["GAUSSIAN_PRTP_MEAN"])
    support_ubs = gaussian_support_upper_bounds()
    if np.any(means < np.asarray(lbs, dtype=float)) or np.any(means > support_ubs):
        raise ValueError("Gaussian preference modes must lie within the configured support")
    return means


def gaussian_preference_tag():
    means = gaussian_preference_means()
    default = np.asarray(PARAMETER_PRIOR_MEANS, dtype=float)
    indices = [param_names.index(name) for name in ("RA", "EIS", "PRTP")]
    if np.allclose(means[indices], default[indices]):
        return ""
    return "_RA{:g}_EIS{:g}_PRTP{:g}".format(
        means[param_names.index("RA")], means[param_names.index("EIS")],
        means[param_names.index("PRTP")],
    ).replace("-", "m").replace(".", "p")


def gaussian_support_tag():
    """Stable filename tag that prevents reuse across EIS supports."""

    return ("EISmax{:g}".format(gaussian_eis_upper_bound()).replace("-", "m").replace(".", "p") + gaussian_preference_tag())


def gaussian_support_metadata():
    """Persist the actual ensemble support with every result row."""

    eis_index = param_names.index("EIS")
    eis_upper = gaussian_eis_upper_bound()
    return {
        "gaussian_prior_set": GAUSSIAN_PRIOR_SET_NAME,
        "gaussian_eis_lower_bound": float(lbs[eis_index]),
        "gaussian_eis_upper_bound": float(eis_upper),
        "gaussian_eis_truncated": bool(eis_upper < float(ubs[eis_index])),
        "gaussian_sample_seed": gaussian_sample_seed(),
        "gaussian_ra_mean": float(gaussian_preference_means()[param_names.index("RA")]),
        "gaussian_eis_mean": float(gaussian_preference_means()[param_names.index("EIS")]),
        "gaussian_prtp_mean": float(gaussian_preference_means()[param_names.index("PRTP")]),
    }

# Fixed parameters (not sampled): use research_runs.csv row 0 unless a
# robustness runner explicitly overrides one of these globals.
baseline_num = RUN0_FIXED_PARAMETERS["baseline_num"]
dam_func = RUN0_FIXED_PARAMETERS["dam_func"]
tip_on = RUN0_FIXED_PARAMETERS["tip_on"]
d_unc = RUN0_FIXED_PARAMETERS["d_unc"]
t_unc = RUN0_FIXED_PARAMETERS["t_unc"]
no_free_lunch = RUN0_FIXED_PARAMETERS["no_free_lunch"]

output_folder = "ensemble-BY2025-fixedlearn-run0gauss-N10000-delay5-15-v1"

test_mode = os.environ.get('TEST_MODE', '0').lower() in ('1', 'true', 'yes')
import_damages = os.environ.get('IMPORT_DAMAGES', '0').lower() in ('1', 'true', 'yes')

DATA_DIR = os.path.join(str(PROJECT_ROOT), "data", "new_outputs")

START_YEAR = DEFAULT_BASE_YEAR
COMMON_YEARS = sorted(set(
    DEFAULT_CALENDAR_YEARS
    + [START_YEAR + delay for delay in delay_years]
    + [START_YEAR + dt for dt in fixed_delay_decision_times()]
))


def env_int(name, default):
    """Read a positive integer environment override."""

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


no_free_lunch = env_bool("NO_FREE_LUNCH", no_free_lunch)


def require_damage_import_enabled():
    """Return whether damage artifacts are mandatory for this run."""

    return env_bool("REQUIRE_DAMAGE_IMPORT", False)


def validate_damage_import_configuration(import_enabled):
    """Reject benchmark configurations that could regenerate damage inputs."""

    if require_damage_import_enabled() and not import_enabled:
        raise ValueError(
            "REQUIRE_DAMAGE_IMPORT=1 requires IMPORT_DAMAGES=1; refusing to "
            "simulate a replacement damage artifact."
        )


def raise_required_damage_import_failure(exc, artifact_description):
    """Fail hard on an import error when reproducible artifacts are required."""

    if require_damage_import_enabled():
        raise RuntimeError(
            "REQUIRE_DAMAGE_IMPORT=1: failed to import {}; refusing to "
            "fall back to damage simulation.".format(artifact_description)
        ) from exc


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


def lbfgsb_policy_upper_bound():
    """Return the experimental uniform cap for free L-BFGS-B policy nodes."""

    upper_bound = env_float("LBFGSB_POLICY_UPPER_BOUND", 1.5)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError(
            "LBFGSB_POLICY_UPPER_BOUND must be finite and positive, got {!r}".format(
                upper_bound
            )
        )
    return float(upper_bound)


def backstop_smoothing_width():
    """Return the removal-premium smoothing width in mitigation units."""

    width = env_float("BACKSTOP_SMOOTHING_WIDTH", 0.0)
    if not np.isfinite(width) or width < 0.0:
        raise ValueError(
            "BACKSTOP_SMOOTHING_WIDTH must be finite and nonnegative, got {!r}".format(
                width
            )
        )
    return float(width)


def backstop_smoothing_mode():
    """Return the shape of the configured differentiable premium transition."""

    mode = os.environ.get("BACKSTOP_SMOOTHING_MODE", "one_sided_huber").strip().lower()
    if mode not in ("one_sided_huber", "symmetric_huber"):
        raise ValueError(
            "BACKSTOP_SMOOTHING_MODE must be one_sided_huber or symmetric_huber, "
            "got {!r}".format(mode)
        )
    return mode


def cost_formulation_name():
    """Return the provenance label for the configured removal-cost curve."""

    if backstop_smoothing_width() <= 0.0:
        return "additive_removal_premium_v1"
    if backstop_smoothing_mode() == "symmetric_huber":
        return "additive_removal_premium_symmetric_huber_v1"
    return "additive_removal_premium_huber_v1"


def removal_active_set_required(gradient_mode):
    """Whether the sharp removal kink requires an active-set solve."""

    requested = env_bool("LBFGSB_REMOVAL_ACTIVE_SET", gradient_mode == "adjoint")
    return bool(requested and backstop_smoothing_width() == 0.0)


def prefixed_diagnostics(prefix, diagnostics):
    return {
        f"{prefix}_{key}": value
        for key, value in diagnostics.items()
        if not str(key).startswith("_")
    }


def adjoint_local_optimizer_mode():
    return optimizer_mode() in (
        "adjoint_lbfgsb",
        "ga_adjoint_lbfgsb",
        "coarse_to_fine_adjoint_lbfgsb",
    )


def lbfgsb_optimizer_mode():
    return optimizer_mode() in (
        "lbfgsb_multistart",
        "adjoint_lbfgsb",
        "ga_adjoint_lbfgsb",
        "coarse_to_fine_adjoint_lbfgsb",
    )


def configured_optimizer_diagnostics(n_generations_ga, n_iters_gs, n_topk_gs):
    """Capture the configured search budget alongside realized diagnostics."""

    mode = optimizer_mode()
    configured = {
        "configured_optimizer": mode,
        "configured_require_damage_import": require_damage_import_enabled(),
        "configured_backstop_smoothing_width": backstop_smoothing_width(),
        "configured_backstop_smoothing_mode": backstop_smoothing_mode(),
        "configured_cost_formulation": cost_formulation_name(),
        "configured_random_seed_base": os.environ.get(
            "RANDOM_SEED_BASE", "20250706"
        ),
        "configured_n_generations_ga": int(n_generations_ga),
        "configured_n_iters_gs": int(n_iters_gs),
        "configured_n_topk_gs": int(n_topk_gs),
    }
    if mode == "ga_gs":
        configured["configured_ga_population"] = 400

    if not lbfgsb_optimizer_mode():
        return configured

    n_workers = env_int("LBFGSB_N_WORKERS", default_lbfgsb_workers())
    configured.update({
        "configured_n_candidates": env_int("N_CANDIDATES", 256),
        "configured_n_local_starts": env_int("N_LOCAL_STARTS", 8),
        "configured_max_candidates": env_int("MAX_CANDIDATES", 1024),
        "configured_max_local_starts": env_int("MAX_LOCAL_STARTS", 16),
        "configured_lbfgsb_maxiter": env_int("LBFGSB_MAXITER", 150),
        "configured_lbfgsb_ftol": env_float("LBFGSB_FTOL", 1e-7),
        "configured_lbfgsb_gtol": env_float("LBFGSB_GTOL", 1e-5),
        "configured_lbfgsb_n_workers": int(n_workers),
        "configured_lbfgsb_screening_workers": env_int(
            "LBFGSB_SCREENING_WORKERS", n_workers
        ),
        "configured_lbfgsb_gradient_workers": env_int(
            "LBFGSB_GRADIENT_WORKERS", n_workers
        ),
        "configured_lbfgsb_local_start_workers": env_int(
            "LBFGSB_LOCAL_START_WORKERS", 1
        ),
        "configured_escalate_on_dispersion": env_bool(
            "ESCALATE_ON_DISPERSION", True
        ),
        "configured_lbfgsb_kkt_check": env_bool("LBFGSB_KKT_CHECK", True),
        "configured_lbfgsb_projected_gradient_tol": env_float(
            "LBFGSB_PROJECTED_GRADIENT_TOL", 1e-5
        ),
        "configured_lbfgsb_nonsmooth_kkt_check": env_bool(
            "LBFGSB_NONSMOOTH_KKT_CHECK", True
        ),
        "configured_lbfgsb_nonsmooth_kkt_step": env_float(
            "LBFGSB_NONSMOOTH_KKT_STEP", 1e-4
        ),
        "configured_lbfgsb_nonsmooth_kkt_utility_gain_tol": env_float(
            "LBFGSB_NONSMOOTH_KKT_UTILITY_GAIN_TOL", 1e-8
        ),
        "configured_lbfgsb_nonsmooth_kkt_max_coordinates": env_int(
            "LBFGSB_NONSMOOTH_KKT_MAX_COORDINATES", 32
        ),
        "configured_lbfgsb_perturbation_check": env_bool(
            "LBFGSB_PERTURBATION_CHECK", True
        ),
        "configured_adjoint_validate_gradient": env_bool(
            "ADJOINT_VALIDATE_GRADIENT", adjoint_local_optimizer_mode()
        ),
        "configured_adjoint_validation_directions": env_int(
            "ADJOINT_VALIDATION_DIRECTIONS", 4
        ),
        "configured_adjoint_validation_abs_tol": env_float(
            "ADJOINT_VALIDATION_ABS_TOL", 1e-5
        ),
        "configured_adjoint_validation_rel_tol": env_float(
            "ADJOINT_VALIDATION_REL_TOL", 1e-3
        ),
        "configured_lbfgsb_abort_on_diagnostic_failure": env_bool(
            "LBFGSB_ABORT_ON_DIAGNOSTIC_FAILURE", True
        ),
        "configured_lbfgsb_policy_upper_bound": lbfgsb_policy_upper_bound(),
        "configured_lbfgsb_removal_active_set_requested": env_bool(
            "LBFGSB_REMOVAL_ACTIVE_SET", adjoint_local_optimizer_mode()
        ),
        "configured_lbfgsb_removal_active_set": removal_active_set_required(
            "adjoint" if adjoint_local_optimizer_mode() else "finite_difference"
        ),
        "configured_lbfgsb_removal_probe_steps": os.environ.get(
            "LBFGSB_REMOVAL_PROBE_STEPS", "1e-6,1e-5,1e-4,1e-3,1e-2"
        ),
        "configured_lbfgsb_removal_screen_steps": os.environ.get(
            "LBFGSB_REMOVAL_SCREEN_STEPS", "1e-4,1e-2"
        ),
        "configured_lbfgsb_removal_gain_tol": env_float(
            "LBFGSB_REMOVAL_GAIN_TOL", 1e-8
        ),
        "configured_lbfgsb_removal_max_rounds": env_int(
            "LBFGSB_REMOVAL_MAX_ROUNDS", 8
        ),
        "configured_lbfgsb_removal_batch_size": env_int(
            "LBFGSB_REMOVAL_BATCH_SIZE", 1
        ),
        "configured_lbfgsb_removal_proposal_smoothing_width": env_float(
            "LBFGSB_REMOVAL_PROPOSAL_SMOOTHING_WIDTH", 0.0
        ),
        "configured_lbfgsb_removal_proposal_smoothing_mode": os.environ.get(
            "LBFGSB_REMOVAL_PROPOSAL_SMOOTHING_MODE", "symmetric_huber"
        ).strip().lower(),
        "configured_lbfgsb_removal_proposal_margin": env_float(
            "LBFGSB_REMOVAL_PROPOSAL_MARGIN", 1e-6
        ),
        "configured_lbfgsb_removal_proposal_maxiter": env_int(
            "LBFGSB_REMOVAL_PROPOSAL_MAXITER", 200
        ),
        "configured_lbfgsb_removal_polish_restarts": env_int(
            "LBFGSB_REMOVAL_POLISH_RESTARTS", 2
        ),
        "configured_lbfgsb_removal_stage0_polish_restarts": env_int(
            "LBFGSB_REMOVAL_STAGE0_POLISH_RESTARTS", 2
        ),
        "configured_start_design": os.environ.get("START_DESIGN", "sobol"),
    })

    if mode == "ga_adjoint_lbfgsb":
        configured.update({
            "configured_ga_adjoint_pop_amount": env_int(
                "GA_ADJOINT_POP_AMOUNT", env_int("N_GA_POP", 400)
            ),
            "configured_ga_adjoint_generations": env_int(
                "GA_ADJOINT_GENERATIONS", env_int("N_GENERATIONS_GA", 200)
            ),
            "configured_ga_adjoint_top_starts": env_int(
                "GA_ADJOINT_TOP_STARTS", 8
            ),
            "configured_ga_adjoint_diverse_starts": env_int(
                "GA_ADJOINT_DIVERSE_STARTS", 8
            ),
        })
    return configured


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
        "smooth_projected_gradient_pass",
        "nonsmooth_kkt_status",
        "nonsmooth_kkt_detected_knots",
        "nonsmooth_kkt_detected_damage_knots",
        "nonsmooth_kkt_detected_cost_kink_nodes",
        "nonsmooth_kkt_evals",
        "nonsmooth_kkt_max_utility_gain",
        "nonsmooth_kkt_pass",
        "nonsmooth_kkt_override_applied",
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
        "removal_active_set_status",
        "removal_active_set_pass",
        "removal_active_set_activated_nodes",
        "removal_active_set_final_best_inactive_gain",
        "removal_active_set_no_improving_inactive_nodes",
    ]
    details = ", ".join(
        "{}={}".format(field, diagnostics.get(field))
        for field in fields
        if field in diagnostics
    )
    if not env_bool("LBFGSB_ABORT_ON_DIAGNOSTIC_FAILURE", True):
        print(
            "WARNING: L-BFGS-B {} solve failed diagnostics; continuing because "
            "LBFGSB_ABORT_ON_DIAGNOSTIC_FAILURE=0. {}".format(
                scenario_name, details
            ),
            flush=True,
        )
        return
    raise RuntimeError(
        "L-BFGS-B {} solve failed diagnostics; aborting because GA fallback is disabled. {}".format(
            scenario_name, details
        )
    )


def make_policy_bounds(num_nodes, fixed_indices=None, fixed_values=None, upper_bounds=None,
                       policy_upper_bound=None):
    lower = np.zeros(num_nodes, dtype=float)
    policy_cap = (
        lbfgsb_policy_upper_bound()
        if policy_upper_bound is None else float(policy_upper_bound)
    )
    if not np.isfinite(policy_cap) or policy_cap <= 0.0:
        raise ValueError("policy_upper_bound must be finite and positive")
    upper = np.full(num_nodes, policy_cap, dtype=float)
    if upper_bounds is not None:
        upper = np.minimum(upper, np.asarray(upper_bounds, dtype=float))
    if fixed_indices is not None:
        fixed_indices = np.asarray(fixed_indices, dtype=int)
        fixed_values = np.asarray(fixed_values, dtype=float).flatten()
        lower[fixed_indices] = fixed_values
        upper[fixed_indices] = fixed_values
    return lower, upper


def run_lbfgsb_policy(utility, num_nodes, scenario_name, warm_starts=None,
                      fixed_indices=None, fixed_values=None, upper_bounds=None,
                      seed_parts=(), print_progress=False,
                      gradient_mode="finite_difference", policy_upper_bound=None,
                      mandatory_starts=None, optimizer_options=None):
    lower, upper = make_policy_bounds(
        num_nodes,
        fixed_indices=fixed_indices,
        fixed_values=fixed_values,
        upper_bounds=upper_bounds,
        policy_upper_bound=policy_upper_bound,
    )
    optimizer_options = dict(optimizer_options or {})
    seed = stable_seed(os.environ.get("RANDOM_SEED_BASE", "20250706"), scenario_name, *seed_parts)
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
        warm_starts=[] if warm_starts is None else warm_starts,
        mandatory_starts=[] if mandatory_starts is None else mandatory_starts,
        n_candidates=int(optimizer_options.get("n_candidates", env_int("N_CANDIDATES", 256))),
        n_local_starts=int(optimizer_options.get("n_local_starts", env_int("N_LOCAL_STARTS", 8))),
        max_candidates=int(optimizer_options.get("max_candidates", env_int("MAX_CANDIDATES", 1024))),
        max_local_starts=int(optimizer_options.get("max_local_starts", env_int("MAX_LOCAL_STARTS", 16))),
        maxiter=int(optimizer_options.get(
            "maxiter", env_int("LBFGSB_MAXITER", 150)
        )),
        ftol=env_float("LBFGSB_FTOL", 1e-7),
        gtol=env_float("LBFGSB_GTOL", 1e-5),
        utility_spread_tol=env_float("UTILITY_SPREAD_TOL", 1e-7),
        utility_spread_rel_tol=env_float("UTILITY_SPREAD_REL_TOL", 1e-3),
        escalate_on_dispersion=bool(optimizer_options.get(
            "escalate_on_dispersion", env_bool("ESCALATE_ON_DISPERSION", True)
        )),
        start_design=os.environ.get("START_DESIGN", "sobol"),
        seed=seed,
        print_progress=print_progress,
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
        warm_start_perturbations=int(optimizer_options.get(
            "warm_start_perturbations", env_int("LBFGSB_WARM_START_PERTURBATIONS", 16)
        )),
        warm_start_perturbation_scale=env_float(
            "LBFGSB_WARM_START_PERTURBATION_SCALE", 0.05
        ),
        structured_start_count=int(optimizer_options.get(
            "structured_start_count", env_int("LBFGSB_STRUCTURED_STARTS", 64)
        )),
        near_full_mitigation=env_float("LBFGSB_NEAR_FULL_MITIGATION", 0.98),
        start_boundary_epsilon=float(optimizer_options.get(
            "start_boundary_epsilon", env_float("LBFGSB_START_BOUNDARY_EPSILON", 1e-6)
        )),
        preserve_diverse_starts=bool(optimizer_options.get(
            "preserve_diverse_starts", env_bool("LBFGSB_PRESERVE_DIVERSE_STARTS", True)
        )),
        min_diverse_start_groups=int(optimizer_options.get(
            "min_diverse_start_groups", env_int("LBFGSB_MIN_DIVERSE_START_GROUPS", 4)
        )),
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
        nonsmooth_kkt_check=env_bool("LBFGSB_NONSMOOTH_KKT_CHECK", True),
        nonsmooth_kkt_step=env_float("LBFGSB_NONSMOOTH_KKT_STEP", 1e-4),
        nonsmooth_kkt_utility_gain_tol=env_float(
            "LBFGSB_NONSMOOTH_KKT_UTILITY_GAIN_TOL", 1e-8
        ),
        nonsmooth_kkt_max_coordinates=env_int(
            "LBFGSB_NONSMOOTH_KKT_MAX_COORDINATES", 32
        ),
        validate_gradient=bool(optimizer_options.get(
            "validate_gradient",
            env_bool("ADJOINT_VALIDATE_GRADIENT", gradient_mode == "adjoint"),
        )),
        gradient_validation_directions=env_int("ADJOINT_VALIDATION_DIRECTIONS", 4),
        gradient_validation_abs_tol=env_float("ADJOINT_VALIDATION_ABS_TOL", 1e-5),
        gradient_validation_rel_tol=env_float("ADJOINT_VALIDATION_REL_TOL", 1e-3),
    )
    mitigation, utility_value, diagnostics = optimizer.run()
    diagnostics["lbfgsb_policy_upper_bound"] = float(
        lbfgsb_policy_upper_bound() if policy_upper_bound is None else policy_upper_bound
    )
    diagnostics["lbfgsb_effective_upper_bound_min"] = float(np.min(upper))
    diagnostics["lbfgsb_effective_upper_bound_max"] = float(np.max(upper))
    diagnostics["lbfgsb_n_removal_enabled_nodes"] = int(np.sum(upper > 1.0 + 1e-8))
    diagnostics["lbfgsb_seed"] = int(seed)
    return mitigation, utility_value, diagnostics


def _smoothed_removal_proposal_utility(utility, smoothing_width, smoothing_mode):
    """Return a shallow utility clone with a smooth removal premium.

    This object is a proposal generator only.  Callers must still evaluate
    every selected node and the final policy under the original sharp utility.
    The shallow copies preserve the tree, damages, and calibrated cost
    parameters while replacing only the numerical transition at full mitigation.
    """

    if not np.isfinite(smoothing_width) or smoothing_width <= 0.0:
        raise ValueError("proposal smoothing width must be finite and positive")
    if smoothing_mode not in ("one_sided_huber", "symmetric_huber"):
        raise ValueError("unrecognized proposal smoothing mode")
    if not hasattr(utility, "cost"):
        raise TypeError("smooth removal proposal requires a utility with a cost")
    proposal_utility = copy.copy(utility)
    proposal_cost = copy.copy(utility.cost)
    if not hasattr(proposal_cost, "backstop_smoothing_width"):
        raise TypeError("smooth removal proposal requires a configurable cost")
    proposal_cost.backstop_smoothing_width = float(smoothing_width)
    proposal_cost.backstop_smoothing_mode = str(smoothing_mode)
    proposal_utility.cost = proposal_cost
    return proposal_utility


def _prioritize_removal_candidates(candidates, proposal_nodes):
    """Put smooth-proposal nodes first, retaining sharp-audit order within groups."""

    proposal_nodes = set(int(node) for node in proposal_nodes)
    proposed = [
        candidate for candidate in candidates
        if int(candidate["node"]) in proposal_nodes
    ]
    remaining = [
        candidate for candidate in candidates
        if int(candidate["node"]) not in proposal_nodes
    ]
    return proposed + remaining


def _parse_positive_float_list(name, default):
    raw = os.environ.get(name, default)
    values = tuple(sorted(set(
        float(value.strip()) for value in raw.split(",") if value.strip()
    )))
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("{} must contain positive finite values".format(name))
    return values


def _policy_utility_value(utility, mitigation):
    value = float(np.asarray(utility.utility(
        np.asarray(mitigation, dtype=float)
    )).reshape(-1)[0])
    if not np.isfinite(value):
        raise FloatingPointError("non-finite utility in removal active-set audit")
    return value


def _removal_probe_audit(utility, mitigation, eligible_nodes, active_nodes,
                         final_upper, probe_steps, screen_steps, gain_tol,
                         kink_tol, min_positive_scales):
    mitigation = np.asarray(mitigation, dtype=float)
    active_nodes = set(int(node) for node in active_nodes)
    base_utility = _policy_utility_value(utility, mitigation)
    candidates = []
    ambiguous_candidates = []
    probe_evals = 0
    best_gain = -np.inf
    best_node = -1
    best_step = np.nan
    tested_nodes = 0
    all_finite = True
    full_probe_coverage_complete = True
    gain_exceedance_count = 0
    full_steps = tuple(sorted(set(probe_steps)))
    screen_steps = tuple(sorted(set(screen_steps)))

    for node in np.asarray(eligible_nodes, dtype=int):
        node = int(node)
        if node in active_nodes or abs(float(mitigation[node]) - 1.0) > kink_tol:
            continue
        tested_nodes += 1
        gains_by_step = {}

        def evaluate(step):
            nonlocal probe_evals, all_finite
            actual_step = min(
                float(step),
                float(final_upper[node] - mitigation[node]),
            )
            if actual_step <= 0.0:
                return
            trial = mitigation.copy()
            trial[node] += actual_step
            try:
                gain = _policy_utility_value(utility, trial) - base_utility
            except FloatingPointError:
                all_finite = False
                gain = np.nan
            probe_evals += 1
            gains_by_step[actual_step] = float(gain)

        for step in screen_steps:
            evaluate(step)
        for step in full_steps:
            actual_step = min(
                float(step), float(final_upper[node] - mitigation[node])
            )
            if actual_step > 0.0 and actual_step not in gains_by_step:
                evaluate(step)
        expected_full_steps = {
            min(float(step), float(final_upper[node] - mitigation[node]))
            for step in full_steps
            if min(float(step), float(final_upper[node] - mitigation[node])) > 0.0
        }
        full_probe_coverage_complete = bool(
            full_probe_coverage_complete
            and expected_full_steps.issubset(set(gains_by_step))
        )

        finite_items = [
            (step, gain) for step, gain in gains_by_step.items()
            if np.isfinite(gain)
        ]
        gain_exceedance_count += sum(
            gain > gain_tol for _, gain in finite_items
        )
        for step, gain in finite_items:
            if gain > best_gain:
                best_gain = float(gain)
                best_node = node
                best_step = float(step)
        positive = [
            (step, gain) for step, gain in finite_items
            if gain > gain_tol
        ]
        if positive:
            candidate = {
                "node": node,
                "max_gain": float(max(gain for _, gain in positive)),
                "seed_step": float(max(positive, key=lambda item: item[0])[0]),
                "positive_scales": int(len(positive)),
            }
            if len(positive) >= min_positive_scales:
                candidates.append(candidate)
            else:
                ambiguous_candidates.append(candidate)

    candidates.sort(key=lambda item: (item["max_gain"], -item["node"]), reverse=True)
    ambiguous_candidates.sort(
        key=lambda item: (item["max_gain"], -item["node"]), reverse=True
    )
    return {
        "base_utility": float(base_utility),
        "candidates": candidates,
        "ambiguous_candidates": ambiguous_candidates,
        "probe_evals": int(probe_evals),
        "tested_nodes": int(tested_nodes),
        "all_finite": bool(all_finite),
        "full_probe_coverage_complete": bool(full_probe_coverage_complete),
        "full_probe_scale_count": int(len(full_steps)),
        "gain_exceedance_count": int(gain_exceedance_count),
        "best_gain": float(best_gain) if np.isfinite(best_gain) else np.nan,
        "best_node": int(best_node),
        "best_step": float(best_step) if np.isfinite(best_step) else np.nan,
    }


def _tag_local_results(diagnostics, stage_name):
    tagged = []
    for result in diagnostics.get("_local_results", []) or []:
        copied = dict(result)
        copied["removal_active_set_stage"] = str(stage_name)
        tagged.append(copied)
    return tagged


def _retryable_lbfgsb_stationarity_failure(diagnostics):
    return bool(
        diagnostics.get("lbfgsb_best_result_accepted", False)
        and diagnostics.get("stationarity_failed", False)
        and not diagnostics.get("dispersion_failed", False)
        and not diagnostics.get("perturbation_failed", False)
        and diagnostics.get("gradient_validation_status") == "passed"
    )


def _cap_one_domain_infeasible(diagnostics):
    """Whether gradient validation found no finite policy in the m <= 1 face."""

    return bool(
        diagnostics.get("gradient_validation_status") == "failed"
        and "no finite interior objective point" in str(
            diagnostics.get("gradient_validation_message", "")
        )
    )


def _cap_one_gradient_validation_exception(exc):
    """Whether the cap-one solver raised the expected infeasibility error."""

    return "gradient validation failed: no finite interior objective point" in str(exc)


def run_lbfgsb_policy_removal_active_set(
        utility, num_nodes, scenario_name, warm_starts=None,
        fixed_indices=None, fixed_values=None, upper_bounds=None,
        seed_parts=(), print_progress=False, gradient_mode="adjoint"):
    started = time.time()
    final_cap = lbfgsb_policy_upper_bound()
    final_lower, final_upper = make_policy_bounds(
        num_nodes,
        fixed_indices=fixed_indices,
        fixed_values=fixed_values,
        upper_bounds=upper_bounds,
        policy_upper_bound=final_cap,
    )
    fixed_set = set(
        int(node) for node in (
            [] if fixed_indices is None else np.asarray(fixed_indices, dtype=int)
        )
    )
    kink_tol = env_float("LBFGSB_REMOVAL_KINK_TOL", 1e-6)
    gain_tol = env_float("LBFGSB_REMOVAL_GAIN_TOL", 1e-8)
    probe_steps = _parse_positive_float_list(
        "LBFGSB_REMOVAL_PROBE_STEPS", "1e-6,1e-5,1e-4,1e-3,1e-2"
    )
    screen_steps = _parse_positive_float_list(
        "LBFGSB_REMOVAL_SCREEN_STEPS", "1e-4,1e-2"
    )
    min_positive_scales = env_int("LBFGSB_REMOVAL_MIN_POSITIVE_SCALES", 3)
    escalate_ambiguous_branches = env_bool(
        "LBFGSB_REMOVAL_ESCALATE_AMBIGUOUS", True
    )
    face_escalation = env_bool("LBFGSB_REMOVAL_FACE_ESCALATION", True)
    max_rounds = env_int("LBFGSB_REMOVAL_MAX_ROUNDS", 8)
    batch_size = env_int("LBFGSB_REMOVAL_BATCH_SIZE", 1)
    proposal_smoothing_width = env_float(
        "LBFGSB_REMOVAL_PROPOSAL_SMOOTHING_WIDTH", 0.0
    )
    proposal_smoothing_mode = os.environ.get(
        "LBFGSB_REMOVAL_PROPOSAL_SMOOTHING_MODE", "symmetric_huber"
    ).strip().lower()
    proposal_margin = env_float(
        "LBFGSB_REMOVAL_PROPOSAL_MARGIN", kink_tol
    )
    proposal_maxiter = env_int("LBFGSB_REMOVAL_PROPOSAL_MAXITER", 200)
    polish_restarts = env_int("LBFGSB_REMOVAL_POLISH_RESTARTS", 2)
    stage0_polish_restarts = env_int(
        "LBFGSB_REMOVAL_STAGE0_POLISH_RESTARTS", 2
    )
    if not np.isfinite(kink_tol) or kink_tol <= 0.0:
        raise ValueError("LBFGSB_REMOVAL_KINK_TOL must be finite and positive")
    if not np.isfinite(gain_tol) or gain_tol < 0.0:
        raise ValueError("LBFGSB_REMOVAL_GAIN_TOL must be finite and nonnegative")
    if min_positive_scales <= 0 or min_positive_scales > len(probe_steps):
        raise ValueError(
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES must be between 1 and the "
            "number of LBFGSB_REMOVAL_PROBE_STEPS"
        )
    if not set(screen_steps).issubset(set(probe_steps)):
        raise ValueError(
            "LBFGSB_REMOVAL_SCREEN_STEPS must be a subset of "
            "LBFGSB_REMOVAL_PROBE_STEPS"
        )
    if max_rounds < 0:
        raise ValueError("LBFGSB_REMOVAL_MAX_ROUNDS must be nonnegative")
    if batch_size <= 0:
        raise ValueError("LBFGSB_REMOVAL_BATCH_SIZE must be positive")
    if (
        not np.isfinite(proposal_smoothing_width)
        or proposal_smoothing_width < 0.0
    ):
        raise ValueError(
            "LBFGSB_REMOVAL_PROPOSAL_SMOOTHING_WIDTH must be finite and nonnegative"
        )
    if proposal_smoothing_mode not in ("one_sided_huber", "symmetric_huber"):
        raise ValueError(
            "LBFGSB_REMOVAL_PROPOSAL_SMOOTHING_MODE must be one_sided_huber or symmetric_huber"
        )
    if not np.isfinite(proposal_margin) or proposal_margin < 0.0:
        raise ValueError(
            "LBFGSB_REMOVAL_PROPOSAL_MARGIN must be finite and nonnegative"
        )
    if proposal_maxiter <= 0:
        raise ValueError("LBFGSB_REMOVAL_PROPOSAL_MAXITER must be positive")
    if polish_restarts < 0:
        raise ValueError("LBFGSB_REMOVAL_POLISH_RESTARTS must be nonnegative")
    if stage0_polish_restarts <= 0:
        raise ValueError(
            "LBFGSB_REMOVAL_STAGE0_POLISH_RESTARTS must be positive"
        )
    eligible_nodes = np.asarray([
        node for node in range(num_nodes)
        if node not in fixed_set and final_upper[node] > 1.0 + kink_tol
    ], dtype=int)

    single_start_options = {
        "n_candidates": 1,
        "n_local_starts": 1,
        "max_candidates": 1,
        "max_local_starts": 1,
        "escalate_on_dispersion": False,
        "warm_start_perturbations": 0,
        "structured_start_count": 0,
        "start_boundary_epsilon": 0.0,
        "preserve_diverse_starts": False,
        "min_diverse_start_groups": 0,
    }
    try:
        stage0_m, stage0_u, stage0_diag = run_lbfgsb_policy(
            utility,
            num_nodes,
            "{}_cap1_stage".format(scenario_name),
            warm_starts=warm_starts,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            upper_bounds=upper_bounds,
            seed_parts=tuple(seed_parts) + ("cap1_stage",),
            print_progress=print_progress,
            gradient_mode=gradient_mode,
            policy_upper_bound=min(1.0, final_cap),
        )
    except RuntimeError as exc:
        if not _cap_one_gradient_validation_exception(exc):
            raise
        _, cap_one_upper = make_policy_bounds(
            num_nodes,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            upper_bounds=upper_bounds,
            policy_upper_bound=min(1.0, final_cap),
        )
        stage0_m = np.asarray(cap_one_upper, dtype=float).copy()
        stage0_u = -np.inf
        stage0_diag = {
            "lbfgsb_success": False,
            "success_diagnostics": False,
            "gradient_validation_status": "failed",
            "gradient_validation_message": (
                "no finite interior objective point found for gradient validation"
            ),
            "lbfgsb_message": str(exc),
            "runtime_seconds": np.nan,
            "_local_results": [],
        }
    cap_one_initial_utility = float(stage0_u)
    cap_one_initial_success = bool(stage0_diag.get("lbfgsb_success", False))
    cap_one_infeasible = _cap_one_domain_infeasible(stage0_diag)
    full_removal_feasibility_fallback = False
    full_removal_boundary_validation_override = False
    full_removal_feasibility_seed = np.asarray(final_upper, dtype=float).copy()
    stage0_initial_utility = float(stage0_u)
    stage0_initial_success = bool(stage0_diag.get("lbfgsb_success", False))
    combined_local_results = _tag_local_results(stage0_diag, "cap1_stage")
    stage_summaries = [{
        "stage": "cap1_stage",
        "utility": float(stage0_u),
        "success": stage0_initial_success,
        "runtime_seconds": float(stage0_diag.get("runtime_seconds", np.nan)),
        "active_nodes": "",
    }]
    mandatory_selection_results = []
    stage0_polish_attempts = 0

    if cap_one_infeasible:
        print(
            "{} cap-one domain has no finite interior policy; starting exact "
            "full-removal feasibility solve".format(scenario_name),
            flush=True,
        )
        stage_label = "full_removal_feasibility_stage"
        try:
            stage0_m, stage0_u, stage0_diag = run_lbfgsb_policy(
                utility,
                num_nodes,
                "{}_{}".format(scenario_name, stage_label),
                warm_starts=[],
                fixed_indices=fixed_indices,
                fixed_values=fixed_values,
                upper_bounds=upper_bounds,
                seed_parts=tuple(seed_parts) + ("full_removal_feasibility",),
                print_progress=print_progress,
                gradient_mode=gradient_mode,
                policy_upper_bound=final_cap,
                mandatory_starts=[full_removal_feasibility_seed],
                optimizer_options=dict(
                    single_start_options, start_boundary_epsilon=0.0
                ),
            )
        except RuntimeError as exc:
            if not _cap_one_gradient_validation_exception(exc):
                raise
            print(
                "{} full-removal domain has no finite interior policy; "
                "using exact boundary-feasibility solve".format(scenario_name),
                flush=True,
            )
            stage_label = "full_removal_boundary_feasibility_stage"
            stage0_m, stage0_u, stage0_diag = run_lbfgsb_policy(
                utility,
                num_nodes,
                "{}_{}".format(scenario_name, stage_label),
                warm_starts=[],
                fixed_indices=fixed_indices,
                fixed_values=fixed_values,
                upper_bounds=upper_bounds,
                seed_parts=tuple(seed_parts) + ("full_removal_boundary_feasibility",),
                print_progress=print_progress,
                gradient_mode=gradient_mode,
                policy_upper_bound=final_cap,
                mandatory_starts=[full_removal_feasibility_seed],
                optimizer_options=dict(
                    single_start_options,
                    start_boundary_epsilon=0.0,
                    validate_gradient=False,
                ),
            )
            full_removal_boundary_validation_override = True
        mandatory_selected = bool(
            stage0_diag.get("mandatory_starts_selected", False)
        )
        mandatory_selection_results.append(mandatory_selected)
        if not mandatory_selected:
            raise RuntimeError(
                "{} {} did not run its mandatory full-removal seed".format(
                    scenario_name, stage_label
                )
            )
        combined_local_results.extend(
            _tag_local_results(stage0_diag, stage_label)
        )
        stage_summaries.append({
            "stage": stage_label,
            "utility": float(stage0_u),
            "success": bool(stage0_diag.get("lbfgsb_success", False)),
            "runtime_seconds": float(
                stage0_diag.get("runtime_seconds", np.nan)
            ),
            "active_nodes": "",
        })
        require_lbfgsb_success(
            stage0_diag, "{}_{}".format(scenario_name, stage_label)
        )
        full_removal_feasibility_fallback = True
        stage0_initial_utility = float(stage0_u)
        stage0_initial_success = bool(stage0_diag.get("lbfgsb_success", False))

    for polish_attempt in range(1, stage0_polish_restarts + 1):
        if stage0_diag.get("lbfgsb_success", False):
            break
        if not _retryable_lbfgsb_stationarity_failure(stage0_diag):
            require_lbfgsb_success(
                stage0_diag, "{}_cap1_stage".format(scenario_name)
            )
            raise RuntimeError(
                "{} cap-one stage failed a nonretryable diagnostic".format(
                    scenario_name
                )
            )
        previous_m = np.asarray(stage0_m, dtype=float).copy()
        previous_u = float(stage0_u)
        stage_label = "cap1_stage_polish_{}".format(polish_attempt)
        stage0_m, stage0_u, stage0_diag = run_lbfgsb_policy(
            utility,
            num_nodes,
            "{}_{}".format(scenario_name, stage_label),
            warm_starts=[],
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            upper_bounds=upper_bounds,
            seed_parts=tuple(seed_parts) + (
                "cap1_stage_polish", polish_attempt
            ),
            print_progress=print_progress,
            gradient_mode=gradient_mode,
            policy_upper_bound=min(1.0, final_cap),
            mandatory_starts=[previous_m],
            optimizer_options=single_start_options,
        )
        stage0_polish_attempts += 1
        mandatory_selected = bool(
            stage0_diag.get("mandatory_starts_selected", False)
        )
        mandatory_selection_results.append(mandatory_selected)
        if not mandatory_selected:
            raise RuntimeError(
                "{} {} did not run its mandatory incumbent seed".format(
                    scenario_name, stage_label
                )
            )
        if float(stage0_u) + gain_tol < previous_u:
            raise RuntimeError(
                "{} {} reduced utility from {:.15g} to {:.15g}".format(
                    scenario_name, stage_label, previous_u, float(stage0_u)
                )
            )
        combined_local_results.extend(
            _tag_local_results(stage0_diag, stage_label)
        )
        stage_summaries.append({
            "stage": stage_label,
            "utility": float(stage0_u),
            "success": bool(stage0_diag.get("lbfgsb_success", False)),
            "runtime_seconds": float(
                stage0_diag.get("runtime_seconds", np.nan)
            ),
            "active_nodes": "",
        })

    if not stage0_diag.get("lbfgsb_success", False):
        require_lbfgsb_success(
            stage0_diag, "{}_cap1_stage".format(scenario_name)
        )
        raise RuntimeError(
            "{} cap-one stage exhausted {} stationarity polishes".format(
                scenario_name, stage0_polish_restarts
            )
        )

    proposal_nodes = set()
    proposal_diagnostics = {
        "requested": bool(proposal_smoothing_width > 0.0),
        "ran": False,
        "finite": False,
        "solver_success": False,
        "support_count": 0,
        "error": "",
    }
    if proposal_smoothing_width > 0.0:
        try:
            proposal_utility = _smoothed_removal_proposal_utility(
                utility, proposal_smoothing_width, proposal_smoothing_mode
            )
            proposal_m, proposal_u, proposal_diag = run_lbfgsb_policy(
                proposal_utility,
                num_nodes,
                "{}_smooth_removal_proposal".format(scenario_name),
                fixed_indices=fixed_indices,
                fixed_values=fixed_values,
                upper_bounds=upper_bounds,
                seed_parts=tuple(seed_parts) + ("smooth_removal_proposal",),
                print_progress=print_progress,
                gradient_mode=gradient_mode,
                policy_upper_bound=final_cap,
                mandatory_starts=[stage0_m],
                optimizer_options=dict(single_start_options, maxiter=proposal_maxiter),
            )
            proposal_m = np.asarray(proposal_m, dtype=float)
            proposal_finite = bool(
                np.isfinite(proposal_u) and np.all(np.isfinite(proposal_m))
            )
            proposal_diagnostics.update({
                "ran": True,
                "finite": proposal_finite,
                "solver_success": bool(proposal_diag.get("lbfgsb_success", False)),
            })
            if proposal_finite:
                proposal_nodes = set(
                    int(node) for node in eligible_nodes
                    if proposal_m[node] > 1.0 + proposal_margin
                )
                proposal_diagnostics["support_count"] = int(len(proposal_nodes))
            else:
                proposal_diagnostics["error"] = "nonfinite proposal result"
        except Exception as exc:
            proposal_diagnostics.update({"ran": True, "error": repr(exc)})
            print(
                "Smooth removal proposal unavailable; continuing with exact sharp audit: {}".format(exc),
                flush=True,
            )

    if proposal_smoothing_width > 0.0:
        print(
            "Smooth removal proposal: finite={}; solver_success={}; support_nodes={}"
            .format(
                proposal_diagnostics["finite"],
                proposal_diagnostics["solver_success"],
                sorted(proposal_nodes),
            ),
            flush=True,
        )

    current_m = np.asarray(stage0_m, dtype=float)
    current_u = float(stage0_u)
    current_diag = stage0_diag
    active_nodes = set(
        int(node) for node in eligible_nodes
        if full_removal_feasibility_fallback
        and current_m[node] > 1.0 + kink_tol
    )
    nodes_by_round = []
    deactivated_nodes = []
    deactivation_events = []
    total_probe_evals = 0
    all_probe_evals_finite = True
    final_audit = None
    ambiguous_branch_activations = []
    face_escalation_used = False

    for active_round in range(max_rounds + 1):
        audit = _removal_probe_audit(
            utility,
            current_m,
            eligible_nodes,
            active_nodes,
            final_upper,
            probe_steps,
            screen_steps,
            gain_tol,
            kink_tol,
            min_positive_scales,
        )
        total_probe_evals += audit["probe_evals"]
        all_probe_evals_finite = bool(
            all_probe_evals_finite and audit["all_finite"]
        )
        if not audit["all_finite"]:
            raise RuntimeError(
                "non-finite removal probe in {} active-set audit".format(
                    scenario_name
                )
            )
        if not audit["full_probe_coverage_complete"]:
            raise RuntimeError(
                "{} removal audit did not cover every configured probe scale".format(
                    scenario_name
                )
            )
        final_audit = audit
        print(
            "Removal active-set audit round {}: tested_nodes={}; "
            "robust_candidates={}; best_gain={:.12g}; best_node={}".format(
                active_round + 1,
                audit["tested_nodes"],
                len(audit["candidates"]),
                audit["best_gain"],
                audit["best_node"],
            ),
            flush=True,
        )
        candidate_pool = audit["candidates"]
        activation_source = "multi-scale sharp-verified"
        if not candidate_pool:
            if (
                not np.isfinite(audit["best_gain"])
                or audit["best_gain"] <= gain_tol
            ):
                break
            if not escalate_ambiguous_branches:
                raise RuntimeError(
                    "{} removal audit is inconclusive: inactive node {} has "
                    "positive gain {:.15g} at step {:.3g}, but fewer than {} "
                    "probe scales passed".format(
                        scenario_name, audit["best_node"], audit["best_gain"],
                        audit["best_step"], min_positive_scales
                    )
                )
            candidate_pool = audit.get("ambiguous_candidates", [])
            if not candidate_pool:
                raise RuntimeError(
                    "{} removal audit found a positive gain but no exact branch "
                    "seed".format(scenario_name)
                )
            activation_source = "single-scale exact-branch"
        if active_round >= max_rounds:
            raise RuntimeError(
                "{} removal active set exhausted {} rounds with an improving "
                "inactive node remaining".format(scenario_name, max_rounds)
            )

        ranked_candidates = _prioritize_removal_candidates(
            candidate_pool, proposal_nodes
        )
        chosen_batch = ranked_candidates[:batch_size]
        chosen_nodes = [int(chosen["node"]) for chosen in chosen_batch]
        if activation_source != "multi-scale sharp-verified":
            ambiguous_branch_activations.append(
                "round{}:{}".format(
                    active_round + 1, "+".join(str(node) for node in chosen_nodes)
                )
            )
        print(
            "Removal active-set round {} activating {} nodes: {}"
            .format(active_round + 1, activation_source, chosen_nodes),
            flush=True,
        )
        active_nodes.update(chosen_nodes)
        nodes_by_round.append("+".join(str(node) for node in chosen_nodes))
        seed = current_m.copy()
        for chosen in chosen_batch:
            chosen_node = int(chosen["node"])
            seed[chosen_node] = min(
                float(final_upper[chosen_node]),
                float(current_m[chosen_node]) + float(chosen["seed_step"]),
            )
        incumbent_u = current_u
        candidate_m = None
        candidate_u = None
        candidate_diag = None
        face_restart = 0
        while True:
            mixed_upper = np.minimum(final_upper, 1.0)
            if active_nodes:
                active_index = np.asarray(sorted(active_nodes), dtype=int)
                mixed_upper[active_index] = final_upper[active_index]

            accepted = False
            restart_face = False
            for polish_attempt in range(polish_restarts + 1):
                stage_label = "removal_round_{}_polish_{}".format(
                    active_round + 1, polish_attempt
                )
                if face_restart:
                    stage_label = "{}_face_{}".format(stage_label, face_restart)
                candidate_m, candidate_u, candidate_diag = run_lbfgsb_policy(
                    utility,
                    num_nodes,
                    "{}_{}".format(scenario_name, stage_label),
                    warm_starts=[],
                    fixed_indices=fixed_indices,
                    fixed_values=fixed_values,
                    upper_bounds=mixed_upper,
                    seed_parts=(
                        tuple(seed_parts)
                        + ("removal_active_set", active_round + 1, polish_attempt)
                        + ((face_restart,) if face_restart else ())
                    ),
                    print_progress=print_progress,
                    gradient_mode=gradient_mode,
                    policy_upper_bound=final_cap,
                    mandatory_starts=[seed],
                    optimizer_options=single_start_options,
                )
                mandatory_selection_results.append(bool(
                    candidate_diag.get("mandatory_starts_selected", False)
                ))
                if not mandatory_selection_results[-1]:
                    raise RuntimeError(
                        "{} {} did not run its mandatory incumbent seed".format(
                            scenario_name, stage_label
                        )
                    )
                combined_local_results.extend(
                    _tag_local_results(candidate_diag, stage_label)
                )
                stage_summaries.append({
                    "stage": stage_label,
                    "utility": float(candidate_u),
                    "success": bool(candidate_diag.get("lbfgsb_success", False)),
                    "runtime_seconds": float(
                        candidate_diag.get("runtime_seconds", np.nan)
                    ),
                    "active_nodes": ",".join(
                        str(node) for node in sorted(active_nodes)
                    ),
                })

                candidate_array = np.asarray(candidate_m, dtype=float)
                settled_nodes = []
                if np.all(np.isfinite(candidate_array)):
                    settled_nodes = [
                        node for node in sorted(active_nodes)
                        if candidate_array[node] <= 1.0 + kink_tol
                    ]
                if settled_nodes:
                    print(
                        "Removal active-set round {} deactivating nodes at sharp "
                        "kink: {}".format(active_round + 1, settled_nodes),
                        flush=True,
                    )
                    active_nodes.difference_update(settled_nodes)
                    deactivated_nodes.extend(settled_nodes)
                    deactivation_events.append(
                        "round{}:kink:{}".format(
                            active_round + 1,
                            "+".join(str(node) for node in settled_nodes),
                        )
                    )
                    seed = candidate_array.copy()
                    seed[settled_nodes] = 1.0
                    restart_face = True
                    break

                if candidate_diag.get("lbfgsb_success", False):
                    accepted = True
                    break
                retryable = _retryable_lbfgsb_stationarity_failure(
                    candidate_diag
                )
                if not retryable:
                    require_lbfgsb_success(
                        candidate_diag,
                        "{}_{}".format(scenario_name, stage_label),
                    )
                if polish_attempt >= polish_restarts:
                    break
                seed = candidate_array.copy()

            if restart_face:
                if not active_nodes:
                    raise RuntimeError(
                        "{} removal branch lost every activated node at the sharp "
                        "kink".format(scenario_name)
                    )
                face_restart += 1
                continue
            if (
                not accepted
                and face_escalation
                and candidate_diag is not None
                and _retryable_lbfgsb_stationarity_failure(candidate_diag)
            ):
                face_escalation_used = True
                stage_label = "removal_round_{}_escalated".format(active_round + 1)
                rescue_seed = np.asarray(candidate_m, dtype=float).copy()
                candidate_m, candidate_u, candidate_diag = run_lbfgsb_policy(
                    utility,
                    num_nodes,
                    "{}_{}".format(scenario_name, stage_label),
                    warm_starts=[],
                    fixed_indices=fixed_indices,
                    fixed_values=fixed_values,
                    upper_bounds=mixed_upper,
                    seed_parts=tuple(seed_parts) + (
                        "removal_active_set", active_round + 1, "escalated"
                    ),
                    print_progress=print_progress,
                    gradient_mode=gradient_mode,
                    policy_upper_bound=final_cap,
                    mandatory_starts=[rescue_seed],
                )
                mandatory_selection_results.append(bool(
                    candidate_diag.get("mandatory_starts_selected", False)
                ))
                if not mandatory_selection_results[-1]:
                    raise RuntimeError(
                        "{} {} did not run its mandatory incumbent seed".format(
                            scenario_name, stage_label
                        )
                    )
                combined_local_results.extend(
                    _tag_local_results(candidate_diag, stage_label)
                )
                stage_summaries.append({
                    "stage": stage_label,
                    "utility": float(candidate_u),
                    "success": bool(candidate_diag.get("lbfgsb_success", False)),
                    "runtime_seconds": float(
                        candidate_diag.get("runtime_seconds", np.nan)
                    ),
                    "active_nodes": ",".join(
                        str(node) for node in sorted(active_nodes)
                    ),
                })
                accepted = bool(candidate_diag.get("lbfgsb_success", False))
            if not accepted:
                require_lbfgsb_success(
                    candidate_diag, "{} removal branch".format(scenario_name)
                )
                raise RuntimeError(
                    "{} removal branch was not accepted".format(scenario_name)
                )
            candidate_u = float(candidate_u)
            if candidate_u <= incumbent_u + gain_tol:
                raise RuntimeError(
                    "{} removal branch did not improve utility beyond tolerance: "
                    "incumbent={:.15g}, candidate={:.15g}, tolerance={:.3g}".format(
                        scenario_name, incumbent_u, candidate_u, gain_tol
                    )
                )

            redundant_nodes = []
            for node in sorted(active_nodes):
                dropped = np.asarray(candidate_m, dtype=float).copy()
                dropped[node] = 1.0
                drop_gain = _policy_utility_value(utility, dropped) - candidate_u
                total_probe_evals += 1
                if np.isfinite(drop_gain) and drop_gain > gain_tol:
                    redundant_nodes.append(node)
            if redundant_nodes:
                print(
                    "Removal active-set round {} deactivating redundant nodes: {}"
                    .format(active_round + 1, redundant_nodes),
                    flush=True,
                )
                active_nodes.difference_update(redundant_nodes)
                deactivated_nodes.extend(redundant_nodes)
                deactivation_events.append(
                    "round{}:redundant:{}".format(
                        active_round + 1,
                        "+".join(str(node) for node in redundant_nodes),
                    )
                )
                seed = np.asarray(candidate_m, dtype=float).copy()
                seed = np.asarray(candidate_m, dtype=float).copy()
                seed[redundant_nodes] = 1.0
                if not active_nodes:
                    raise RuntimeError(
                        "{} removal branch lost every activated node as redundant"
                        .format(scenario_name)
                    )
                face_restart += 1
                continue
            break

        current_m = np.asarray(candidate_m, dtype=float)
        current_u = candidate_u
        current_diag = candidate_diag

    if final_audit is None:
        raise RuntimeError("{} removal audit did not run".format(scenario_name))

    final_testable_nodes = int(sum(
        node not in active_nodes
        and abs(float(current_m[node]) - 1.0) <= kink_tol
        for node in eligible_nodes
    ))
    final_coverage_complete = bool(
        final_audit["tested_nodes"] == final_testable_nodes
    )
    no_improving_inactive_nodes = bool(
        not np.isfinite(final_audit["best_gain"])
        or final_audit["best_gain"] <= gain_tol
    )
    accepted_stages_success = bool(
        stage0_diag.get("lbfgsb_success", False)
        and current_diag.get("lbfgsb_success", False)
    )
    certificate_pass = bool(
        accepted_stages_success
        and all_probe_evals_finite
        and final_audit["full_probe_coverage_complete"]
        and final_coverage_complete
        and no_improving_inactive_nodes
    )

    diagnostics = dict(current_diag)
    diagnostics.update({
        "removal_active_set_enabled": True,
        "removal_active_set_status": "passed" if certificate_pass else "failed",
        "removal_active_set_pass": certificate_pass,
        "removal_active_set_base_cap": 1.0,
        "removal_active_set_final_cap": float(final_cap),
        "removal_active_set_cap_one_infeasible": bool(cap_one_infeasible),
        "removal_active_set_full_removal_feasibility_fallback": bool(
            full_removal_feasibility_fallback
        ),
        "removal_active_set_full_removal_feasibility_support_count": int(
            len(active_nodes) if full_removal_feasibility_fallback else 0
        ),
        "removal_active_set_full_removal_boundary_validation_override": bool(
            full_removal_boundary_validation_override
        ),
        "removal_active_set_probe_steps": ",".join(
            "{:.0e}".format(step) for step in probe_steps
        ),
        "removal_active_set_screen_steps": ",".join(
            "{:.0e}".format(step) for step in screen_steps
        ),
        "removal_active_set_gain_tol": float(gain_tol),
        "removal_active_set_eligible_nodes": int(len(eligible_nodes)),
        "removal_active_set_activated_count": int(len(active_nodes)),
        "removal_active_set_activated_nodes": ",".join(
            str(node) for node in sorted(active_nodes)
        ),
        "removal_active_set_nodes_by_round": ",".join(
            str(node) for node in nodes_by_round
        ),
        "removal_active_set_deactivated_count": int(len(set(deactivated_nodes))),
        "removal_active_set_deactivated_nodes": ",".join(
            str(node) for node in sorted(set(deactivated_nodes))
        ),
        "removal_active_set_deactivation_events": ",".join(
            str(event) for event in deactivation_events
        ),
        "removal_active_set_batch_size": int(batch_size),
        "removal_active_set_strategy": (
            "batch" if batch_size > 1 else "single_node"
        ),
        "removal_active_set_smooth_proposal_requested": bool(
            proposal_diagnostics["requested"]
        ),
        "removal_active_set_smooth_proposal_ran": bool(
            proposal_diagnostics["ran"]
        ),
        "removal_active_set_smooth_proposal_width": float(
            proposal_smoothing_width
        ),
        "removal_active_set_smooth_proposal_mode": proposal_smoothing_mode,
        "removal_active_set_smooth_proposal_margin": float(proposal_margin),
        "removal_active_set_smooth_proposal_maxiter": int(proposal_maxiter),
        "removal_active_set_smooth_proposal_finite": bool(
            proposal_diagnostics["finite"]
        ),
        "removal_active_set_smooth_proposal_solver_success": bool(
            proposal_diagnostics["solver_success"]
        ),
        "removal_active_set_smooth_proposal_support_count": int(
            proposal_diagnostics["support_count"]
        ),
        "removal_active_set_smooth_proposal_support_nodes": ",".join(
            str(node) for node in sorted(proposal_nodes)
        ),
        "removal_active_set_smooth_proposal_error": proposal_diagnostics["error"],
        "removal_active_set_rounds": int(len(nodes_by_round)),
        "removal_active_set_stage_count": int(len(stage_summaries)),
        "removal_active_set_branch_attempts": int(
            len(mandatory_selection_results) - stage0_polish_attempts
        ),
        "removal_active_set_stage0_polish_attempts": int(
            stage0_polish_attempts
        ),
        "removal_active_set_stage0_polish_restarts": int(
            stage0_polish_restarts
        ),
        "removal_active_set_all_mandatory_starts_selected": bool(
            all(mandatory_selection_results)
        ),
        "removal_active_set_stage_utilities": ",".join(
            "{:.15g}".format(item["utility"]) for item in stage_summaries
        ),
        "removal_active_set_all_solver_attempts_success": bool(
            all(item["success"] for item in stage_summaries)
        ),
        "removal_active_set_failed_solver_attempts": int(
            sum(not item["success"] for item in stage_summaries)
        ),
        "removal_active_set_accepted_stages_success": accepted_stages_success,
        "removal_active_set_probe_evals": int(total_probe_evals),
        "removal_active_set_all_probe_evals_finite": bool(
            all_probe_evals_finite
        ),
        "removal_active_set_full_probe_coverage_complete": bool(
            final_audit["full_probe_coverage_complete"]
        ),
        "removal_active_set_full_probe_scale_count": int(
            final_audit["full_probe_scale_count"]
        ),
        "removal_active_set_final_tested_nodes": int(
            final_audit["tested_nodes"]
        ),
        "removal_active_set_final_testable_nodes": final_testable_nodes,
        "removal_active_set_final_coverage_complete": final_coverage_complete,
        "removal_active_set_final_audit_complete": bool(
            all_probe_evals_finite
            and final_audit["full_probe_coverage_complete"]
            and final_coverage_complete
        ),
        "removal_active_set_final_all_scales_tested": bool(
            final_audit["full_probe_coverage_complete"]
        ),
        "removal_active_set_final_gain_exceedance_count": int(
            final_audit["gain_exceedance_count"]
        ),
        "removal_active_set_final_max_inactive_gain": float(
            final_audit["best_gain"]
        ),
        "removal_active_set_final_best_inactive_gain": float(
            final_audit["best_gain"]
        ),
        "removal_active_set_final_best_inactive_node": int(
            final_audit["best_node"]
        ),
        "removal_active_set_final_best_inactive_step": float(
            final_audit["best_step"]
        ),
        "removal_active_set_no_improving_inactive_nodes": (
            no_improving_inactive_nodes
        ),
        "removal_active_set_kink_tol": float(kink_tol),
        "removal_active_set_min_positive_scales": int(min_positive_scales),
        "removal_active_set_escalate_ambiguous": bool(
            escalate_ambiguous_branches
        ),
        "removal_active_set_ambiguous_branch_activations": ",".join(
            ambiguous_branch_activations
        ),
        "removal_active_set_face_escalation": bool(face_escalation),
        "removal_active_set_face_escalation_used": bool(face_escalation_used),
        "removal_active_set_max_rounds": int(max_rounds),
        "removal_active_set_polish_restarts": int(polish_restarts),
        "removal_active_set_full_domain_upper_bound_min": float(
            np.min(final_upper)
        ),
        "removal_active_set_full_domain_upper_bound_max": float(
            np.max(final_upper)
        ),
        "removal_active_set_full_domain_removal_enabled_nodes": int(
            np.sum(final_upper > 1.0 + kink_tol)
        ),
        "removal_active_set_final_untested_eligible_nodes": int(
            len(eligible_nodes) - len(active_nodes) - final_audit["tested_nodes"]
        ),
        "removal_active_set_initial_utility": float(stage0_u),
        "removal_active_set_final_utility": float(current_u),
        "removal_active_set_utility_gain": float(current_u - stage0_u),
        "removal_active_set_runtime_seconds": float(time.time() - started),
        "removal_stage0_utility": float(stage0_u),
        "removal_stage0_initial_utility": float(stage0_initial_utility),
        "removal_stage0_initial_success": bool(stage0_initial_success),
        "removal_cap_one_initial_utility": float(cap_one_initial_utility),
        "removal_cap_one_initial_success": bool(cap_one_initial_success),
        "removal_stage0_attempt_count": int(1 + stage0_polish_attempts),
        "removal_stage0_failed_attempts": int(sum(
            not item["success"]
            for item in stage_summaries[:1 + stage0_polish_attempts]
        )),
        "removal_stage0_attempt_utilities": ",".join(
            "{:.15g}".format(item["utility"])
            for item in stage_summaries[:1 + stage0_polish_attempts]
        ),
        "removal_stage0_all_mandatory_starts_selected": bool(all(
            mandatory_selection_results[:stage0_polish_attempts]
        )),
        "removal_stage0_lbfgsb_success": bool(
            stage0_diag.get("lbfgsb_success", False)
        ),
        "removal_stage0_final_attempt_runtime_seconds": float(
            stage0_diag.get("runtime_seconds", np.nan)
        ),
        "removal_stage0_runtime_seconds": float(np.nansum([
            item["runtime_seconds"]
            for item in stage_summaries[:1 + stage0_polish_attempts]
        ])),
        "lbfgsb_policy_upper_bound": float(final_cap),
    })
    diagnostics["removal_active_set_all_stages_success"] = diagnostics[
        "removal_active_set_all_solver_attempts_success"
    ]
    diagnostics["_local_results"] = combined_local_results
    diagnostics["lbfgsb_success"] = bool(
        diagnostics.get("lbfgsb_success", False)
        and diagnostics["removal_active_set_pass"]
    )
    diagnostics["success_diagnostics"] = bool(
        diagnostics["lbfgsb_success"]
    )
    return current_m, float(current_u), diagnostics


def solve_lbfgsb_policy(utility, num_nodes, scenario_name, warm_starts=None,
                        fixed_indices=None, fixed_values=None, upper_bounds=None,
                        seed_parts=(), print_progress=False,
                        gradient_mode="finite_difference"):
    removal_requested = env_bool(
        "LBFGSB_REMOVAL_ACTIVE_SET", gradient_mode == "adjoint"
    )
    removal_enabled = removal_active_set_required(gradient_mode)
    if (
        removal_enabled
        and gradient_mode == "adjoint"
        and lbfgsb_policy_upper_bound() > 1.0 + 1e-8
    ):
        return run_lbfgsb_policy_removal_active_set(
            utility,
            num_nodes,
            scenario_name,
            warm_starts=warm_starts,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            upper_bounds=upper_bounds,
            seed_parts=seed_parts,
            print_progress=print_progress,
            gradient_mode=gradient_mode,
        )
    mitigation, utility_value, diagnostics = run_lbfgsb_policy(
        utility,
        num_nodes,
        scenario_name,
        warm_starts=warm_starts,
        fixed_indices=fixed_indices,
        fixed_values=fixed_values,
        upper_bounds=upper_bounds,
        seed_parts=seed_parts,
        print_progress=print_progress,
        gradient_mode=gradient_mode,
    )
    diagnostics.update({
        "removal_active_set_requested": bool(removal_requested),
        "removal_active_set_enabled": bool(removal_enabled),
        "removal_active_set_not_required": bool(
            backstop_smoothing_width() > 0.0
        ),
        "removal_active_set_status": (
            "not_required_smooth_premium"
            if backstop_smoothing_width() > 0.0
            else "skipped_unverified"
        ),
        "removal_active_set_pass": bool(
            backstop_smoothing_width() > 0.0
        ),
    })
    return mitigation, utility_value, diagnostics


def parse_decision_times_env(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def select_diverse_candidates(sorted_population, utilities, top_count, diverse_count,
                              fixed_indices=None, fixed_values=None, upper_bounds=None):
    candidates = []
    seen = set()

    def add(candidate):
        projected = project_initial_point(
            candidate,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            upper_bounds=upper_bounds,
        )
        key = tuple(np.round(projected, 10))
        if key in seen:
            return False
        seen.add(key)
        candidates.append(projected)
        return True

    for candidate in sorted_population[:max(0, int(top_count))]:
        add(candidate)

    ranked = sorted_population[:min(len(sorted_population), max(
        int(top_count) + 10 * max(1, int(diverse_count)), int(top_count)
    ))]
    while len(candidates) < int(top_count) + int(diverse_count) and len(candidates) < len(ranked):
        best_idx = None
        best_distance = -1.0
        for idx, candidate in enumerate(ranked):
            projected = project_initial_point(
                candidate,
                fixed_indices=fixed_indices,
                fixed_values=fixed_values,
                upper_bounds=upper_bounds,
            )
            key = tuple(np.round(projected, 10))
            if key in seen:
                continue
            if candidates:
                distance = min(float(np.linalg.norm(projected - existing)) for existing in candidates)
            else:
                distance = float("inf")
            if distance > best_distance:
                best_distance = distance
                best_idx = idx
        if best_idx is None:
            break
        add(ranked[best_idx])

    return candidates


def ga_adjoint_warm_starts(utility, num_nodes, scenario_name, seed_parts,
                           fixed_indices=None, fixed_values=None, upper_bounds=None):
    pop_amount = env_int("GA_ADJOINT_POP_AMOUNT", env_int("N_GA_POP", 400))
    generations = env_int("GA_ADJOINT_GENERATIONS", env_int("N_GENERATIONS_GA", 200))
    top_count = env_int("GA_ADJOINT_TOP_STARTS", 8)
    diverse_count = env_int("GA_ADJOINT_DIVERSE_STARTS", 8)
    seed = set_solver_seed("{}_ga_adjoint".format(scenario_name), *seed_parts)
    ga_model = GeneticAlgorithm(
        pop_amount=pop_amount,
        num_generations=generations,
        cx_prob=0.8,
        mut_prob=0.50,
        bound=1.5,
        num_feature=num_nodes,
        utility=utility,
        fixed_values=fixed_values,
        fixed_indices=fixed_indices,
        print_progress=True,
        upper_bounds=upper_bounds,
    )
    print(
        "Running GA basin generator for {}; pop {}; generations {}; seed {}".format(
            scenario_name, pop_amount, generations, seed
        )
    )
    final_pop, fitness = ga_model.run()
    order = np.argsort(fitness)[::-1]
    sorted_pop = final_pop[order]
    sorted_fitness = np.asarray(fitness)[order]
    starts = select_diverse_candidates(
        sorted_pop, sorted_fitness, top_count, diverse_count,
        fixed_indices=fixed_indices, fixed_values=fixed_values, upper_bounds=upper_bounds,
    )
    diag = {
        "ga_adjoint_seed": int(seed),
        "ga_adjoint_pop_amount": int(pop_amount),
        "ga_adjoint_generations": int(generations),
        "ga_adjoint_best_utility": float(np.max(fitness)),
        "ga_adjoint_selected_starts": int(len(starts)),
        "ga_adjoint_top_starts": int(top_count),
        "ga_adjoint_diverse_starts": int(diverse_count),
    }
    return starts, diag


def save_lbfgsb_local_optima(diag, scenario_name, sample_id, delay_year, run_type,
                              tree_spec, decision_times_label, out_folder):
    rows = []
    for index, result in enumerate(diag.get("_local_results", []) or []):
        mitigation = np.asarray(result.get("m", []), dtype=float)
        rows.append({
            "sample_index": sample_id,
            "delay_year": delay_year,
            "task_id": os.environ.get("SGE_TASK_ID", "unknown"),
            "run_type": run_type,
            "tree_spec": tree_spec,
            "scenario": scenario_name,
            "decision_times": decision_times_label,
            "optimizer_arm": optimizer_mode(),
            "local_solver": "adjoint_lbfgsb" if adjoint_local_optimizer_mode() else "lbfgsb_multistart",
            "local_start_index": index,
            "start_source": result.get("start_source", ""),
            "removal_active_set_stage": result.get(
                "removal_active_set_stage", ""
            ),
            "initial_utility": float(result.get("start_utility", np.nan)),
            "final_utility": float(result.get("utility", np.nan)),
            "utility_gain_from_polish": float(result.get("utility", np.nan) - result.get("start_utility", np.nan)),
            "success_scipy": bool(result.get("scipy_success", False)),
            "success_diagnostics": bool(result.get("success", False)),
            "guarded_start_kept": bool(result.get("guarded_start_kept", False)),
            "best_eval_retained": bool(result.get("best_eval_retained", False)),
            "best_eval_source": result.get("best_eval_source", ""),
            "best_eval_utility": float(result.get("best_eval_utility", np.nan)),
            "nfev": int(result.get("nfev", 0)),
            "ngev": int(result.get("ngev", 0)),
            "nit": int(result.get("nit", 0)),
            "runtime_seconds": float(result.get("runtime_seconds", np.nan)),
            "message": result.get("message", ""),
            "mitigation": "|".join("{:.12g}".format(float(value)) for value in mitigation),
        })
    if not rows:
        return
    csv_path = os.path.join(DATA_DIR, out_folder, "analysis", f"{out_folder}_local_optima.csv")
    print("Appending local optima to: {}".format(csv_path))
    append_rows_to_csv(rows, csv_path)

def load_external_optimal_warm_start(
        num_nodes, decision_times_label, delay_year, sample_id=None,
        tree_spec=None, expected_backstop_premium=None,
        expected_backstop_smoothing_width=None,
        expected_backstop_smoothing_mode=None, damage_filename=None):
    """Load a saved optimum, with strict provenance checks for exact replay."""

    replay = env_bool("REPLAY_EXTERNAL_OPTIMAL_BASELINE", False)
    csv_path = os.environ.get(
        "EXTERNAL_OPTIMAL_WARM_START_NODE_PRICES", ""
    ).strip()
    if not csv_path:
        if replay:
            raise ValueError(
                "REPLAY_EXTERNAL_OPTIMAL_BASELINE=1 requires "
                "EXTERNAL_OPTIMAL_WARM_START_NODE_PRICES"
            )
        return None

    selected_delay = os.environ.get(
        "EXTERNAL_OPTIMAL_WARM_START_DELAY_YEAR", str(delay_year)
    ).strip()
    selected_sample = os.environ.get(
        "EXTERNAL_OPTIMAL_WARM_START_SAMPLE_INDEX",
        str(sample_id) if sample_id is not None else "",
    ).strip()
    selected_tree_spec = os.environ.get(
        "EXTERNAL_OPTIMAL_WARM_START_TREE_SPEC",
        str(tree_spec) if tree_spec is not None else "",
    ).strip()
    selected_task = os.environ.get(
        "EXTERNAL_OPTIMAL_WARM_START_TASK_ID", ""
    ).strip()
    policy = {}
    matched_rows = []
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scenario") != "optimal":
                continue
            if row.get("delay_year") != selected_delay:
                continue
            if row.get("decision_times") != decision_times_label:
                continue
            if selected_sample and row.get("sample_index") != selected_sample:
                continue
            if selected_tree_spec and row.get("tree_spec") != selected_tree_spec:
                continue
            if selected_task and row.get("task_id") != selected_task:
                continue
            try:
                node = int(row["node"])
                value = float(row["mitigation"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "External optimal warm start has malformed node data"
                ) from exc
            if node < 0 or node >= num_nodes:
                raise ValueError(
                    "External optimal warm start has invalid node {}".format(node)
                )
            if not np.isfinite(value):
                raise ValueError(
                    "External optimal warm start has non-finite mitigation"
                )
            if node in policy:
                if replay:
                    raise ValueError(
                        "Replay matched duplicate node {}; set "
                        "EXTERNAL_OPTIMAL_WARM_START_TASK_ID".format(node)
                    )
                if value != policy[node]:
                    raise ValueError(
                        "Conflicting duplicate mitigation for node {}".format(node)
                    )
            policy[node] = value
            matched_rows.append(row)

    if len(policy) != num_nodes:
        raise ValueError(
            "External optimal warm start from {} matched {} rows / {} unique "
            "nodes for delay_year={} and decision_times={}. Expected {} nodes.".format(
                csv_path, len(matched_rows), len(policy), selected_delay,
                decision_times_label, num_nodes
            )
        )
    mitigation = np.asarray(
        [policy[node] for node in range(num_nodes)], dtype=float
    )
    current_cap = lbfgsb_policy_upper_bound()
    if np.any(mitigation < 0.0) or np.any(mitigation > current_cap):
        raise ValueError(
            "External optimal warm start lies outside current [0, {}] bounds".format(
                current_cap
            )
        )
    if not replay:
        return mitigation
    if len(matched_rows) != num_nodes:
        raise ValueError(
            "Replay must select exactly one row per node; set "
            "EXTERNAL_OPTIMAL_WARM_START_TASK_ID"
        )

    first = matched_rows[0]
    invariant_fields = (
        "baseline_only", "job_id", "task_id", "code_revision",
        "code_worktree_dirty", "code_tracked_diff_sha256",
        "damage_artifact_filename", "damage_artifact_sha256",
        "damage_artifact_size_bytes", "lbfgsb_policy_upper_bound",
        "cost_formulation", "bs_premium", "backstop_smoothing_width",
        "backstop_smoothing_mode", "solver_diagnostics_json",
    )
    for field in invariant_fields:
        if any(row.get(field) != first.get(field) for row in matched_rows):
            raise ValueError("Replay rows disagree on {}".format(field))
    if str(first.get("baseline_only", "")).lower() not in ("true", "1"):
        raise ValueError("Replay artifact is not a baseline-only certificate")
    if first.get("cost_formulation") != cost_formulation_name():
        raise ValueError(
            "Replay artifact has the wrong backstop-premium formulation"
        )
    try:
        source_cap = float(first.get("lbfgsb_policy_upper_bound", "nan"))
        source_premium = float(first.get("bs_premium", "nan"))
        source_smoothing_width = float(
            first.get("backstop_smoothing_width", "0")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Replay artifact has malformed cap or premium") from exc
    if not np.isclose(source_cap, current_cap, rtol=0.0, atol=1e-12):
        raise ValueError("Replay artifact policy cap does not match current cap")
    if expected_backstop_premium is not None and not np.isclose(
        source_premium, float(expected_backstop_premium), rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "Replay artifact removal premium does not match the current model"
        )
    if (
        expected_backstop_smoothing_width is not None
        and not np.isclose(
            source_smoothing_width, float(expected_backstop_smoothing_width),
            rtol=0.0, atol=1e-12
        )
    ):
        raise ValueError(
            "Replay artifact removal-premium smoothing does not match the current model"
        )
    if (
        expected_backstop_smoothing_mode is not None
        and str(first.get("backstop_smoothing_mode", "")).strip().lower()
        != str(expected_backstop_smoothing_mode).strip().lower()
    ):
        raise ValueError(
            "Replay artifact removal-premium smoothing mode does not match the current model"
        )

    code_metadata = current_code_worktree_metadata()
    if first.get("code_revision") != current_code_revision():
        raise ValueError("Replay artifact code revision does not match current code")
    if first.get("code_tracked_diff_sha256") != str(
        code_metadata["code_tracked_diff_sha256"]
    ):
        raise ValueError("Replay artifact tracked-diff fingerprint does not match")
    if str(first.get("code_worktree_dirty", "")).lower() != str(
        code_metadata["code_worktree_dirty"]
    ).lower():
        raise ValueError("Replay artifact dirty-worktree status does not match")
    if damage_filename:
        damage_metadata = damage_artifact_metadata(damage_filename)
        for field in (
            "damage_artifact_filename", "damage_artifact_sha256",
            "damage_artifact_size_bytes",
        ):
            if str(first.get(field)) != str(damage_metadata[field]):
                raise ValueError(
                    "Replay artifact {} does not match".format(field)
                )

    try:
        diagnostics = json.loads(first.get("solver_diagnostics_json", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("Replay artifact has invalid solver diagnostics") from exc
    if backstop_smoothing_width() > 0.0:
        required_true = (
            "optimal_lbfgsb_success",
            "optimal_success_diagnostics",
            "optimal_removal_active_set_pass",
            "optimal_removal_active_set_not_required",
        )
        failed_fields = [
            field for field in required_true if diagnostics.get(field) is not True
        ]
        if failed_fields:
            raise ValueError(
                "Replay artifact lacks a passing smooth-premium certificate: {}".format(
                    ",".join(failed_fields)
                )
            )
        required_values = {
            "optimal_gradient_mode": "adjoint",
            "optimal_gradient_validation_status": "passed",
            "optimal_removal_active_set_status": "not_required_smooth_premium",
        }
        for field, expected in required_values.items():
            if diagnostics.get(field) != expected:
                raise ValueError(
                    "Replay diagnostic {} must equal {}".format(field, expected)
                )
        for field in (
            "configured_lbfgsb_policy_upper_bound",
            "optimal_lbfgsb_policy_upper_bound",
        ):
            try:
                value = float(diagnostics.get(field, np.nan))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Replay diagnostic {} has a malformed cap".format(field)
                ) from exc
            if not np.isclose(value, current_cap, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "Replay diagnostic {} has the wrong cap".format(field)
                )
        return mitigation

    required_true = (
        "configured_lbfgsb_removal_active_set",
        "optimal_lbfgsb_success",
        "optimal_success_diagnostics",
        "optimal_removal_active_set_enabled",
        "optimal_removal_active_set_pass",
        "optimal_removal_active_set_accepted_stages_success",
        "optimal_removal_active_set_all_probe_evals_finite",
        "optimal_removal_active_set_full_probe_coverage_complete",
        "optimal_removal_active_set_final_audit_complete",
        "optimal_removal_active_set_final_coverage_complete",
        "optimal_removal_active_set_final_all_scales_tested",
        "optimal_removal_active_set_no_improving_inactive_nodes",
        "optimal_removal_active_set_all_mandatory_starts_selected",
    )
    failed_fields = [
        field for field in required_true if diagnostics.get(field) is not True
    ]
    if failed_fields:
        raise ValueError(
            "Replay artifact lacks a passing removal certificate: {}".format(
                ",".join(failed_fields)
            )
        )
    required_values = {
        "optimal_gradient_mode": "adjoint",
        "optimal_gradient_validation_status": "passed",
        "optimal_removal_active_set_status": "passed",
    }
    for field, expected in required_values.items():
        if diagnostics.get(field) != expected:
            raise ValueError(
                "Replay diagnostic {} must equal {}".format(field, expected)
            )
    for field in (
        "configured_lbfgsb_policy_upper_bound",
        "optimal_lbfgsb_policy_upper_bound",
        "optimal_removal_active_set_final_cap",
        "optimal_removal_active_set_full_domain_upper_bound_max",
    ):
        try:
            value = float(diagnostics.get(field, np.nan))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Replay diagnostic {} has a malformed cap".format(field)
            ) from exc
        if not np.isclose(value, current_cap, rtol=0.0, atol=1e-12):
            raise ValueError(
                "Replay diagnostic {} has the wrong cap".format(field)
            )
    try:
        max_gain = float(
            diagnostics["optimal_removal_active_set_final_max_inactive_gain"]
        )
        gain_tol = float(diagnostics["optimal_removal_active_set_gain_tol"])
        gain_exceedances = int(
            diagnostics[
                "optimal_removal_active_set_final_gain_exceedance_count"
            ]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Replay artifact has malformed final-gain evidence") from exc
    if (
        not np.isfinite(max_gain)
        or not np.isfinite(gain_tol)
        or gain_tol < 0.0
        or max_gain > gain_tol
        or gain_exceedances != 0
    ):
        raise ValueError("Replay artifact failed the final inactive-gain gate")
    return mitigation


def stable_seed(*parts):
    """Build a deterministic 32-bit seed from run identifiers."""

    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def optimal_baseline_seed_parts(run_type, sample_id, tree_spec):
    """Seed an identical baseline independently of the downstream delay label."""

    return (run_type, sample_id, tree_spec, "optimal_baseline")


def set_solver_seed(label, *parts):
    seed = stable_seed(os.environ.get("RANDOM_SEED_BASE", "20250706"), label, *parts)
    np.random.seed(seed)
    print(f"Random seed for {label}: {seed}")
    return seed


def project_initial_point(point, upper_bound=1.5, fixed_indices=None,
                          fixed_values=None, upper_bounds=None):
    """Project a candidate point into the same constraints used by the solvers."""

    projected = np.asarray(point, dtype=float).copy()
    projected = np.clip(projected, 0.0, upper_bound)
    if upper_bounds is not None:
        projected = np.minimum(projected, np.asarray(upper_bounds, dtype=float))
    if fixed_indices is not None:
        projected[np.asarray(fixed_indices, dtype=int)] = np.asarray(
            fixed_values, dtype=float
        ).flatten()
    return projected


def prepend_warm_starts(sorted_population, warm_starts, topk, upper_bound=1.5,
                        fixed_indices=None, fixed_values=None,
                        upper_bounds=None):
    """Put deterministic warm starts ahead of GA candidates for GradientSearch."""

    candidates = []
    seen = set()

    def add_candidate(candidate):
        projected = project_initial_point(
            candidate,
            upper_bound=upper_bound,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            upper_bounds=upper_bounds,
        )
        key = tuple(np.round(projected, 12))
        if key not in seen:
            seen.add(key)
            candidates.append(projected)

    for warm_start in warm_starts:
        if warm_start is not None:
            add_candidate(warm_start)
    for candidate in sorted_population:
        add_candidate(candidate)
        if len(candidates) >= max(topk, len(warm_starts)) + 8:
            break

    return np.asarray(candidates, dtype=float)


def damage_cache_tag(decision_times, prob_scale=1.0, prefix=''):
    """Suffix damage files by tree structure so cluster tasks cannot collide."""

    dt_tag = '-'.join(str(int(x)) for x in decision_times)
    prob_tag = f"{float(prob_scale):g}".replace('.', 'p')
    return f"{prefix or ''}_EA20260811_DT{dt_tag}_PS{prob_tag}"


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

    price_gap_mapped = price_delay_mapped - price_opt_mapped
    post_delay_mask = np.asarray(common_years) >= reentry_year
    post_delay_m = m_delay_mapped[post_delay_mask]
    post_delay_m = post_delay_m[~np.isnan(post_delay_m)]
    peak_post_delay_mitigation = np.max(post_delay_m) if len(post_delay_m) else np.nan

    post_delay_price_gap = price_gap_mapped[post_delay_mask]
    post_delay_price_gap = post_delay_price_gap[~np.isnan(post_delay_price_gap)]
    avg_first_two_post_delay_price_gap = (
        float(np.mean(post_delay_price_gap[:2]))
        if len(post_delay_price_gap) >= 2 else
        (float(post_delay_price_gap[0]) if len(post_delay_price_gap) else np.nan)
    )
    peak_post_delay_price_gap = (
        float(np.max(post_delay_price_gap)) if len(post_delay_price_gap) else np.nan
    )
    cumulative_post_delay_price_gap = (
        float(np.sum(post_delay_price_gap)) if len(post_delay_price_gap) else np.nan
    )
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
        'avg_first_two_post_delay_price_gap': (
            float(avg_first_two_post_delay_price_gap)
            if not np.isnan(avg_first_two_post_delay_price_gap) else np.nan
        ),
        'peak_post_delay_price_gap': (
            float(peak_post_delay_price_gap)
            if not np.isnan(peak_post_delay_price_gap) else np.nan
        ),
        'cumulative_post_delay_price_gap': (
            float(cumulative_post_delay_price_gap)
            if not np.isnan(cumulative_post_delay_price_gap) else np.nan
        ),
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
    """Append one result without allowing branch-specific fields to shift CSV columns."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    for attempt in range(max_retries):
        try:
            with open(csv_path, "a+", newline="") as f:
                if not acquire_file_lock(f, timeout=lock_timeout):
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
    """Append rows without allowing optional fields to shift existing columns."""

    if not rows:
        return True

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    incoming_fieldnames = []
    for row in rows:
        incoming_fieldnames.extend(
            key for key in row if key not in incoming_fieldnames
        )

    for attempt in range(max_retries):
        try:
            with open(csv_path, "a+", newline="") as f:
                if not acquire_file_lock(f, timeout=lock_timeout):
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
                            key for key in incoming_fieldnames
                            if key not in fieldnames
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

        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"Warning: Failed to write rows to CSV (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"ERROR: Failed to write rows to CSV after {max_retries} attempts: {e}")
                return False

    return False

def _json_default(value):
    """Convert NumPy values in solver diagnostics to JSON-compatible values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def current_code_revision():
    """Return the current Git commit without making benchmark export brittle."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def current_code_worktree_metadata():
    """Fingerprint tracked benchmark-code edits beyond the recorded commit."""

    try:
        diff = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "diff", "--binary", "HEAD", "--"],
            stderr=subprocess.DEVNULL,
        )
        return {
            "code_worktree_dirty": bool(diff),
            "code_tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "code_worktree_dirty": "unknown",
            "code_tracked_diff_sha256": "unknown",
        }


def damage_artifact_metadata(damage_filename):
    """Return the exact saved damage artifact name and content digest."""

    if not damage_filename:
        return {
            "damage_artifact_filename": "",
            "damage_artifact_sha256": "",
            "damage_artifact_size_bytes": 0,
        }

    artifact_filename = (
        damage_filename if str(damage_filename).endswith(".csv")
        else "{}.csv".format(damage_filename)
    )
    artifact_path = (
        artifact_filename if os.path.isabs(artifact_filename)
        else os.path.join(str(PROJECT_ROOT), "data", artifact_filename)
    )
    if not os.path.isfile(artifact_path):
        raise IOError("Damage artifact not found for baseline export: {}".format(artifact_path))

    digest = hashlib.sha256()
    with open(artifact_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "damage_artifact_filename": os.path.basename(artifact_path),
        "damage_artifact_sha256": digest.hexdigest(),
        "damage_artifact_size_bytes": int(os.path.getsize(artifact_path)),
    }


def persist_baseline_outputs(
        sample_id, delay_year, task_id, out_folder, run_type, comparison_type,
        tree_spec, decision_times_label, tree, climate_output, mitigation, utility_value,
        model_params, prob_scale_baseline, output_metadata, solver_diagnostics,
        damage_filename, runtime_seconds):
    """Append one baseline benchmark result and its long-form node prices."""

    mitigation = np.asarray(mitigation, dtype=float).flatten()
    diagnostics = {
        str(key): value for key, value in solver_diagnostics.items()
        if not str(key).startswith("_")
    }
    diagnostics_json = json.dumps(
        diagnostics, sort_keys=True, separators=(",", ":"), default=_json_default
    )
    artifact_metadata = damage_artifact_metadata(damage_filename)
    benchmark_metadata = {
        "baseline_only": True,
        "require_damage_import": require_damage_import_enabled(),
        "job_id": os.environ.get("JOB_ID", "unknown"),
        "random_seed_base": os.environ.get("RANDOM_SEED_BASE", "20250706"),
        "optimizer": optimizer_mode(),
        "baseline_utility": float(utility_value),
        "baseline_runtime_seconds": float(runtime_seconds),
        "code_revision": current_code_revision(),
        "lbfgsb_policy_upper_bound": (
            lbfgsb_policy_upper_bound() if lbfgsb_optimizer_mode() else np.nan
        ),
        "cost_formulation": cost_formulation_name(),
        "backstop_smoothing_width": backstop_smoothing_width(),
        "backstop_smoothing_mode": backstop_smoothing_mode(),
    }
    benchmark_metadata.update(current_code_worktree_metadata())
    benchmark_metadata.update(artifact_metadata)

    results = {
        "sample_index": sample_id,
        "delay_year": delay_year,
        "task_id": task_id,
        "run_type": run_type,
        "comparison_type": comparison_type,
        "tree_spec": tree_spec,
        "decision_times": decision_times_label,
        "decision_times_optimal": decision_times_label,
        "base_year": int(tree.base_year),
        "num_decision_times": int(len(tree.decision_times)),
        "num_decision_nodes": int(tree.num_decision_nodes),
        "prob_scale_baseline": float(prob_scale_baseline),
        "utility": float(utility_value),
        "u_optimal": float(utility_value),
        "mitigation": "|".join(
            "{:.17g}".format(float(value)) for value in mitigation
        ),
        "mitigation_num_nodes": int(len(mitigation)),
    }
    results.update(model_params)
    results.update(benchmark_metadata)
    results.update(output_metadata or {})
    results["solver_diagnostics_json"] = diagnostics_json
    results.update(diagnostics)

    analysis_dir = os.path.join(DATA_DIR, out_folder, "analysis")
    results_path = os.path.join(
        analysis_dir, "{}_baseline_results.csv".format(out_folder)
    )
    print("Appending baseline benchmark result to: {}".format(results_path))
    results_ok = append_results_to_csv(results, results_path)

    node_metadata = dict(output_metadata or {})
    node_metadata.update(benchmark_metadata)
    node_metadata.update({
        "solver_diagnostics_json": diagnostics_json,
        "prob_scale_baseline": float(prob_scale_baseline),
    })
    node_rows = build_node_price_rows(
        sample_id, delay_year, task_id, "optimal", tree, climate_output,
        mitigation, run_type, tree_spec, decision_times_label, model_params,
        comparison_type, output_metadata=node_metadata,
    )
    node_prices_path = os.path.join(
        analysis_dir, "{}_baseline_node_prices.csv".format(out_folder)
    )
    print("Appending baseline node prices to: {}".format(node_prices_path))
    nodes_ok = append_rows_to_csv(node_rows, node_prices_path)

    if not results_ok or not nodes_ok:
        raise IOError(
            "Baseline-only output persistence failed: results_ok={}, nodes_ok={}".format(
                results_ok, nodes_ok
            )
        )
    print("Baseline-only benchmark outputs persisted successfully.")


def build_node_price_rows(sample_id, delay_year, task_id, scenario, tree, climate_output,
                          mitigation, run_type, tree_spec, decision_times_label,
                          params, comparison_type, output_metadata=None):
    """Build long-form node metric output for notebook posterior analysis."""

    metadata = dict(output_metadata or {})
    utility = climate_output.utility
    utility_trees = utility.utility(mitigation, return_trees=True)
    continuation_utility_tree = utility_trees['Utility']
    consumption_tree = utility_trees['Consumption']
    certain_equivalence_tree = utility_trees['CertainEquivalence']
    rows = []
    for node in range(tree.num_decision_nodes):
        period = tree.get_period(node)
        state = tree.get_state(node, period)
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
            'consumption': float(consumption_tree.tree[tree.decision_times[period]][state]),
            # Conditional continuation utility at this information set, and
            # its derivative with respect to consumption at this node.  This
            # is the local Epstein-Zin marginal utility, not a root-date
            # state price (which would additionally include path probabilities
            # and all upstream continuation derivatives).
            'utility': float(
                continuation_utility_tree.tree[tree.decision_times[period]][state]
            ),
            'marginal_utility': float(
                utility._mu_0(
                    consumption_tree.tree[tree.decision_times[period]][state],
                    certain_equivalence_tree.tree[tree.decision_times[period]][state],
                )
            ),
            'damage': float(utility.damage._damage_function_node(mitigation, node)),
            'climate_damage': float(utility.damage.climate_damage_node(mitigation, node)),
            'ra': float(params['ra']),
            'eis': float(params['eis']),
            'pref': float(params['pref']),
            'tech_chg': float(params['tech_chg']),
            'tech_scale': float(params['tech_scale']),
            'bs_premium': float(params['bs_premium']),
            'backstop_smoothing_width': float(
                params.get('backstop_smoothing_width', 0.0)
            ),
            'backstop_smoothing_mode': str(
                params.get('backstop_smoothing_mode', 'one_sided_huber')
            ),
            'cost_formulation': str(
                params.get('cost_formulation', 'additive_removal_premium_v1')
            ),
            'growth': float(params['growth']),
        }
        row.update(metadata)
        rows.append(row)
    return rows


class ZeroDamage(BPWDamage):
    """Damage object for Shapley coalitions with climate damages switched off."""

    zero_damage = True

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
    return os.path.join(
        DATA_DIR,
        "Gaussian_samples_N{}_DIMS{}_{}_{}_seed{}_ensemble_delayed.csv".format(
            N_SAMPLES, DIMS, GAUSSIAN_PRIOR_SET_NAME, gaussian_support_tag(),
            gaussian_sample_seed()
        ),
    )


def generate_gaussian_ensemble_samples():
    samp_fname = get_sample_filename()
    means = gaussian_preference_means()
    stds = PARAMETER_PRIOR_STDS
    support_ubs = gaussian_support_upper_bounds()
    
    print(f"\nGenerating {N_SAMPLES} bounded Gaussian samples...")
    print(f"Parameter space dimension: {DIMS}")
    print(f"Parameter support, Gaussian mode/main-spec value, and standard deviation:")
    for i, name in enumerate(param_names):
        print(f"  {name}: [{lbs[i]}, {support_ubs[i]}], mode={means[i]}, std={stds[i]}")
    
    generate_gaussian_samples(N_SAMPLES, DIMS, lbs, support_ubs, means=means, stds=stds,
                              save_file=True, filename=samp_fname,
                              random_seed=gaussian_sample_seed())
    
    print(f"Samples saved to: {samp_fname}\n")


def load_or_generate_gaussian_samples():
    samp_fname = get_sample_filename()
    lock_fname = samp_fname + ".lock"
    with open(lock_fname, "a+") as lock_file:
        if not acquire_file_lock(lock_file, timeout=600.0):
            raise TimeoutError(
                "Timed out waiting for Gaussian sample lock: {}".format(lock_fname)
            )
        try:
            if not os.path.exists(samp_fname):
                print(f"\nSample file not found, generating new samples...")
                generate_gaussian_ensemble_samples()
            else:
                print(f"\nSample file found: {samp_fname}")
            param_vals = np.atleast_2d(np.loadtxt(samp_fname, delimiter=","))
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    support_ubs = gaussian_support_upper_bounds()
    if param_vals.shape != (N_SAMPLES, DIMS):
        raise ValueError(
            "Gaussian sample file has shape {}, expected ({}, {})".format(
                param_vals.shape, N_SAMPLES, DIMS
            )
        )
    if not np.all(np.isfinite(param_vals)):
        raise ValueError("Gaussian sample file contains non-finite values")
    if (np.any(param_vals < np.asarray(lbs, dtype=float))
            or np.any(param_vals > support_ubs)):
        raise ValueError("Gaussian sample file contains values outside configured support")
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
                                  comparison_type='fixed_learning_grid',
                                  decision_times_baseline=None,
                                  decision_times_delay=None,
                                  prob_scale_baseline=1.0,
                                  prob_scale_delay=1.0,
                                  sample_label=None,
                                  common_years=None,
                                  delay_periods=None,
                                  period_len=FIXED_DELAY_PERIOD_LEN,
                                  emissions_time_step=FIXED_DELAY_EMISSIONS_TIME_STEP,
                                  damage_file_tag=FIXED_DELAY_DAMAGE_FILE_TAG,
                                  output_metadata=None,
                                  zero_climate_damages=False,
                                  mac_horizontal_shift=0.0,
                                  mac_vertical_shift=0.0,
                                  delay_window_years=None):

    validate_damage_import_configuration(import_damages)

    ra, eis, tech_chg, tech_scale, pref, bs_premium, growth = param_vals[sample_index]
    sample_id = sample_label if sample_label is not None else sample_index
    smoothing_width = backstop_smoothing_width()
    smoothing_mode = backstop_smoothing_mode()
    mac_horizontal_shift = float(mac_horizontal_shift)
    mac_vertical_shift = float(mac_vertical_shift)
    if not np.isfinite(mac_horizontal_shift) or mac_horizontal_shift < 0.0:
        raise ValueError("mac_horizontal_shift must be finite and nonnegative")
    if not np.isfinite(mac_vertical_shift) or mac_vertical_shift < 0.0:
        raise ValueError("mac_vertical_shift must be finite and nonnegative")
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
        'backstop_smoothing_width': smoothing_width,
        'backstop_smoothing_mode': smoothing_mode,
        'cost_formulation': cost_formulation_name(),
        'mac_horizontal_shift': mac_horizontal_shift,
        'mac_vertical_shift': mac_vertical_shift,
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
        N_generations_ga = env_int("N_GENERATIONS_GA", 200)
        N_iters_gs = env_int("N_ITERS_GS", 150)
    N_topk_gs = env_int("N_TOPK_GS", 8)
    print(
        "Optimization settings: "
        f"GA generations={N_generations_ga}, "
        f"gradient iterations={N_iters_gs}, "
        f"gradient topk={N_topk_gs}"
    )
    
    print("\nInitializing model components...")
    
    if decision_times_delay is None:
        decision_times_delay = fixed_delay_decision_times()
    else:
        decision_times_delay = list(decision_times_delay)

    if decision_times_baseline is None:
        # Same-grid comparison: the unconstrained comparator and delayed run
        # use identical decision times, isolating the mitigation constraint.
        decision_times_baseline = decision_times_delay.copy()
    else:
        decision_times_baseline = list(decision_times_baseline)

    if delay_periods is None:
        delay_periods = get_delay_periods_for_year(decision_times_delay, delay_year)

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
                backstop_premium=bs_premium, no_free_lunch=no_free_lunch,
                backstop_smoothing_width=smoothing_width,
                backstop_smoothing_mode=smoothing_mode,
                mac_horizontal_shift=mac_horizontal_shift,
                mac_vertical_shift=mac_vertical_shift)
    
    emit_at_0_delay = np.interp(2030, baseline_emission_model_delay.times,
                          baseline_emission_model_delay.baseline_gtco2)
    c_delay = BPWCost(t_delay, emit_at_0=emit_at_0_delay,
                baseline_num=baseline, tech_const=tech_chg,
                tech_scale=tech_scale, cons_at_0=86252.0, # 2025 estimatedfrom https://data.worldbank.org/indicator/NE.CON.TOTL.CD
                backstop_premium=bs_premium, no_free_lunch=no_free_lunch,
                backstop_smoothing_width=smoothing_width,
                backstop_smoothing_mode=smoothing_mode,
                mac_horizontal_shift=mac_horizontal_shift,
                mac_vertical_shift=mac_vertical_shift)
    
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
            raise_required_damage_import_failure(
                e,
                "baseline artifact {!r} and delayed artifact {!r}".format(
                    damsim_filename_baseline, damsim_filename_delay
                ),
            )
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

    def build_utility_for_decision_times(decision_times, tag_suffix="coarse"):
        coarse_tree = TreeModel(decision_times=list(decision_times), prob_scale=prob_scale_baseline)
        coarse_emission = BPWEmissionBaseline(
            tree=coarse_tree,
            baseline_num=baseline,
            emissions_time_step=emissions_time_step,
        )
        coarse_emission.baseline_emission_setup()
        coarse_climate = BPWClimate(coarse_tree, coarse_emission, draws=draws)
        coarse_emit_at_0 = np.interp(
            2030, coarse_emission.times, coarse_emission.baseline_gtco2
        )
        coarse_cost = BPWCost(
            coarse_tree,
            emit_at_0=coarse_emit_at_0,
            baseline_num=baseline,
            tech_const=tech_chg,
            tech_scale=tech_scale,
            cons_at_0=86252.0,
            backstop_premium=bs_premium,
            no_free_lunch=no_free_lunch,
            backstop_smoothing_width=smoothing_width,
            backstop_smoothing_mode=smoothing_mode,
            mac_horizontal_shift=mac_horizontal_shift,
            mac_vertical_shift=mac_vertical_shift,
        )
        coarse_damage = damage_class(
            tree=coarse_tree,
            emit_baseline=coarse_emission,
            climate=coarse_climate,
            mitigation_constants=mitigation_constants,
            draws=draws,
        )
        coarse_tag = damage_cache_tag(
            coarse_tree.decision_times, prob_scale_baseline, damage_file_tag
        )
        coarse_filename = ''.join([
            "simulated_damages_df", str(dam_func),
            "_TP", str(tip_on), "_SSP", str(baseline),
            "_BY", str(coarse_tree.base_year),
            "_dunc", str(d_unc), "_tunc", str(t_unc),
            coarse_tag,
        ])
        print("Coarse-to-fine damage simulation ({}): {}".format(tag_suffix, coarse_filename))
        if zero_climate_damages:
            coarse_damage.damage_simulation(
                filename=coarse_filename, save_simulation=False,
                dam_func=dam_func, tip_on=tip_on, d_unc=d_unc, t_unc=t_unc,
            )
        elif import_damages:
            try:
                coarse_damage.import_damages(file_name=coarse_filename)
            except Exception as exc:
                raise_required_damage_import_failure(
                    exc, "coarse artifact {!r}".format(coarse_filename)
                )
                print("Warning: could not import coarse damages ({}); simulating".format(exc))
                coarse_damage.damage_simulation(
                    filename=coarse_filename, save_simulation=True,
                    dam_func=dam_func, tip_on=tip_on, d_unc=d_unc, t_unc=t_unc,
                )
        else:
            coarse_damage.damage_simulation(
                filename=coarse_filename, save_simulation=True,
                dam_func=dam_func, tip_on=tip_on, d_unc=d_unc, t_unc=t_unc,
            )
        coarse_utility = EZUtility(
            tree=coarse_tree, damage=coarse_damage, cost=coarse_cost,
            period_len=period_len, eis=eis, ra=ra, time_pref=pref,
            cons_growth=growth,
        )
        return coarse_tree, coarse_utility

    def coarse_to_fine_warm_starts(target_tree, target_utility, scenario_name,
                                   seed_parts, fixed_indices=None, fixed_values=None,
                                   upper_bounds=None):
        coarse_times = parse_decision_times_env(
            "COARSE_DECISION_TIMES", [0, 5, 10, 15, 35, 125, 225]
        )
        coarse_tree, coarse_utility = build_utility_for_decision_times(
            coarse_times, tag_suffix=scenario_name
        )
        coarse_fixed_indices = None
        coarse_fixed_values = None
        if fixed_indices is not None:
            coarse_delay_periods = get_delay_periods_for_year(coarse_times, delay_year)
            if coarse_delay_periods > 0:
                coarse_fixed_indices = get_delay_nodes(coarse_tree, coarse_delay_periods)
                coarse_fixed_values = np.zeros(len(coarse_fixed_indices))
        coarse_m, coarse_u, coarse_diag = run_lbfgsb_policy(
            coarse_utility,
            coarse_tree.num_decision_nodes,
            "{}_coarse".format(scenario_name),
            fixed_indices=coarse_fixed_indices,
            fixed_values=coarse_fixed_values,
            seed_parts=("coarse",) + tuple(seed_parts),
            print_progress=True,
            gradient_mode="adjoint",
        )
        save_lbfgsb_local_optima(
            coarse_diag, "{}_coarse".format(scenario_name), sample_id, delay_year,
            run_type, tree_spec, '|'.join(str(int(x)) for x in coarse_times), out_folder
        )
        require_lbfgsb_success(coarse_diag, "{}_coarse".format(scenario_name))
        full_start = prolong_policy_nearest_ancestor(coarse_m, coarse_tree, target_tree)
        full_start = project_initial_point(
            full_start, fixed_indices=fixed_indices, fixed_values=fixed_values,
            upper_bounds=upper_bounds,
        )
        diag = {
            "coarse_to_fine_decision_times": '|'.join(str(int(x)) for x in coarse_times),
            "coarse_to_fine_num_nodes": int(coarse_tree.num_decision_nodes),
            "coarse_to_fine_utility": float(coarse_u),
            "coarse_to_fine_full_start_utility": float(np.asarray(target_utility.utility(full_start)).reshape(-1)[0]),
        }
        diag.update(prefixed_diagnostics("coarse", coarse_diag))
        return [full_start], diag

    solver_diagnostics = {
        "n_generations_ga": int(N_generations_ga),
        "n_iters_gs": int(N_iters_gs),
        "n_topk_gs": int(N_topk_gs),
        "shared_optimal_cache_enabled": False,
        "optimizer": optimizer_mode(),
    }
    solver_diagnostics.update(configured_optimizer_diagnostics(
        N_generations_ga, N_iters_gs, N_topk_gs
    ))

    print("\nPREPARING INDEPENDENT OPTIMAL AND DELAYED SCENARIOS\n")

    print("Shared optimal cache is disabled permanently; this task computes its own optimal baseline.")

    def compute_optimal_scenario():
        print("Computing unconstrained optimum.")
        if lbfgsb_optimizer_mode():
            gradient_mode = "adjoint" if adjoint_local_optimizer_mode() else "finite_difference"
            optimal_warm_starts = []
            optimal_seed_parts = optimal_baseline_seed_parts(
                run_type, sample_id, tree_spec
            )
            external_warm_start = load_external_optimal_warm_start(
                t_baseline.num_decision_nodes, decision_times_label, delay_year,
                sample_id=sample_id, tree_spec=tree_spec,
                expected_backstop_premium=bs_premium,
                expected_backstop_smoothing_width=smoothing_width,
                expected_backstop_smoothing_mode=smoothing_mode,
                damage_filename=(
                    None if zero_climate_damages else damsim_filename_baseline
                ),
            )
            if external_warm_start is not None:
                external_utility = float(np.asarray(u_baseline.utility(external_warm_start)).reshape(-1)[0])
                print(
                    "External optimal warm start utility under current objective: {:.12f}".format(
                        external_utility
                    )
                )
                print(
                    "External optimal warm start mitigation range: [{:.6f}, {:.6f}]".format(
                        float(np.min(external_warm_start)),
                        float(np.max(external_warm_start)),
                    )
                )
                optimal_warm_starts.append(external_warm_start)
            if external_warm_start is not None and env_bool("REPLAY_EXTERNAL_OPTIMAL_BASELINE", False):
                print("Replaying the supplied optimal baseline without re-optimizing it.")
                solver_diagnostics.update({
                    "optimal_external_baseline_replayed": True,
                    "optimal_external_baseline_utility": external_utility,
                })
                co_opt_local = ClimateOutput(u_baseline)
                co_opt_local.calculate_output(external_warm_start)
                return external_warm_start, external_utility, co_opt_local
            if optimizer_mode() == "ga_adjoint_lbfgsb":
                ga_starts, ga_diag = ga_adjoint_warm_starts(
                    u_baseline, t_baseline.num_decision_nodes, "optimal",
                    optimal_seed_parts,
                )
                optimal_warm_starts.extend(ga_starts)
                solver_diagnostics.update(prefixed_diagnostics("optimal", ga_diag))
            if optimizer_mode() == "coarse_to_fine_adjoint_lbfgsb":
                coarse_starts, coarse_diag = coarse_to_fine_warm_starts(
                    t_baseline, u_baseline, "optimal",
                    optimal_seed_parts,
                )
                optimal_warm_starts.extend(coarse_starts)
                solver_diagnostics.update(prefixed_diagnostics("optimal", coarse_diag))
            m_lbfgsb, u_lbfgsb, diag_lbfgsb = solve_lbfgsb_policy(
                u_baseline,
                t_baseline.num_decision_nodes,
                "optimal",
                warm_starts=optimal_warm_starts,
                seed_parts=optimal_seed_parts,
                print_progress=True,
                gradient_mode=gradient_mode,
            )
            save_lbfgsb_local_optima(
                diag_lbfgsb, "optimal", sample_id, delay_year, run_type,
                tree_spec, decision_times_label, out_folder
            )
            solver_diagnostics.update(prefixed_diagnostics("optimal", diag_lbfgsb))
            require_lbfgsb_success(diag_lbfgsb, "optimal")
            co_opt_local = ClimateOutput(u_baseline)
            co_opt_local.calculate_output(m_lbfgsb)
            return m_lbfgsb, float(u_lbfgsb), co_opt_local
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
        solver_diagnostics["seed_optimal"] = set_solver_seed(
            "optimal", run_type, sample_id, delay_year, tree_spec
        )
        final_pop_opt, fitness_opt = ga_model_opt.run()
        sort_pop_opt = final_pop_opt[np.argsort(fitness_opt)][::-1]
        solver_diagnostics["ga_best_optimal"] = float(np.max(fitness_opt))

        print("Running Gradient Search (optimal)...")
        topk_opt = min(N_topk_gs, len(sort_pop_opt))
        m_optimal_local, u_optimal_local = gs_model_opt.run(
            initial_point_list=sort_pop_opt, topk=topk_opt
        )
        solver_diagnostics["gs_topk_optimal"] = int(topk_opt)
        solver_diagnostics["gs_final_optimal"] = float(u_optimal_local)

        co_opt_local = ClimateOutput(u_baseline)
        co_opt_local.calculate_output(m_optimal_local)
        return m_optimal_local, float(u_optimal_local), co_opt_local

    def run_delayed_scenario(m_optimal_warm=None):
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

        if lbfgsb_optimizer_mode():
            gradient_mode = "adjoint" if adjoint_local_optimizer_mode() else "finite_difference"
            warm_starts_delay = []
            if m_optimal_warm is not None and len(m_optimal_warm) == t_delay.num_decision_nodes:
                warm_starts_delay.append(m_optimal_warm)
            if optimizer_mode() == "ga_adjoint_lbfgsb":
                ga_starts, ga_diag = ga_adjoint_warm_starts(
                    u_delay, t_delay.num_decision_nodes, "delayed",
                    (run_type, sample_id, delay_year, tree_spec),
                    fixed_indices=fixed_indices_delay, fixed_values=fixed_values_delay,
                )
                warm_starts_delay.extend(ga_starts)
                solver_diagnostics.update(prefixed_diagnostics("delayed", ga_diag))
            if optimizer_mode() == "coarse_to_fine_adjoint_lbfgsb":
                coarse_starts, coarse_diag = coarse_to_fine_warm_starts(
                    t_delay, u_delay, "delayed",
                    (run_type, sample_id, delay_year, tree_spec),
                    fixed_indices=fixed_indices_delay, fixed_values=fixed_values_delay,
                )
                warm_starts_delay.extend(coarse_starts)
                solver_diagnostics.update(prefixed_diagnostics("delayed", coarse_diag))
            m_lbfgsb, u_lbfgsb, diag_lbfgsb = solve_lbfgsb_policy(
                u_delay,
                t_delay.num_decision_nodes,
                "delayed",
                warm_starts=warm_starts_delay,
                fixed_indices=fixed_indices_delay,
                fixed_values=fixed_values_delay,
                seed_parts=(run_type, sample_id, delay_year, tree_spec),
                print_progress=True,
                gradient_mode=gradient_mode,
            )
            save_lbfgsb_local_optima(
                diag_lbfgsb, "delayed", sample_id, delay_year, run_type,
                tree_spec, delay_decision_times_label, out_folder
            )
            solver_diagnostics.update(prefixed_diagnostics("delayed", diag_lbfgsb))
            require_lbfgsb_success(diag_lbfgsb, "delayed")
            co_delay_local = ClimateOutput(u_delay)
            co_delay_local.calculate_output(m_lbfgsb)
            return m_lbfgsb, float(u_lbfgsb), co_delay_local

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
        solver_diagnostics["seed_delayed"] = set_solver_seed(
            "delayed", run_type, sample_id, delay_year, tree_spec
        )
        final_pop_delay, fitness_delay = ga_model_delay.run()
        sort_pop_delay = final_pop_delay[np.argsort(fitness_delay)][::-1]
        solver_diagnostics["ga_best_delayed"] = float(np.max(fitness_delay))

        warm_starts_delay = []
        if m_optimal_warm is not None and len(m_optimal_warm) == t_delay.num_decision_nodes:
            warm_starts_delay.append(m_optimal_warm)
        gs_initial_delay = prepend_warm_starts(
            sort_pop_delay, warm_starts_delay, N_topk_gs,
            fixed_indices=fixed_indices_delay,
            fixed_values=fixed_values_delay,
        )

        print("Running Gradient Search (delayed)...")
        topk_delay = min(N_topk_gs, len(gs_initial_delay))
        m_delayed_local, u_delayed_local = gs_model_delay.run(
            initial_point_list=gs_initial_delay, topk=topk_delay
        )
        solver_diagnostics["gs_topk_delayed"] = int(topk_delay)
        solver_diagnostics["gs_final_delayed"] = float(u_delayed_local)

        if fixed_indices_delay is not None:
            for idx in fixed_indices_delay:
                if abs(m_delayed_local[idx]) > 1e-10:
                    print(f"Warning: Constrained node {idx} not zero: m_delayed[{idx}]={m_delayed_local[idx]:.10f}")

        print(f"\nDelayed scenario complete:")
        print(f"  First-period mitigation:  {m_delayed_local[0]:.6f} (constrained to 0)")
        print(f"  Carbon price:             ${c_delay.price(0, m_delayed_local[0], 0):.2f} per ton")
        print(f"  Utility:                  {u_delayed_local:.10f}\n")

        co_delay_local = ClimateOutput(u_delay)
        co_delay_local.calculate_output(m_delayed_local)
        return m_delayed_local, u_delayed_local, co_delay_local

    baseline_only = env_bool("BASELINE_ONLY", False)
    if baseline_only:
        print("Shared optimal cache disabled; running the baseline-only optimum in this task.")
    else:
        print("Shared optimal cache disabled; running optimal first and delayed second in this task.")
    m_delayed = u_delayed = co_delay = None
    baseline_started_at = time.monotonic() if baseline_only else None
    m_optimal, u_optimal, co_opt = compute_optimal_scenario()

    if baseline_only:
        baseline_runtime_seconds = time.monotonic() - baseline_started_at
        print(
            "BASELINE_ONLY=1; persisting the unconstrained baseline and "
            "skipping all delayed optimization and comparison analysis."
        )
        persist_baseline_outputs(
            sample_id=sample_id,
            delay_year=delay_year,
            task_id=task_id,
            out_folder=out_folder,
            run_type=run_type,
            comparison_type=comparison_type,
            tree_spec=tree_spec,
            decision_times_label=decision_times_label,
            tree=t_baseline,
            climate_output=co_opt,
            mitigation=m_optimal,
            utility_value=u_optimal,
            model_params=model_params,
            prob_scale_baseline=prob_scale_baseline,
            output_metadata=output_metadata,
            solver_diagnostics=solver_diagnostics,
            damage_filename=(None if zero_climate_damages else damsim_filename_baseline),
            runtime_seconds=baseline_runtime_seconds,
        )
        return

    if m_delayed is None:
        if (
            delay_periods == 0
            and list(decision_times_delay) == list(decision_times_baseline)
            and float(prob_scale_delay) == float(prob_scale_baseline)
        ):
            print("No delay constraints; reusing optimal solution for delayed scenario.")
            m_delayed = np.asarray(m_optimal, dtype=float).copy()
            u_delayed = float(u_optimal)
            co_delay = ClimateOutput(u_delay)
            co_delay.calculate_output(m_delayed)
        else:
            m_delayed, u_delayed, co_delay = run_delayed_scenario(m_optimal)

    solver_diagnostics["constrained_beats_optimal_initial"] = bool(
        float(u_delayed) > float(u_optimal) + 1e-8
    )
    if solver_diagnostics["constrained_beats_optimal_initial"] and len(m_delayed) == len(m_optimal):
        print("Delayed solution exceeds comparator utility; refining unconstrained comparator locally.")
        gs_repair_opt = GradientSearch(
            var_nums=t_baseline.num_decision_nodes,
            utility=u_baseline,
            accuracy=5.e-7,
            iterations=N_iters_gs,
            print_progress=True
        )
        repair_candidates = prepend_warm_starts(
            np.asarray([m_optimal]), [m_delayed], N_topk_gs
        )
        m_repaired, u_repaired = gs_repair_opt.run(
            initial_point_list=repair_candidates,
            topk=min(N_topk_gs, len(repair_candidates))
        )
        if float(u_repaired) > float(u_optimal):
            m_optimal = np.asarray(m_repaired, dtype=float)
            u_optimal = float(u_repaired)
            co_opt = ClimateOutput(u_baseline)
            co_opt.calculate_output(m_optimal)
            solver_diagnostics["optimal_repaired"] = True
            solver_diagnostics["gs_final_optimal_repaired"] = float(u_optimal)
        else:
            solver_diagnostics["optimal_repaired"] = False
    else:
        solver_diagnostics["optimal_repaired"] = False
    solver_diagnostics["constrained_beats_optimal_final"] = bool(
        float(u_delayed) > float(u_optimal) + 1e-8
    )
    for key in (
        "seed_delayed", "ga_best_delayed", "gs_topk_delayed",
        "gs_final_delayed", "gs_final_optimal_repaired"
    ):
        solver_diagnostics.setdefault(key, np.nan)

    print("\nCONSTRAINT ANALYSIS (COMPARING OPTIMAL VS DELAYED)\n")
    
    ca = ConstraintAnalysis(
        u_delay, u_baseline, m_delayed, m_optimal,
        delay_window_years=delay_window_years,
    )

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
    if ca.delay_window_dwl_pct is not None:
        print(
            f"  Delay-window proportional DWL ({ca.delay_window_years:g} years): "
            f"{format_optional(ca.delay_window_dwl_pct, '.4f', suffix='%')}"
        )
    print(f"  First-period compensation, robustness (abs): {format_optional(ca.delta_c, '.6f')}")
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
        'backstop_smoothing_width': float(smoothing_width),
        'backstop_smoothing_mode': smoothing_mode,
        'cost_formulation': cost_formulation_name(),
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
        
        # Primary delay-window proportional DWL (delay-frontier runs only)
        'delay_window_years': (
            float(ca.delay_window_years)
            if ca.delay_window_years is not None else np.nan
        ),
        'delay_window_dwl_fraction': (
            float(ca.delay_window_dwl_fraction)
            if ca.delay_window_dwl_fraction is not None else np.nan
        ),
        'delay_window_dwl_pct': (
            float(ca.delay_window_dwl_pct)
            if ca.delay_window_dwl_pct is not None else np.nan
        ),

        # Legacy first-period additive compensation (robustness)
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
    results_dict.update(solver_diagnostics)
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
        'backstop_smoothing_width': float(smoothing_width),
        'backstop_smoothing_mode': smoothing_mode,
        'cost_formulation': cost_formulation_name(),
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
        samples_copy = os.path.join(
            DATA_DIR, out_folder, "samples",
            "Gaussian_samples_N{}_DIMS{}_{}_{}_seed{}.csv".format(
                N_SAMPLES, DIMS, GAUSSIAN_PRIOR_SET_NAME, gaussian_support_tag(),
                gaussian_sample_seed()
            ),
        )
        np.savetxt(samples_copy, param_vals, delimiter=",")
        print(f"\nSaved copy of samples to: {samples_copy}")
    

    print(f"RUNNING: Sample {sample_index}/{len(param_vals)-1}")
    print(f"DELAY YEAR: {delay_year}")
    print(f"TASK: {task_id}\n")
    
    print(f"\nExecution Configuration:")
    print(f"  Test mode:       {test_mode}")
    print(f"  Import damages:  {import_damages}")
    print(f"  Require import:  {require_damage_import_enabled()}")
    print(f"  Baseline (SSP):  {baseline}")
    
    try:
        run_ensemble_delayed_analysis(
            sample_index, delay_year, param_vals,
            out_folder, baseline, test_mode, import_damages,
            output_metadata=gaussian_support_metadata(),
            delay_window_years=delay_year,
        )
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

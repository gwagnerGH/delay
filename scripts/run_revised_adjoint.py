#!/usr/bin/env python3
"""Solve the revised EZDelay model with its production adjoint L-BFGS-B.

Damage simulation is an explicit input and excluded from optimizer timings.
The finite upper bound of four is a numerical guard for the saved damage grid,
not a proposed economic cap on carbon removal.
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimization import CandidateScreenedLBFGSB  # noqa: E402
from src.config.revised_baseline import BASELINE  # noqa: E402
from src.config.terminal_growth import (add_terminal_arguments, terminal_settings_from_args)  # noqa: E402
from src.revised_experiment import build_saved_revised_model  # noqa: E402
from scripts.export_revised_state import export_state_trees  # noqa: E402


TREES = {
    "short": [0, 10, 35, 75, 125, 175, 225],
    "full": [0, 5, 10, 15, 35, 75, 125, 175, 225],
    "front_loaded": [0, 5, 10, 15, 50, 100, 160, 225],
    "back_loaded": [0, 5, 10, 15, 50, 100, 150, 200, 225],
}
DEFAULT_INPUTS = ROOT / "data" / "revised_default"
DEFAULT_OUTPUTS = ROOT / "data" / "new_outputs"


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()
                if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def feasible_foc_gap(gaps, mitigation, upper, fixed=()):
    gaps = np.asarray(gaps, float)
    mitigation = np.asarray(mitigation, float)
    upper = np.asarray(upper, float)
    violations = np.where(mitigation <= 1e-8, np.maximum(gaps, 0.0),
        np.where(mitigation >= upper-1e-8, np.maximum(-gaps, 0.0),
                 np.abs(gaps)))
    violations[np.asarray(fixed, dtype=int)] = 0.0
    return float(np.max(violations))


def audited_finite_difference(objective, policy, node, adjoint_gradient, upper,
                              tolerance=1e-7, allow_probe=True):
    """Check one adjoint component with a step chosen without its value.

    A fixed central step can lose precision for large welfare levels and
    truncate near a steep smooth backstop. Adjacent five-point estimates must
    agree with each other independently of the adjoint.
    """
    base_utility = float(objective.utility(policy))
    machine_eps = np.finfo(float).eps

    def roundoff_bound(step):
        return 10.0*machine_eps*max(abs(base_utility), 1.0)/step

    def utility_at(displacement):
        trial = policy.copy()
        trial[node] += displacement
        return float(objective.utility(trial))

    central_step = 2e-5
    central = (utility_at(central_step)-utility_at(-central_step))/(2*central_step)
    central_error = abs(adjoint_gradient-central)
    if central_error < tolerance and roundoff_bound(central_step) < tolerance:
        return central, central_step, "central", 0.0, adjoint_gradient, policy[node]

    estimates = []
    for step in (2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3):
        if (policy[node]-2*step <= 0.0 or
                policy[node]+2*step >= upper[node] or
                roundoff_bound(step) >= tolerance):
            continue
        try:
            plus_two = utility_at(2*step)
            plus_one = utility_at(step)
            minus_one = utility_at(-step)
            minus_two = utility_at(-2*step)
            derivative = (-plus_two+8*plus_one-8*minus_one+minus_two)/(12*step)
        except (FloatingPointError, ValueError):
            continue
        if np.isfinite(derivative):
            estimates.append((step, derivative))
    if len(estimates) < 2:
        if allow_probe and policy[node] < 0.005 and upper[node] > 0.02:
            # The saved policy is too near zero for a reliable symmetric
            # difference. Check the same adjoint implementation nearby.
            probe = policy.copy()
            probe[node] = 0.02
            _, probe_gradient = objective.value_and_gradient(probe)
            fd, step, method, consistency, audited_gradient, audited_m = (
                audited_finite_difference(objective, probe, node,
                    probe_gradient[node], upper, tolerance, allow_probe=False))
            return fd, step, "nearby_"+method, consistency, audited_gradient, audited_m
        return central, central_step, "central_unresolved", float("inf"), adjoint_gradient, policy[node]
    pairs = [(abs(right[1]-left[1]), right)
             for left, right in zip(estimates[:-1], estimates[1:])]
    consistency_error, (step, derivative) = min(pairs, key=lambda pair: pair[0])
    if allow_probe and consistency_error >= tolerance and upper[node] > 0.05:
        # A local transition may sit inside every roundoff-safe stencil.
        # Verify the adjoint at a nearby interior point and label that check.
        probe = policy.copy()
        probe[node] = (0.02 if policy[node] < 0.04
                       else policy[node]-0.02)
        _, probe_gradient = objective.value_and_gradient(probe)
        fd, h, method, consistency, audited_gradient, audited_m = (
            audited_finite_difference(objective, probe, node,
                probe_gradient[node], upper, tolerance, allow_probe=False))
        if abs(fd-audited_gradient) < tolerance and consistency < tolerance:
            return fd, h, "nearby_"+method, consistency, audited_gradient, audited_m
    return derivative, step, "five_point", consistency_error, adjoint_gradient, policy[node]



class NormalizedObjective:
    """Numerically scale welfare around a fixed reference without changing policy."""

    def __init__(self, underlying, reference, scale):
        self.underlying, self.reference, self.scale = underlying, reference, scale

    def utility(self, m):
        return (self.scale/self.reference)*(self.underlying.utility(m)-self.reference)

    def value_and_gradient(self, m):
        value, gradient = self.underlying.value_and_gradient(m)
        factor = self.scale/self.reference
        return (factor*(value-self.reference), factor*gradient)

    def gradient(self, m):
        return self.value_and_gradient(m)[1]

    def diagnostics(self):
        return self.underlying.diagnostics()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--damage", type=Path,
                        help="Saved damage grid; defaults to data/revised_default")
    parser.add_argument("--common-damage-grid", type=Path,
                        help="Use one full-tree damage simulation on a common five-year physical calendar")
    parser.add_argument("--start", type=Path,
                        help="Same-tree starting policy; defaults to data/revised_default")
    parser.add_argument("--output", type=Path,
                        help="New result directory; defaults to timestamped data/new_outputs")
    parser.add_argument("--tree", choices=TREES, default="full")
    parser.add_argument("--prob-scale", type=float, default=1.0)
    parser.add_argument("--ra", type=float, default=BASELINE.ra)
    parser.add_argument("--eis", type=float, default=BASELINE.eis)
    parser.add_argument("--prtp", type=float, default=BASELINE.prtp,
                        help="Annual pure rate of time preference as a fraction")
    parser.add_argument("--growth", type=float, default=BASELINE.growth,
                        help="Annual exogenous consumption growth")
    add_terminal_arguments(parser)
    parser.add_argument("--tech-const", type=float, default=BASELINE.tech_const,
                        help="Annual exogenous abatement technology improvement in percent")
    parser.add_argument("--tech-scale", type=float, default=BASELINE.tech_scale,
                        help="Endogenous abatement learning coefficient in percent")
    parser.add_argument("--cost-floor", type=float, default=0.0,
                        help="Non-compressible share of 2030 marginal abatement and removal costs")
    parser.add_argument("--learning-reference-year", type=int,
                        help="Use cumulative mitigated tons divided by fixed cumulative BAU tons through this calendar year; omit for inherited average-share learning")
    parser.add_argument("--backstop", choices=("s_curve", "cubic", "exp_cubic"),
                        default=BASELINE.backstop)
    parser.add_argument("--gamma", type=float, default=BASELINE.gamma,
                        help="Cubic coefficient, used with cubic or exp_cubic backstop")
    parser.add_argument("--backstop-curvature", type=float,
                        default=BASELINE.backstop_curvature,
                        help="Positive removal curvature for the smooth S-curve (default 1)")
    parser.add_argument("--mac-horizontal-shift", type=float, default=0.0,
                        help="Mitigation-unit leftward shift of the revised smooth MAC")
    parser.add_argument("--mac-vertical-shift", type=float, default=0.0,
                        help="Base-year dollar-per-tonne upward shift of the revised MAC")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--refinement-maxiter", type=int, default=2000,
                        help="Adjoint L-BFGS-B refinement limit if FOC gap remains above $1/t")
    parser.add_argument("--safe-refinement", action="store_true",
                        help="Reject infeasible consumption trial steps during direct refinement")
    parser.add_argument("--coordinate-refinement-steps", type=int, default=0,
                        help="Optional gradient-based FOC polish after welfare line search")
    parser.add_argument("--coupled-refinement-rounds", type=int, default=0,
                        help="Optional coupled KKT polish for rare-node FOC failures")
    parser.add_argument("--initial-maxiter", type=int, default=150)
    parser.add_argument("--guard", type=float, default=4.0)
    parser.add_argument("--economic-upper-bound", type=float,
                        help="Optional economic mitigation cap; distinct from numerical guard")
    parser.add_argument("--objective-scale", type=float,
                        help="Defaults to 1e4 for both optimal and delayed solves")
    parser.add_argument("--fixed-years", type=float, default=0.0,
                        help="Set pre-delay mitigation controls to zero")
    parser.add_argument("--early-cap", type=float,
                        help="Optional maximum mitigation before --cap-years")
    parser.add_argument("--cap-years", type=float, default=0.0,
                        help="Years during which --early-cap applies")
    args = parser.parse_args(argv)
    terminal_settings = terminal_settings_from_args(args)
    try:
        terminal_metadata = terminal_settings.describe(ra=args.ra, eis=args.eis,
            prtp=args.prtp, modeled_growth=args.growth)
    except ValueError as exc:
        parser.error(str(exc))
    if args.economic_upper_bound is not None and not 0 < args.economic_upper_bound <= args.guard:
        parser.error("--economic-upper-bound must be positive and at most --guard")
    if not np.isfinite(args.prob_scale) or args.prob_scale <= 0:
        parser.error("--prob-scale must be finite and positive")
    if args.early_cap is not None and (args.early_cap < 0 or args.cap_years <= 0):
        parser.error("--early-cap requires a nonnegative cap and positive --cap-years")
    if args.backstop == "s_curve" and args.gamma != 0:
        parser.error("--gamma requires --backstop cubic or exp_cubic")
    if not np.isfinite(args.backstop_curvature) or args.backstop_curvature <= 0:
        parser.error("--backstop-curvature must be finite and positive")
    if args.backstop != "s_curve" and args.backstop_curvature != 1.0:
        parser.error("--backstop-curvature applies only to the smooth S-curve")
    if (not np.isfinite(args.mac_horizontal_shift) or args.mac_horizontal_shift < 0 or
            not np.isfinite(args.mac_vertical_shift) or args.mac_vertical_shift < 0):
        parser.error("MAC shifts must be finite and nonnegative")
    if (args.mac_horizontal_shift or args.mac_vertical_shift) and args.backstop != "s_curve":
        parser.error("MAC shifts require --backstop s_curve")
    if args.backstop == "exp_cubic" and args.guard > 1.5:
        parser.error("--backstop exp_cubic requires --guard <= 1.5 to keep costs finite")
    if args.tech_scale < 0 or args.tech_const < 0 or args.gamma < 0:
        parser.error("Technology rates and gamma must be nonnegative")
    if not np.isfinite(args.cost_floor) or not 0 <= args.cost_floor < 1:
        parser.error("--cost-floor must be finite and in [0, 1)")
    if args.damage is None:
        args.damage = DEFAULT_INPUTS / f"{args.tree}_damages.npz"
    if args.start is None:
        args.start = DEFAULT_INPUTS / f"{args.tree}_baseline_start.npz"
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        delay = f"delay{args.fixed_years:g}" if args.fixed_years else "optimal"
        args.output = DEFAULT_OUTPUTS / f"revised_{args.tree}_{delay}_{stamp}"
    for source in (args.damage, args.start, args.common_damage_grid):
        if source is None:
            continue
        if not source.is_file():
            parser.error(f"Missing default input {source}; pass an explicit path")
    if args.output.exists():
        raise FileExistsError(args.output)
    objective = build_saved_revised_model(args.damage.resolve(),
        decision_times=TREES[args.tree], prob_scale=args.prob_scale,
        ra=args.ra, eis=args.eis, prtp=args.prtp,
        **terminal_settings.model_kwargs(),
        growth=args.growth, tech_const=args.tech_const,
        cost_floor=args.cost_floor,
        tech_scale=args.tech_scale, learning_reference_year=args.learning_reference_year,
        backstop=args.backstop, gamma=args.gamma,
        backstop_curvature=args.backstop_curvature,
        mac_horizontal_shift=args.mac_horizontal_shift,
        mac_vertical_shift=args.mac_vertical_shift,
        common_damage_path=(None if args.common_damage_grid is None else
                            args.common_damage_grid.resolve()))
    with np.load(args.start.resolve(), allow_pickle=False) as saved:
        start = np.asarray(saved["mitigation"], float).copy()
    n = objective.tree.num_decision_nodes
    if start.shape != (n,):
        raise ValueError("Saved starting policy belongs to another tree")
    lower = np.zeros(n)
    upper = np.full(n, args.guard if args.economic_upper_bound is None
                    else args.economic_upper_bound)
    fixed = []
    for period, elapsed in enumerate(objective.tree.decision_times[:-1]):
        if elapsed >= args.fixed_years:
            break
        first, last = objective.tree.get_nodes_in_period(period)
        upper[first:last+1] = 0.0
        fixed.extend(range(first, last+1))
    if args.early_cap is not None:
        for period, elapsed in enumerate(objective.tree.decision_times[:-1]):
            if elapsed >= args.cap_years:
                break
            first, last = objective.tree.get_nodes_in_period(period)
            upper[first:last+1] = np.minimum(upper[first:last+1], args.early_cap)
            if args.early_cap == 0:
                fixed.extend(i for i in range(first, last+1) if i not in fixed)
    start[fixed] = 0.0
    independent = np.full(n, 0.8)
    independent[fixed] = 0.0
    independent = np.clip(independent, lower, upper)
    start = np.clip(start, lower, upper)
    if args.backstop in ("cubic", "exp_cubic"):
        # The saved S-curve policy can have substantial removal and is not a
        # feasible warm start under a steep alternative removal technology.
        # Up to m=1 all three curves coincide, so this remains a comparable,
        # strictly feasible starting policy without constraining the solve.
        start = np.minimum(start, 1.0)
    start_value, start_grad = objective.value_and_gradient(start)
    start_diag = objective.diagnostics()
    start_gap = feasible_foc_gap(start_diag["interval_foc_gap_dollars_per_ton"],
                                 start, upper, fixed)
    independent_value = objective.utility(independent)
    objective_scale = (args.objective_scale if args.objective_scale is not None
                       else 1e4)
    print("START", json.dumps(dict(saved=start_value, independent=independent_value,
        max_feasible_foc_gap_dollars_per_ton=start_gap)), flush=True)

    def solve_stage(stage_objective, starts, maxiter, name):
        solver = CandidateScreenedLBFGSB(
            utility=stage_objective, objective_with_gradient=stage_objective,
            gradient_mode="adjoint", optimizer_name="adjoint_lbfgsb",
            lower_bounds=lower, upper_bounds=upper,
            mandatory_starts=starts, warm_starts=starts,
            n_candidates=4, max_candidates=4, n_local_starts=len(starts),
            max_local_starts=len(starts), warm_start_perturbations=0,
            structured_start_count=0, preserve_diverse_starts=False,
            escalate_on_dispersion=False, maxiter=maxiter,
            ftol=1e-13, gtol=1e-7, kkt_check=True, validate_gradient=True,
            recover_infeasible_consumption=True,
            gradient_validation_epsilons=(
                (1e-5, 1e-6) if args.common_damage_grid is not None else None),
            print_progress=True, callback_progress_every=10,
            gradient_progress_every=0, scenario_name=name)
        return solver.run()

    clock = time.perf_counter()
    # Unscaled welfare keeps the first line-search steps inside the feasible
    # consumption region.  Only after that robust warm start do we rescale
    # the near-flat objective to resolve low-probability late-node FOCs.
    first_policy, first_welfare, first_solver_diag = solve_stage(
        objective, [start, independent], args.initial_maxiter,
        "revised_smooth_flow_initial")
    first_value, _ = objective.value_and_gradient(first_policy)
    first_diag = objective.diagnostics()
    first_gap = feasible_foc_gap(first_diag["interval_foc_gap_dollars_per_ton"],
                                 first_policy, upper, fixed)
    if abs(first_value-first_welfare) > 1e-9:
        raise RuntimeError("Initial-stage welfare does not match objective")
    continuation_solver_diag = None
    if fixed and first_gap > 5.0:
        continuation_policy, continuation_welfare, continuation_solver_diag = solve_stage(
            objective, [first_policy, independent], args.initial_maxiter,
            "revised_smooth_flow_delay_continuation")
        if continuation_welfare >= first_value-1e-10:
            first_policy = continuation_policy
            first_value, _ = objective.value_and_gradient(first_policy)
            first_diag = objective.diagnostics()
            first_gap = feasible_foc_gap(first_diag["interval_foc_gap_dollars_per_ton"],
                                         first_policy, upper, fixed)
    normalized = NormalizedObjective(objective, first_value, objective_scale)
    # A nearby but distinct start probes the local basin without proposing a
    # large, infeasible scaled step from the distant constant policy again.
    nearby = np.clip(0.98*first_policy+0.02*independent, lower, upper)
    second_policy, second_welfare, second_solver_diag = solve_stage(
        normalized, [first_policy, nearby], args.maxiter,
        "revised_smooth_flow_scaled")
    solve_seconds = time.perf_counter()-clock
    second_value, _ = objective.value_and_gradient(second_policy)
    second_diag = objective.diagnostics()
    second_gap = feasible_foc_gap(second_diag["interval_foc_gap_dollars_per_ton"],
                                  second_policy, upper, fixed)
    if abs(normalized.utility(second_policy)-second_welfare) > 1e-7:
        raise RuntimeError("Scaled-stage welfare does not match objective")
    if ((first_gap < 1.0 and second_gap >= 1.0)
            or (second_value < first_value-1e-9)
            or (first_gap < second_gap and second_value-first_value < 1e-9)):
        policy, selected_stage = first_policy, "initial"
    else:
        policy, selected_stage = second_policy, "scaled"
    value, gradient = objective.value_and_gradient(policy)
    final_diag = objective.diagnostics()
    final_gap = feasible_foc_gap(final_diag["interval_foc_gap_dollars_per_ton"],
                                 policy, upper, fixed)
    refinement_diagnostics = None
    if final_gap >= 1.0 and args.refinement_maxiter > 0:
        active = np.flatnonzero(upper > lower+1e-14)
        reference = value
        initial_policy = policy.copy()
        invalid_refinement_trials = 0
        last_feasible_refinement = {"x": initial_policy[active].copy(), "fun": 0.0}
        def refine_objective(active_values):
            nonlocal invalid_refinement_trials
            candidate = initial_policy.copy()
            candidate[active] = active_values
            try:
                candidate_value, candidate_gradient = objective.value_and_gradient(candidate)
            except FloatingPointError as error:
                if not args.safe_refinement or str(error) != "Nonpositive five-year consumption":
                    raise
                invalid_refinement_trials += 1
                # Give the line search an uphill value and a direction back
                # toward a feasible evaluated policy.
                displacement = active_values-last_feasible_refinement["x"]
                weight = 1e4*max(1.0, abs(last_feasible_refinement["fun"]))
                return (last_feasible_refinement["fun"]
                        + weight*(1.0+float(np.dot(displacement, displacement))),
                        2.0*weight*displacement)
            scale = objective_scale/reference
            fun = -(candidate_value-reference)*scale
            grad = -candidate_gradient[active]*scale
            last_feasible_refinement["x"] = active_values.copy()
            last_feasible_refinement["fun"] = float(fun)
            return fun, grad
        try:
            refined = minimize(refine_objective, initial_policy[active], jac=True,
                method="L-BFGS-B", bounds=list(zip(lower[active], upper[active])),
                options=dict(maxiter=args.refinement_maxiter, maxls=100,
                             ftol=1e-15, gtol=1e-10))
            candidate = initial_policy.copy()
            candidate[active] = refined.x
            candidate_value, candidate_gradient = objective.value_and_gradient(candidate)
            candidate_diag = objective.diagnostics()
            candidate_gap = feasible_foc_gap(
                candidate_diag["interval_foc_gap_dollars_per_ton"],
                candidate, upper, fixed)
            refinement_diagnostics = dict(success=bool(refined.success),
                message=str(refined.message), iterations=int(refined.nit),
                evaluations=int(refined.nfev), candidate_gap=candidate_gap,
                candidate_utility=candidate_value,
                n_infeasible_consumption_trials=invalid_refinement_trials)
            if candidate_value >= value-1e-9 and candidate_gap < final_gap:
                policy, value, gradient = candidate, candidate_value, candidate_gradient
                final_diag, final_gap = candidate_diag, candidate_gap
                selected_stage = "direct_adjoint_refinement"
        except FloatingPointError as error:
            refinement_diagnostics = dict(success=False, message=str(error),
                n_infeasible_consumption_trials=invalid_refinement_trials)
    if final_gap >= 1.0 and args.coupled_refinement_rounds > 0:
        from scripts.coupled_refinement import refine_coupled_stationarity
        coupled = refine_coupled_stationarity(
            objective, policy, upper, fixed,
            max_rounds=args.coupled_refinement_rounds)
        if coupled[4] < final_gap and coupled[1] >= value-1e-12:
            policy, value, gradient, final_diag, final_gap = coupled[:5]
            selected_stage = "coupled_stationarity_refinement"
        if refinement_diagnostics is None:
            refinement_diagnostics = {}
        refinement_diagnostics["coupled_stationarity"] = coupled[5]
    if final_gap >= 1.0 and args.coordinate_refinement_steps > 0:
        from scripts.coordinate_refinement import refine_stationarity
        polished = refine_stationarity(
            objective, policy, upper, fixed,
            max_steps=args.coordinate_refinement_steps)
        if polished[4] < final_gap and polished[1] >= value-1e-9:
            policy, value, gradient, final_diag, final_gap = polished[:5]
            selected_stage = "coordinate_stationarity_refinement"
        if refinement_diagnostics is None:
            refinement_diagnostics = {}
        refinement_diagnostics["coordinate_stationarity"] = polished[5]
    solve_seconds = time.perf_counter()-clock
    # Independent differentiation is deliberately applied to every free node
    # in the saved solution. This is separate from the adjoint used to solve.
    gradient_audit = []
    finite_step = 2e-5
    fixed_set = set(fixed)
    for node in range(n):
        if node in fixed_set or policy[node] <= finite_step or policy[node] >= args.guard-finite_step:
            continue
        (finite_gradient, checked_step, method, consistency_error,
         audited_gradient, audited_mitigation) = audited_finite_difference(
            objective, policy, node, gradient[node], upper)
        error = abs(audited_gradient-finite_gradient)
        gradient_audit.append(dict(node=node, mitigation=float(policy[node]),
            adjoint_gradient=float(gradient[node]),
            audit_mitigation=float(audited_mitigation),
            audit_adjoint_gradient=float(audited_gradient),
            finite_difference=float(finite_gradient),
            finite_difference_step=float(checked_step), method=method,
            self_consistency_error=float(consistency_error),
            absolute_error=float(error)))
    gradient_max_error = max((row["absolute_error"] for row in gradient_audit), default=0.0)
    gradient_max_consistency_error = max(
        (row["self_consistency_error"] for row in gradient_audit), default=0.0)
    probability_sums = []
    for period in range(len(objective.tree.decision_times)-1):
        first, last = objective.tree.get_nodes_in_period(period)
        probability_sums.append(float(np.sum(objective.tree.node_prob[first:last+1])))
    local_stationarity_certified = bool(final_gap < 1.0 and gradient_max_error < 1e-7
        and gradient_max_consistency_error < 1e-7
        and final_diag["minimum_consumption"] > 0
        and not np.any(policy >= args.guard-1e-8)
        and np.allclose(probability_sums, 1.0, rtol=0, atol=1e-12))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/"initial_stage_policy.npz",
                        mitigation=first_policy, utility=first_value)
    np.savez_compressed(args.output/"policy.npz", mitigation=policy, utility=value,
                        gradient=gradient)
    decision_state_count, five_year_state_count, terminal_state_count = export_state_trees(
        objective, policy, args.output, value)
    archived_inputs = {}
    for label, source, name in (
            ("damage_grid", args.damage, "damage_grid_input.npz"),
            ("ssp_baselines", ROOT/"data/SSP_baselines.csv", "SSP_baselines_input.csv")):
        destination = args.output/name
        shutil.copy2(source, destination)
        archived_inputs[label] = dict(file=name,
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    if args.common_damage_grid is not None:
        source = args.common_damage_grid.resolve()
        destination = args.output/"common_damage_grid_input.npz"
        shutil.copy2(source, destination)
        archived_inputs["common_damage_grid"] = dict(
            file=destination.name, source=str(source),
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    with (args.output/"gradient_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gradient_audit[0]))
        writer.writeheader()
        writer.writerows(gradient_audit)
    gaps = np.asarray(final_diag["interval_foc_gap_dollars_per_ton"], float)
    interval_rows = []
    with (args.output/"interval_nodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["year", "node", "probability",
            "mitigation", "status", "interval_mac", "net_climate_benefit",
            "warming_marginal_damage", "low_ghg_adjustment", "learning_benefit",
            "total_marginal_policy_benefit", "fo_violation", "minimum_consumption"])
        writer.writeheader()
        for period, elapsed in enumerate(objective.tree.decision_times[:-1]):
            first, last = objective.tree.get_nodes_in_period(period)
            decision_year = objective.tree.base_year+elapsed
            for node in range(first, last+1):
                denom = final_diag["dollar_denominator"][node]
                mac = -final_diag["direct_cost_gradient"][node]/denom
                climate_benefit = final_diag["climate_gradient"][node]/denom
                learning_benefit = final_diag["learning_gradient"][node]/denom
                warming_damage = final_diag["warming_gradient"][node]/denom
                low_ghg_adjustment = final_diag["low_ghg_gradient"][node]/denom
                if not np.isclose(climate_benefit, warming_damage+low_ghg_adjustment,
                                  rtol=1e-8, atol=1e-6):
                    raise RuntimeError("Warming and low-GHG components do not sum to net climate benefit")
                status = ("fixed" if node in fixed else "lower_bound" if policy[node] <= 1e-8
                          else ("policy_cap" if upper[node] < args.guard-1e-8
                                else "numerical_upper_bound") if policy[node] >= upper[node]-1e-8
                          else "interior")
                violation = (0.0 if status == "fixed" else
                    max(gaps[node], 0.0) if status == "lower_bound" else
                    max(-gaps[node], 0.0) if status in ("numerical_upper_bound", "policy_cap") else
                    abs(gaps[node]))
                row = dict(year=int(decision_year), node=node,
                    probability=float(objective.tree.node_prob[node]),
                    mitigation=float(policy[node]), status=status,
                    interval_mac=mac, net_climate_benefit=climate_benefit,
                    warming_marginal_damage=warming_damage,
                    low_ghg_adjustment=low_ghg_adjustment,
                    learning_benefit=learning_benefit, fo_violation=violation,
                    total_marginal_policy_benefit=climate_benefit+learning_benefit,
                    minimum_consumption=final_diag["minimum_consumption"])
                writer.writerow(row)
                interval_rows.append(row)
    summary_rows = []
    for year in sorted({row["year"] for row in interval_rows}):
        rows = [row for row in interval_rows if row["year"] == year]
        weight = sum(row["probability"] for row in rows)
        weighted = lambda key: sum(row[key]*row["probability"] for row in rows)/weight
        summary_rows.append(dict(year=year, nodes=len(rows),
            interior_nodes=sum(row["status"] == "interior" for row in rows),
            fixed_nodes=sum(row["status"] == "fixed" for row in rows),
            lower_bound_nodes=sum(row["status"] == "lower_bound" for row in rows),
            numerical_upper_bound_nodes=sum(row["status"] == "numerical_upper_bound" for row in rows),
            policy_cap_nodes=sum(row["status"] == "policy_cap" for row in rows),
            mean_mitigation=weighted("mitigation"), mean_interval_mac=weighted("interval_mac"),
            mean_net_climate_benefit=weighted("net_climate_benefit"),
            mean_warming_marginal_damage=weighted("warming_marginal_damage"),
            mean_low_ghg_adjustment=weighted("low_ghg_adjustment"),
            mean_learning_benefit=weighted("learning_benefit"),
            mean_total_marginal_policy_benefit=weighted("total_marginal_policy_benefit"),
            max_feasible_foc_gap_dollars_per_ton=max(row["fo_violation"] for row in rows)))
    with (args.output/"interval_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = dict(model="EZDelay revised smooth five-year-flow",
        tree=args.tree, decision_times=TREES[args.tree], prob_scale=args.prob_scale,
        damage_source=str(args.damage.resolve()),
        start_source=str(args.start.resolve()), endogenous_learning=bool(args.tech_scale),
        growth=args.growth, terminal_continuation=terminal_metadata,
        terminal_start_year=2025+TREES[args.tree][-1],
        tech_const=args.tech_const, tech_scale=args.tech_scale,
        cost_floor=args.cost_floor,
        learning_reference_year=args.learning_reference_year,
        backstop=args.backstop, gamma=args.gamma,
        backstop_curvature=args.backstop_curvature, no_free_lunch=True,
        mac_horizontal_shift=args.mac_horizontal_shift,
        mac_vertical_shift=args.mac_vertical_shift,
        flat_ssp2_after_2100=True, pchip_damage=True, smooth_backstop=(args.backstop == "s_curve"),
        cost_accounting="five_year_emissions_consumption_and_technology",
        climate_consumption_accounting=("common_five_year_physical_calendar"
            if args.common_damage_grid is not None else "inherited_decision_interval_interpolation"),
        technology_timing="five_year", learning_timing=(
            ("cumulative_tons_over_fixed_BAU_to_reference_year"
             if args.learning_reference_year is not None else
             "BAU_weighted_mitigation_to_five_year_date") if args.tech_scale else None),
        ra=args.ra, eis=args.eis, prtp=args.prtp,
        **terminal_settings.model_kwargs(),
        numerical_mitigation_guard=args.guard,
        economic_upper_bound=args.economic_upper_bound,
        fixed_years=args.fixed_years, fixed_nodes=fixed,
        early_cap=args.early_cap, cap_years=args.cap_years,
        initial_utility=start_value, independent_start_utility=independent_value,
        final_utility=value, utility_gain=value-start_value,
        objective_scale=objective_scale,
        initial_stage_maxiter=args.initial_maxiter, selected_stage=selected_stage,
        initial_stage_utility=first_value,
        initial_stage_feasible_foc_gap_dollars_per_ton=first_gap,
        scaled_stage_utility=second_value,
        scaled_stage_feasible_foc_gap_dollars_per_ton=second_gap,
        solver_seconds=solve_seconds, max_mitigation=float(np.max(policy)),
        minimum_consumption=final_diag["minimum_consumption"],
        initial_feasible_foc_gap_dollars_per_ton=start_gap,
        final_feasible_foc_gap_dollars_per_ton=final_gap,
        certified_below_one_dollar=bool(final_gap < 1.0),
        finite_difference_step=finite_step,
        finite_difference_nodes_checked=len(gradient_audit),
        finite_difference_max_absolute_error=gradient_max_error,
        finite_difference_max_self_consistency_error=gradient_max_consistency_error,
        decision_period_probability_sums=probability_sums,
        local_stationarity_certified=local_stationarity_certified,
        numerical_guard_binds=bool(np.any(policy >= args.guard-1e-8)),
        initial_solver_diagnostics=json_safe(first_solver_diag),
        delay_continuation_solver_diagnostics=json_safe(continuation_solver_diag),
        scaled_solver_diagnostics=json_safe(second_solver_diag),
        refinement_diagnostics=json_safe(refinement_diagnostics),
        objective_diagnostics=final_diag)
    manifest["archived_inputs"] = archived_inputs
    manifest["decision_state_rows"] = decision_state_count
    manifest["five_year_state_rows"] = five_year_state_count
    manifest["terminal_state_rows"] = terminal_state_count
    (args.output/"manifest.json").write_text(json.dumps(json_safe(manifest), indent=2)+"\n")
    print("FINAL", json.dumps({k: v for k, v in manifest.items()
        if k not in ("initial_solver_diagnostics", "delay_continuation_solver_diagnostics",
                     "scaled_solver_diagnostics",
                     "objective_diagnostics")}), flush=True)


if __name__ == "__main__":
    main()

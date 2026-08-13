#!/usr/bin/env python3
"""Run a tiny EZ-style tree model through the screened L-BFGS-B optimizers.

This is a laptop-sized smoke check for the optimizer plumbing.  It is not the
full EZDelay model; it is a differentiable 3-node planner with the same broad
channels: mitigation lowers emissions and damages, early mitigation improves
later technology costs through learning, and utility is recursive Epstein-Zin.
"""

from __future__ import annotations

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.optimization import CandidateScreenedLBFGSB, ObjectiveWithGradient
from src.tree import TreeModel


class MiniEZTreeObjective(ObjectiveWithGradient):
    """A 3-decision-node differentiable miniature of the EZDelay channels."""

    def __init__(self, tree: TreeModel):
        self.tree = tree
        self.beta = 0.985 ** 5
        self.ra = 10.0
        self.eis = 0.833
        self.alpha = 1.0 - self.ra
        self.rho = 1.0 - 1.0 / self.eis
        self.growth = 0.015
        self.terminal_multiplier = 1.0 / (1.0 - self.beta * (1.0 + self.growth) ** self.rho) ** (1.0 / self.rho)

        self.cons0 = 1.0
        self.cons1 = (1.0 + self.growth) ** 5
        self.emissions0 = 42.0
        self.emissions1 = 44.0
        self.cost_scale = 0.04
        self.cost_power = 2.0
        self.removal_premium = 6.0
        self.learning_rate = 0.30
        self.damage_scale = 8.0e-6
        self.fragility = np.array([2.0, 0.8])
        self.probs = np.asarray(tree.final_states_prob, dtype=float)

    def _cost_phi(self, m):
        over = np.maximum(m - 1.0, 0.0)
        return (
            (np.exp(self.cost_power * m) - 1.0) / self.cost_power
            - m
            + self.removal_premium * over ** 2
        )

    def _cost_phi_prime(self, m):
        over = np.maximum(m - 1.0, 0.0)
        return np.exp(self.cost_power * m) - 1.0 + 2.0 * self.removal_premium * over

    def _components(self, m):
        m = np.asarray(m, dtype=float)
        m0, m_hi, m_lo = m

        cost0 = self.cost_scale * self._cost_phi(m0)
        tech_child = np.exp(-self.learning_rate * m0)
        cost_hi = self.cost_scale * tech_child * self._cost_phi(m_hi)
        cost_lo = self.cost_scale * tech_child * self._cost_phi(m_lo)

        cum_hi = self.emissions0 * (1.0 - m0) + self.emissions1 * (1.0 - m_hi)
        cum_lo = self.emissions0 * (1.0 - m0) + self.emissions1 * (1.0 - m_lo)
        damage_hi = self.damage_scale * self.fragility[0] * cum_hi ** 2
        damage_lo = self.damage_scale * self.fragility[1] * cum_lo ** 2

        c0 = self.cons0 * (1.0 - cost0)
        c_hi = self.cons1 * (1.0 - damage_hi) * (1.0 - cost_hi)
        c_lo = self.cons1 * (1.0 - damage_lo) * (1.0 - cost_lo)
        c = np.array([c0, c_hi, c_lo])
        if np.any(c <= 0.0):
            return None

        terminal = self.terminal_multiplier * np.array([c_hi, c_lo])
        ce_inner = np.sum(self.probs * terminal ** self.alpha)
        ce = ce_inner ** (1.0 / self.alpha)
        utility_inner = (
            (1.0 - self.beta) * c0 ** self.rho
            + self.beta * ce ** self.rho
        )
        utility = utility_inner ** (1.0 / self.rho)
        return {
            "utility": float(utility),
            "consumption": c,
            "cost": np.array([cost0, cost_hi, cost_lo]),
            "damage": np.array([0.0, damage_hi, damage_lo]),
            "ce": float(ce),
            "terminal": terminal,
            "cumemit": np.array([cum_hi, cum_lo]),
            "tech_child": float(tech_child),
        }

    def utility(self, m):
        components = self._components(m)
        if components is None:
            return -np.inf
        return components["utility"]

    def gradient(self, m):
        m = np.asarray(m, dtype=float)
        m0, m_hi, m_lo = m
        comp = self._components(m)
        if comp is None:
            return np.full_like(m, np.nan)

        utility = comp["utility"]
        c0, c_hi, c_lo = comp["consumption"]
        terminal_hi, terminal_lo = comp["terminal"]
        ce = comp["ce"]
        cost0, cost_hi, cost_lo = comp["cost"]
        damage_hi, damage_lo = comp["damage"][1:]
        cum_hi, cum_lo = comp["cumemit"]
        tech_child = comp["tech_child"]

        d_utility_dc0 = (
            utility ** (1.0 - self.rho)
            * (1.0 - self.beta)
            * c0 ** (self.rho - 1.0)
        )
        d_utility_dce = (
            utility ** (1.0 - self.rho)
            * self.beta
            * ce ** (self.rho - 1.0)
        )
        d_ce_d_terminal = (
            ce ** (1.0 - self.alpha)
            * self.probs
            * np.array([terminal_hi, terminal_lo]) ** (self.alpha - 1.0)
        )
        d_utility_dchild_c = (
            d_utility_dce * d_ce_d_terminal * self.terminal_multiplier
        )
        d_u_dc_hi, d_u_dc_lo = d_utility_dchild_c

        d_cost0_dm0 = self.cost_scale * self._cost_phi_prime(m0)
        d_cost_hi_dm_hi = self.cost_scale * tech_child * self._cost_phi_prime(m_hi)
        d_cost_lo_dm_lo = self.cost_scale * tech_child * self._cost_phi_prime(m_lo)
        d_cost_hi_dm0 = -self.learning_rate * cost_hi
        d_cost_lo_dm0 = -self.learning_rate * cost_lo

        d_damage_hi_dm0 = -2.0 * self.damage_scale * self.fragility[0] * cum_hi * self.emissions0
        d_damage_lo_dm0 = -2.0 * self.damage_scale * self.fragility[1] * cum_lo * self.emissions0
        d_damage_hi_dm_hi = -2.0 * self.damage_scale * self.fragility[0] * cum_hi * self.emissions1
        d_damage_lo_dm_lo = -2.0 * self.damage_scale * self.fragility[1] * cum_lo * self.emissions1

        dc0_dm0 = -self.cons0 * d_cost0_dm0
        dc_hi_dm0 = self.cons1 * (
            -d_damage_hi_dm0 * (1.0 - cost_hi)
            - (1.0 - damage_hi) * d_cost_hi_dm0
        )
        dc_lo_dm0 = self.cons1 * (
            -d_damage_lo_dm0 * (1.0 - cost_lo)
            - (1.0 - damage_lo) * d_cost_lo_dm0
        )
        dc_hi_dm_hi = self.cons1 * (
            -d_damage_hi_dm_hi * (1.0 - cost_hi)
            - (1.0 - damage_hi) * d_cost_hi_dm_hi
        )
        dc_lo_dm_lo = self.cons1 * (
            -d_damage_lo_dm_lo * (1.0 - cost_lo)
            - (1.0 - damage_lo) * d_cost_lo_dm_lo
        )

        grad = np.zeros(3)
        grad[0] = (
            d_utility_dc0 * dc0_dm0
            + d_u_dc_hi * dc_hi_dm0
            + d_u_dc_lo * dc_lo_dm0
        )
        grad[1] = d_u_dc_hi * dc_hi_dm_hi
        grad[2] = d_u_dc_lo * dc_lo_dm_lo
        return grad

    def summary(self, m):
        comp = self._components(m)
        return {
            "consumption_min": float(np.min(comp["consumption"])),
            "consumption_max": float(np.max(comp["consumption"])),
            "cost": comp["cost"],
            "damage": comp["damage"],
            "cumemit": comp["cumemit"],
            "certain_equivalent": comp["ce"],
        }


def solve(label, objective, gradient_mode):
    optimizer = CandidateScreenedLBFGSB(
        utility=objective,
        objective_with_gradient=objective if gradient_mode == "adjoint" else None,
        gradient_mode=gradient_mode,
        optimizer_name="adjoint_lbfgsb" if gradient_mode == "adjoint" else "lbfgsb_multistart",
        lower_bounds=np.zeros(3),
        upper_bounds=np.full(3, 1.5),
        n_candidates=32,
        n_local_starts=5,
        max_candidates=64,
        max_local_starts=8,
        maxiter=100,
        ftol=1e-10,
        gtol=1e-8,
        utility_spread_tol=1e-10,
        utility_spread_rel_tol=1e-8,
        structured_start_count=12,
        perturbation_check=True,
        perturbation_step=0.005,
        projected_gradient_tol=1e-5,
        validate_gradient=(gradient_mode == "adjoint"),
        gradient_validation_directions=5,
        print_progress=False,
    )
    m, utility, diag = optimizer.run()
    print("\n{} ({})".format(label, gradient_mode))
    print("  utility:                 {:.10f}".format(utility))
    print("  mitigation [root, high, low]: {}".format(np.array2string(m, precision=6)))
    print("  success_diagnostics:     {}".format(diag.get("success_diagnostics")))
    print("  scipy_success:           {}".format(diag.get("success_scipy")))
    print("  gradient_validation:     {}".format(diag.get("gradient_validation_status")))
    print("  projected_grad_inf_norm: {:.3e}".format(diag.get("projected_grad_inf_norm", np.nan)))
    print("  max perturbation gain:   {:.3e}".format(diag.get("max_perturbation_utility_gain", np.nan)))
    print("  selected start groups:   {}".format(diag.get("selected_start_source_groups")))
    return m, utility, diag


def main():
    tree = TreeModel([0, 5, 10])
    objective = MiniEZTreeObjective(tree)
    print("Mini EZ-style optimizer smoke check")
    print("  decision times: {}".format(tree.decision_times.tolist()))
    print("  decision nodes: {}".format(tree.num_decision_nodes))
    print("  final-state probabilities: {}".format(np.array2string(tree.final_states_prob, precision=4)))

    m_adj, u_adj, diag_adj = solve("Exact-gradient screened solve", objective, "adjoint")
    m_fd, u_fd, diag_fd = solve("Finite-difference screened solve", objective, "finite_difference")

    summary = objective.summary(m_adj)
    print("\nEconomic summaries at exact-gradient solution")
    print("  consumption range:       [{:.6f}, {:.6f}]".format(summary["consumption_min"], summary["consumption_max"]))
    print("  costs [root, high, low]: {}".format(np.array2string(summary["cost"], precision=6)))
    print("  damages [root, high, low]: {}".format(np.array2string(summary["damage"], precision=6)))
    print("  child cumulative emissions: {}".format(np.array2string(summary["cumemit"], precision=6)))
    print("  certain equivalent:      {:.10f}".format(summary["certain_equivalent"]))
    print("\nComparison")
    print("  adjoint - finite-diff utility: {:.3e}".format(u_adj - u_fd))
    print("  max mitigation difference:     {:.3e}".format(np.max(np.abs(m_adj - m_fd))))

    if not diag_adj.get("success_diagnostics", False):
        raise SystemExit("Exact-gradient mini solve failed diagnostics.")
    if not np.isfinite(u_adj):
        raise SystemExit("Exact-gradient mini solve returned non-finite utility.")


if __name__ == "__main__":
    main()

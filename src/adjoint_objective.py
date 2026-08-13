"""
Theo Moers
tlm2160@columbia.edu
Columbia University

Exact-gradient objective wrapper for the production EZUtility model.
"""

import os
import threading
import time

import numpy as np

from src.optimization import ObjectiveWithGradient
from src.storage_tree import BigStorageTree, SmallStorageTree
from src.tools import get_integral_var_ub


def _num_cost_kink_nodes(cost, mitigation):
    """Return sharp removal-cost kinks relevant to nonsmooth KKT checks."""

    if float(getattr(cost, "backstop_smoothing_width", 0.0)) > 0.0:
        return 0
    return int(np.sum(np.isclose(mitigation, 1.0, rtol=0.0, atol=1e-8)))


class EZForwardSensitivityObjective(ObjectiveWithGradient):
    """Forward-sensitivity reference for :class:`src.utility.EZUtility`.

    The implementation mirrors ``EZUtility.utility`` in a forward sensitivity
    pass. It is retained as a validation oracle for the reverse-mode production
    objective and is not used by production runners.
    """

    def __init__(self, utility, value_parity_tol=None, fail_on_value_mismatch=None):
        self.utility_model = utility
        self.tree = utility.tree
        self.damage = utility.damage
        self.cost = utility.cost
        self.period_len = utility.period_len
        self.n_vars = int(self.tree.num_decision_nodes)
        self.value_parity_tol = float(
            os.environ.get(
                "ADJOINT_VALUE_PARITY_TOL",
                "1e-9" if value_parity_tol is None else str(value_parity_tol),
            )
        )
        if fail_on_value_mismatch is None:
            fail_on_value_mismatch = os.environ.get(
                "ADJOINT_FAIL_ON_VALUE_MISMATCH", "1"
            ) not in ("0", "false", "False", "no", "No")
        self.fail_on_value_mismatch = bool(fail_on_value_mismatch)
        self.last_diagnostics = {}

    def utility(self, m):
        return float(np.asarray(self.utility_model.utility(m)).reshape(-1)[0])

    def gradient(self, m):
        return self.value_and_gradient(m)[1]

    def value_and_gradient(self, m):
        m = np.asarray(m, dtype=float).reshape(-1)
        if len(m) != self.n_vars:
            raise ValueError(
                "mitigation length {} does not match tree decision nodes {}".format(
                    len(m), self.n_vars
                )
            )
        taped_value, grad, diagnostics = self._forward_sensitivity(m)
        legacy_value = self.utility(m)
        abs_diff = abs(taped_value - legacy_value)
        rel_diff = abs_diff / max(1.0, abs(legacy_value))
        diagnostics.update({
            "adjoint_value": float(taped_value),
            "adjoint_legacy_value": float(legacy_value),
            "adjoint_value_abs_diff": float(abs_diff),
            "adjoint_value_rel_diff": float(rel_diff),
        })
        self.last_diagnostics = diagnostics
        if self.fail_on_value_mismatch and rel_diff > self.value_parity_tol:
            raise RuntimeError(
                "EZForwardSensitivityObjective value parity failed: legacy {:.12g}, "
                "taped {:.12g}, rel diff {:.3g}".format(
                    legacy_value, taped_value, rel_diff
                )
            )
        return legacy_value, grad

    def diagnostics(self):
        return dict(self.last_diagnostics)

    def _zeros(self, rows):
        return np.zeros((int(rows), self.n_vars), dtype=float)

    def _forward_sensitivity(self, m):
        utility_tree = BigStorageTree(
            subinterval_len=self.period_len,
            decision_times=self.tree.decision_times,
        )
        cons_tree = BigStorageTree(
            subinterval_len=self.period_len,
            decision_times=self.tree.decision_times,
        )
        ce_tree = BigStorageTree(
            subinterval_len=self.period_len,
            decision_times=self.tree.decision_times,
        )
        cost_tree = SmallStorageTree(decision_times=self.tree.decision_times)

        d_utility = {period: self._zeros(len(values))
                     for period, values in utility_tree.tree.items()}
        d_cons = {period: self._zeros(len(values))
                  for period, values in cons_tree.tree.items()}
        d_cost = {period: self._zeros(len(values))
                  for period, values in cost_tree.tree.items()}

        diagnostics = {
            "num_damage_interp_clipped": 0,
            "num_damage_interp_knots": 0,
            "num_consumption_clipped": 0,
            "num_cost_kink_nodes": _num_cost_kink_nodes(self.cost, m),
            "num_overmitigation_nodes": int(np.sum(m > 1.0)),
        }

        self._terminal_period(
            m, utility_tree, cons_tree, cost_tree, d_utility, d_cons, d_cost,
            diagnostics
        )

        active_damage = None
        active_ddamage = None
        active_cost = None
        active_dcost = None

        for period in utility_tree.periods[::-1][1:]:
            damage_period = utility_tree.between_decision_times(period)
            cert_equiv, d_cert_equiv = self._certain_equivalence_with_deriv(
                period, damage_period, utility_tree, d_utility
            )

            if utility_tree.is_decision_period(period + self.period_len):
                active_damage, active_ddamage = self._damage_period(
                    m, damage_period, False, diagnostics
                )
                active_cost, active_dcost = self._cost_period(
                    m, damage_period, False, diagnostics
                )
                cost_key = cost_tree.index_below(period + self.period_len)
                cost_tree.set_value(cost_key, active_cost)
                d_cost[cost_key] = active_dcost

            period_consumption, d_period_consumption = self._consumption(
                damage_period,
                active_damage,
                active_ddamage,
                active_cost,
                active_dcost,
                diagnostics,
            )

            if not utility_tree.is_decision_period(period):
                period_consumption, d_period_consumption = (
                    self._interpolated_consumption(
                        period,
                        damage_period,
                        period_consumption,
                        d_period_consumption,
                        utility_tree,
                        cons_tree,
                        cost_tree,
                        d_cons,
                        d_cost,
                        active_cost,
                        active_dcost,
                        diagnostics,
                    )
                )

            if period == 0:
                # The frontier optimizer uses utility(), not adjusted_utility().
                pass

            ce_tree.set_value(period, self.utility_model.b * cert_equiv**self.utility_model.r)
            cons_tree.set_value(period, period_consumption)
            d_cons[period] = d_period_consumption
            u, du = self._intertemporal_aggregate_with_deriv(
                period_consumption, d_period_consumption,
                cert_equiv, d_cert_equiv
            )
            utility_tree.set_value(period, u)
            d_utility[period] = du

        value = float(utility_tree[0].reshape(-1)[0])
        grad = d_utility[0].reshape(-1, self.n_vars)[0].copy()
        return value, grad, diagnostics

    def _terminal_period(self, m, utility_tree, cons_tree, cost_tree,
                         d_utility, d_cons, d_cost, diagnostics):
        period = self.tree.num_periods
        damage, ddamage = self._damage_period(m, period, True, diagnostics)
        cost, dcost = self._cost_period(m, period, True, diagnostics)
        cost_tree.set_value(cost_tree.last_period, cost)
        d_cost[cost_tree.last_period] = dcost

        consumption = self.utility_model.potential_cons[-1] * (1.0 - damage)
        dcons = -self.utility_model.potential_cons[-1] * ddamage
        consumption, dcons = self._clip_consumption(consumption, dcons, diagnostics)
        cons_tree.set_value(cons_tree.last_period, consumption)
        d_cons[cons_tree.last_period] = dcons
        terminal_u, terminal_du = self._terminal_utility_with_deriv(consumption, dcons)
        utility_tree.set_value(utility_tree.last_period, terminal_u)
        d_utility[utility_tree.last_period] = terminal_du

    def _consumption(self, damage_period, damage, ddamage, cost, dcost, diagnostics):
        potential = self.utility_model.potential_cons[damage_period]
        consumption = potential * (1.0 - damage) * (1.0 - cost)
        dcons = potential * (
            -(1.0 - cost)[:, None] * ddamage
            - (1.0 - damage)[:, None] * dcost
        )
        return self._clip_consumption(consumption, dcons, diagnostics)

    def _clip_consumption(self, consumption, dcons, diagnostics):
        consumption = np.asarray(consumption, dtype=float).copy()
        clipped = consumption <= 0.0
        if np.any(clipped):
            diagnostics["num_consumption_clipped"] += int(np.sum(clipped))
            consumption[clipped] = 1e-18
            dcons = np.asarray(dcons, dtype=float).copy()
            dcons[clipped, :] = 0.0
        return consumption, dcons

    def _terminal_utility_with_deriv(self, consumption, dcons):
        log_growth = np.log(self.utility_model.growth_term)
        if self.utility_model.is_log_eis:
            continuation_log = self.utility_model.b * log_growth / (
                1.0 - self.utility_model.b
            )
        else:
            continuation_log = -np.log1p(
                -(self.utility_model.b / (1.0 - self.utility_model.b))
                * np.expm1(self.utility_model.r * log_growth)
            ) / self.utility_model.r
        multiplier = np.exp(continuation_log)
        return consumption * multiplier, dcons * multiplier

    def _intertemporal_aggregate_with_deriv(self, consumption, dcons,
                                            cert_equiv, dcert):
        u = self.utility_model._intertemporal_aggregate(consumption, cert_equiv)
        if self.utility_model.is_log_eis:
            du_dc = (1.0 - self.utility_model.b) * u / consumption
            du_dce = self.utility_model.b * u / cert_equiv
        else:
            du_dc = (
                u ** (1.0 - self.utility_model.r)
                * (1.0 - self.utility_model.b)
                * consumption ** (self.utility_model.r - 1.0)
            )
            du_dce = (
                u ** (1.0 - self.utility_model.r)
                * self.utility_model.b
                * cert_equiv ** (self.utility_model.r - 1.0)
            )
        du = du_dc[:, None] * dcons + du_dce[:, None] * dcert
        return u, du

    def _certain_equivalence_with_deriv(self, period, damage_period,
                                        utility_tree, d_utility):
        next_u = utility_tree.get_next_period_array(period)
        next_du = d_utility[period + self.period_len]
        if utility_tree.is_information_period(period):
            damage_nodes = self.tree.get_nodes_in_period(damage_period + 1)
            probs = self.tree.node_prob[damage_nodes[0]:damage_nodes[1] + 1]
            even_probs = probs[::2]
            odd_probs = probs[1::2]
            even_u = next_u[::2]
            odd_u = next_u[1::2]
            even_du = next_du[::2]
            odd_du = next_du[1::2]
            prob_sum = even_probs + odd_probs
            ave_u = (
                even_u**self.utility_model.a * even_probs
                + odd_u**self.utility_model.a * odd_probs
            ) / prob_sum
            cert_equiv = ave_u ** (1.0 / self.utility_model.a)
            ce_power = cert_equiv ** (1.0 - self.utility_model.a)
            even_weight = (
                ce_power * even_u ** (self.utility_model.a - 1.0)
                * even_probs / prob_sum
            )
            odd_weight = (
                ce_power * odd_u ** (self.utility_model.a - 1.0)
                * odd_probs / prob_sum
            )
            dcert = even_weight[:, None] * even_du + odd_weight[:, None] * odd_du
            return cert_equiv, dcert
        return next_u, next_du.copy()

    def _interpolated_consumption(self, period, damage_period, base_consumption,
                                  dbase_consumption, utility_tree, cons_tree,
                                  cost_tree, d_cons, d_cost, period_cost,
                                  dperiod_cost, diagnostics):
        next_period = period + self.period_len
        next_consumption = cons_tree.get_next_period_array(period)
        d_next_consumption = d_cons[next_period].copy()
        segment = period - utility_tree.decision_times[damage_period]
        interval = segment + utility_tree.subinterval_len
        alpha = segment / float(interval)

        if utility_tree.is_decision_period(next_period):
            if period < utility_tree.decision_times[-2]:
                next_cost = cost_tree[next_period]
                dnext_cost = d_cost[next_period]
                repeated_cost = np.repeat(period_cost, 2)
                drepeated_cost = np.repeat(dperiod_cost, 2, axis=0)
                numerator = 1.0 - repeated_cost
                denominator = 1.0 - next_cost
                factor = numerator / denominator
                dfactor = (
                    -drepeated_cost * denominator[:, None]
                    + numerator[:, None] * dnext_cost
                ) / (denominator[:, None] ** 2)
                previous_next_consumption = next_consumption.copy()
                next_consumption = next_consumption * factor
                d_next_consumption = (
                    factor[:, None] * d_next_consumption
                    + previous_next_consumption[:, None] * dfactor
                )
                next_consumption, d_next_consumption = self._clip_consumption(
                    next_consumption, d_next_consumption, diagnostics
                )

        if period < utility_tree.decision_times[-2]:
            repeated_base = np.repeat(base_consumption, 2)
            drepeated_base = np.repeat(dbase_consumption, 2, axis=0)
            return self._geometric_interpolation(
                repeated_base, drepeated_base, next_consumption,
                d_next_consumption, alpha, diagnostics
            )

        return self._geometric_interpolation(
            base_consumption, dbase_consumption, next_consumption,
            d_next_consumption, alpha, diagnostics
        )

    def _geometric_interpolation(self, base, dbase, nxt, dnxt, alpha, diagnostics):
        value = (nxt / base) ** alpha * base
        dvalue = value[:, None] * (
            alpha * dnxt / nxt[:, None]
            + (1.0 - alpha) * dbase / base[:, None]
        )
        return self._clip_consumption(value, dvalue, diagnostics)

    def _cost_period(self, m, period, is_last, diagnostics):
        nodes = self.tree.get_nodes_in_period(period)
        node_ids = np.arange(nodes[0], nodes[1] + 1)
        mitigation = m[node_ids]
        ave, dave = self._average_mitigation_period(m, period, is_last)
        cost = self.cost.cost(period, mitigation, ave)

        years_since_base_year = self.tree.decision_times[period]
        abs_year = self.tree.base_year + years_since_base_year
        exponent = abs_year - self.cost.anchor_year
        tech_base = 1.0 - (
            self.cost.tech_const + self.cost.tech_scale * ave
        ) / 100.0
        tech_term = tech_base ** exponent
        dtech_dave = (
            exponent * tech_base ** (exponent - 1.0)
            * (-self.cost.tech_scale / 100.0)
        )
        mitigation_cost = self.cost._raw_integrated_cost(mitigation)
        dmitigation_cost = self.cost._raw_marginal_cost(mitigation)

        dcost = (
            mitigation_cost[:, None] * dtech_dave[:, None] * dave
        ) / self.cost.cons_per_ton
        for row, node in enumerate(node_ids):
            dcost[row, node] += dmitigation_cost[row] * tech_term[row] / self.cost.cons_per_ton
        return cost, dcost

    def _damage_period(self, m, period, is_last, diagnostics):
        nodes = self.tree.get_num_nodes_period(period)
        damage = np.zeros(nodes)
        ddamage = self._zeros(nodes)
        for state in range(nodes):
            node = self.tree.get_node(period, state)
            damage[state], ddamage[state] = self._damage_node(
                m, node, is_last, diagnostics
            )
        return damage, ddamage

    def _damage_node(self, m, node, is_last, diagnostics):
        if getattr(self.damage, "zero_damage", False):
            return 0.0, np.zeros(self.n_vars)
        if node == 0:
            return 0.0, np.zeros(self.n_vars)
        period = self.tree.get_period(node)
        if is_last:
            period += 1
        cemit, dcemit = self._cumemit_at_node(m, node, is_last)
        conc, dconc = self._conc_at_node(m, node, is_last)
        interp_damage, slope, clipped, knot = self._interp_damage_with_slope(
            cemit, node, period
        )
        if clipped:
            diagnostics["num_damage_interp_clipped"] += 1
        if knot:
            diagnostics["num_damage_interp_knots"] += 1
        penalty = 1.0 / (1.0 + np.exp(0.05 * (conc - 200.0)))
        dpenalty_dconc = -0.05 * penalty * (1.0 - penalty)
        return interp_damage + penalty, slope * dcemit + dpenalty_dconc * dconc

    def _average_mitigation_period(self, m, period, is_last):
        nodes = self.tree.get_num_nodes_period(period)
        ave = np.zeros(nodes)
        dave = self._zeros(nodes)
        for state in range(nodes):
            node = self.tree.get_node(period, state)
            ave[state], dave[state] = self._average_mitigation_node(
                m, node, period, is_last
            )
        return ave, dave

    def _average_mitigation_node(self, m, node, period, is_last):
        if period == 0:
            return 0.0, np.zeros(self.n_vars)
        cemit, dcemit = self._cumemit_at_node(m, node, is_last)
        period_ind = self.damage.emit_baseline.dec_times_ind[period] + 1
        baseline_up_to_period = self.damage.emit_baseline.baseline_cumemit[:period_ind]
        denominator = (
            baseline_up_to_period[-1]
            - self.damage.emit_baseline.CUMEMIT_BASE_YEAR
        )
        ave = 1.0 - (
            cemit - self.damage.emit_baseline.CUMEMIT_BASE_YEAR
        ) / denominator
        return ave, -dcemit / denominator

    def _cumemit_at_node(self, m, node, is_last):
        flow, dflow, trunc_times = self._mitigated_flow_with_deriv(
            m, node, "gtco2", is_last
        )
        legacy_cumemit = self.damage.emit_baseline.CUMEMIT_BASE_YEAR + 1e-3 * (
            self.damage.emit_baseline._cumulative_trapezoid_preserve_jumps(
                flow, trunc_times
            )
        )
        dcumemit = self._cumulative_trapezoid_derivative(dflow, trunc_times) * 1e-3
        return float(legacy_cumemit[-1]), dcumemit[-1].copy()

    def _conc_at_node(self, m, node, is_last):
        if node == 0:
            return self.damage.climate.C_BASE_YEAR, np.zeros(self.n_vars)
        mitigated_baseline, dmitigated, trunc_times = self._mitigated_flow_with_deriv(
            m, node, "ppm", is_last
        )
        final_time = int(trunc_times[-1] - self.damage.climate.base_year)
        time = np.arange(-final_time, final_time, 1)
        time_after = time[final_time:]
        time_before = time[:final_time]
        source_x = trunc_times - self.damage.climate.base_year
        interp_bau = np.interp(time_after, source_x, mitigated_baseline)
        interp_deriv = self._interp_derivative(source_x, time_after, dmitigated)
        full_path = np.hstack((np.zeros_like(time_before), interp_bau))
        full_deriv = np.vstack((self._zeros(len(time_before)), interp_deriv))
        joos = self.damage.climate._make_joos_IRF(time_after, time_before)
        conc = np.convolve(full_path[1:], joos, mode="same")[-1]
        dconc = np.zeros(self.n_vars)
        for col in range(self.n_vars):
            dconc[col] = np.convolve(full_deriv[1:, col], joos, mode="same")[-1]
        preindustrial = (
            self.damage.climate.C_BASE_YEAR
            * (self.damage.climate.A_0 + self.damage.climate.A_1) ** (-1)
            * (
                self.damage.climate.A_0
                + self.damage.climate.A_1
                * np.exp(-time_after / self.damage.climate.TAU_1)
            )
        )
        return float(conc + preindustrial[-1]), dconc

    def _mitigated_flow_with_deriv(self, m, node, baseline, is_last):
        emit = self.damage.emit_baseline
        period = self.tree.get_period(node)
        if is_last:
            period += 1
        path = np.asarray(self.tree.get_path(node), dtype=int)
        trunc_times, raw_baseline, interval_indices = \
            emit._decision_interval_samples(period, baseline)
        mitigated = np.zeros_like(raw_baseline, dtype=np.float64)
        dmitigated = self._zeros(len(raw_baseline))
        if period:
            controlling_nodes = path[interval_indices]
            mitigated = raw_baseline * (1.0 - m[controlling_nodes])
            dmitigated[np.arange(len(raw_baseline)), controlling_nodes] = \
                -raw_baseline
        return mitigated, dmitigated, trunc_times

    def _cumulative_trapezoid_derivative(self, dflow, times):
        out = np.zeros_like(dflow, dtype=float)
        for idx in range(1, len(times)):
            dt = float(times[idx] - times[idx - 1])
            out[idx] = out[idx - 1] + 0.5 * dt * (
                dflow[idx] + dflow[idx - 1]
            )
        return out

    def _interp_derivative(self, source_x, target_x, source_deriv):
        if source_deriv.shape[0] == 0:
            return self._zeros(len(target_x))
        cols = [
            np.interp(target_x, source_x, source_deriv[:, col])
            for col in range(self.n_vars)
        ]
        return np.asarray(cols, dtype=float).T

    def _interp_damage_with_slope(self, cemit, node, period):
        damage = self.damage
        mit_cumemit_per = damage.mitigation_cumulative_emissions_knots(period)
        end_states = self.tree.reachable_end_states(node)
        probs = self.tree.final_states_prob[end_states]
        d_per = np.array([
            (probs * damage.d_rcomb[i, end_states, period - 1]).sum()
            / probs.sum()
            for i in range(damage.dnum)
        ], dtype=float)
        knot = bool(np.any(np.isclose(cemit, mit_cumemit_per, rtol=0.0, atol=1e-8)))
        if not np.isfinite(cemit):
            return float(d_per[-1]), 0.0, True, knot
        try:
            c_ind = np.where(mit_cumemit_per < cemit)[0][-1]
        except IndexError:
            if cemit <= mit_cumemit_per[0]:
                return 0.0, 0.0, True, knot
            if cemit >= mit_cumemit_per[-1]:
                return float(d_per[-1]), 0.0, True, knot
            raise
        try:
            cemit_less = mit_cumemit_per[c_ind]
            cemit_great = mit_cumemit_per[c_ind + 1]
        except IndexError:
            if c_ind == damage.dnum - 1:
                return float(d_per[-1]), 0.0, True, knot
            if c_ind == 0:
                return 0.0, 0.0, True, knot
            raise
        slope = (d_per[c_ind + 1] - d_per[c_ind]) / (cemit_great - cemit_less)
        interpolated = slope * (cemit - cemit_less) + d_per[c_ind]
        return float(interpolated), float(slope), False, knot


class EZAdjointObjective(ObjectiveWithGradient):
    """Reverse-mode adjoint for the production :class:`EZUtility` model.

    The forward pass records only primal arrays and local scalar derivatives.
    The reverse pass propagates utility costates chronologically and scatters
    emissions and concentration adjoints onto each node's short ancestor path.
    It never materializes a state-by-control Jacobian.
    """

    _PARITY_MODES = ("first", "always", "off")

    def __init__(self, utility, value_parity_tol=None, fail_on_value_mismatch=None,
                 value_parity_mode=None):
        self.utility_model = utility
        self.tree = utility.tree
        self.damage = utility.damage
        self.cost = utility.cost
        self.period_len = utility.period_len
        self.n_vars = int(self.tree.num_decision_nodes)
        self.value_parity_tol = float(
            os.environ.get(
                "ADJOINT_VALUE_PARITY_TOL",
                "1e-9" if value_parity_tol is None else str(value_parity_tol),
            )
        )
        if fail_on_value_mismatch is None:
            fail_on_value_mismatch = os.environ.get(
                "ADJOINT_FAIL_ON_VALUE_MISMATCH", "1"
            ) not in ("0", "false", "False", "no", "No")
        self.fail_on_value_mismatch = bool(fail_on_value_mismatch)
        self.value_parity_mode = str(
            value_parity_mode
            if value_parity_mode is not None
            else os.environ.get("ADJOINT_VALUE_PARITY_MODE", "first")
        ).strip().lower()
        if self.value_parity_mode not in self._PARITY_MODES:
            raise ValueError(
                "ADJOINT_VALUE_PARITY_MODE must be one of {}; got {!r}".format(
                    self._PARITY_MODES, self.value_parity_mode
                )
            )
        self._parity_checked = False
        self._state_lock = threading.Lock()
        self.last_diagnostics = {}
        self._response_kernels = {}

    def utility(self, m):
        return float(np.asarray(self.utility_model.utility(m)).reshape(-1)[0])

    def gradient(self, m):
        return self.value_and_gradient(m)[1]

    def value_and_gradient(self, m):
        m = np.asarray(m, dtype=float).reshape(-1)
        if len(m) != self.n_vars:
            raise ValueError(
                "mitigation length {} does not match tree decision nodes {}".format(
                    len(m), self.n_vars
                )
            )

        forward_start = time.perf_counter()
        taped_value, tape, diagnostics = self._forward_tape(m)
        forward_seconds = time.perf_counter() - forward_start
        reverse_start = time.perf_counter()
        gradient = self._reverse_tape(m, tape)
        reverse_seconds = time.perf_counter() - reverse_start

        parity_status = "skipped"
        legacy_value = np.nan
        abs_diff = np.nan
        rel_diff = np.nan
        if self._claim_parity_check():
            legacy_value = self.utility(m)
            abs_diff = abs(taped_value - legacy_value)
            rel_diff = abs_diff / max(1.0, abs(legacy_value))
            parity_status = "passed" if rel_diff <= self.value_parity_tol else "failed"
            if self.fail_on_value_mismatch and parity_status == "failed":
                raise RuntimeError(
                    "EZAdjointObjective value parity failed: legacy {:.12g}, "
                    "taped {:.12g}, rel diff {:.3g}".format(
                        legacy_value, taped_value, rel_diff
                    )
                )

        diagnostics.update({
            "gradient_backend": "reverse_adjoint",
            "adjoint_value": float(taped_value),
            "adjoint_legacy_value": float(legacy_value),
            "adjoint_value_abs_diff": float(abs_diff),
            "adjoint_value_rel_diff": float(rel_diff),
            "adjoint_value_parity_mode": self.value_parity_mode,
            "adjoint_value_parity_status": parity_status,
            "adjoint_forward_seconds": float(forward_seconds),
            "adjoint_reverse_seconds": float(reverse_seconds),
            "adjoint_total_seconds": float(forward_seconds + reverse_seconds),
            "adjoint_tape_bytes": int(self._estimate_bytes(tape)),
            "adjoint_has_state_control_matrix": bool(
                self._contains_state_control_matrix(tape)
            ),
        })
        with self._state_lock:
            self.last_diagnostics = diagnostics
        return float(taped_value), gradient

    def diagnostics(self):
        with self._state_lock:
            return dict(self.last_diagnostics)

    def _claim_parity_check(self):
        if self.value_parity_mode == "off":
            return False
        if self.value_parity_mode == "always":
            return True
        with self._state_lock:
            if self._parity_checked:
                return False
            self._parity_checked = True
            return True

    @classmethod
    def _estimate_bytes(cls, value):
        if isinstance(value, np.ndarray):
            return value.nbytes
        if hasattr(value, "tree") and isinstance(value.tree, dict):
            return cls._estimate_bytes(value.tree)
        if isinstance(value, dict):
            return sum(cls._estimate_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(cls._estimate_bytes(item) for item in value)
        return 0

    def _contains_state_control_matrix(self, value):
        if isinstance(value, np.ndarray):
            return bool(
                value.ndim >= 2
                and value.shape[0] > 1
                and value.shape[-1] == self.n_vars
            )
        if hasattr(value, "tree") and isinstance(value.tree, dict):
            return self._contains_state_control_matrix(value.tree)
        if isinstance(value, dict):
            return any(
                self._contains_state_control_matrix(item)
                for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(
                self._contains_state_control_matrix(item) for item in value
            )
        return False

    @staticmethod
    def _clip_positive(value):
        value = np.asarray(value, dtype=float).copy()
        clipped = value <= 0.0
        value[clipped] = 1e-18
        return value, clipped

    def _forward_tape(self, m):
        utility_tree = BigStorageTree(
            subinterval_len=self.period_len,
            decision_times=self.tree.decision_times,
        )
        cons_tree = BigStorageTree(
            subinterval_len=self.period_len,
            decision_times=self.tree.decision_times,
        )
        ce_tree = BigStorageTree(
            subinterval_len=self.period_len,
            decision_times=self.tree.decision_times,
        )
        cost_tree = SmallStorageTree(decision_times=self.tree.decision_times)
        diagnostics = {
            "num_damage_interp_clipped": 0,
            "num_damage_interp_knots": 0,
            "num_consumption_clipped": 0,
            "num_cost_kink_nodes": _num_cost_kink_nodes(self.cost, m),
            "num_overmitigation_nodes": int(np.sum(m > 1.0)),
        }
        tape = {
            "utility_tree": utility_tree,
            "cons_tree": cons_tree,
            "ce_tree": ce_tree,
            "cost_tree": cost_tree,
            "period_ops": {},
            "local_blocks": [],
            "terminal_block": None,
        }

        terminal_block = self._local_block(
            m, self.tree.num_periods, True, diagnostics
        )
        tape["local_blocks"].append(terminal_block)
        tape["terminal_block"] = terminal_block
        cost_tree.set_value(cost_tree.last_period, terminal_block["cost"])
        terminal_consumption = (
            self.utility_model.potential_cons[-1]
            * (1.0 - terminal_block["damage"])
        )
        terminal_consumption, terminal_clipped = self._clip_positive(
            terminal_consumption
        )
        diagnostics["num_consumption_clipped"] += int(np.sum(terminal_clipped))
        cons_tree.set_value(cons_tree.last_period, terminal_consumption)
        terminal_multiplier = self._terminal_multiplier()
        utility_tree.set_value(
            utility_tree.last_period, terminal_consumption * terminal_multiplier
        )
        tape["terminal_consumption_clipped"] = terminal_clipped
        tape["terminal_multiplier"] = terminal_multiplier

        active_block = None
        for period in utility_tree.periods[::-1][1:]:
            damage_period = utility_tree.between_decision_times(period)
            cert_equiv, ce_op = self._certain_equivalence_forward(
                period, damage_period, utility_tree
            )

            if utility_tree.is_decision_period(period + self.period_len):
                active_block = self._local_block(
                    m, damage_period, False, diagnostics
                )
                tape["local_blocks"].append(active_block)
                cost_key = cost_tree.index_below(period + self.period_len)
                cost_tree.set_value(cost_key, active_block["cost"])

            raw_base = (
                self.utility_model.potential_cons[damage_period]
                * (1.0 - active_block["damage"])
                * (1.0 - active_block["cost"])
            )
            base_consumption, base_clipped = self._clip_positive(raw_base)
            diagnostics["num_consumption_clipped"] += int(np.sum(base_clipped))
            consumption = base_consumption
            interpolation_op = None

            if not utility_tree.is_decision_period(period):
                consumption, interpolation_op = self._interpolation_forward(
                    period,
                    damage_period,
                    base_consumption,
                    utility_tree,
                    cons_tree,
                    cost_tree,
                    active_block,
                    diagnostics,
                )

            ce_tree.set_value(
                period,
                self.utility_model.b * cert_equiv**self.utility_model.r,
            )
            cons_tree.set_value(period, consumption)
            period_utility = self.utility_model._intertemporal_aggregate(
                consumption, cert_equiv
            )
            utility_tree.set_value(period, period_utility)
            tape["period_ops"][period] = {
                "damage_period": damage_period,
                "local_block": active_block,
                "base_consumption": base_consumption,
                "base_clipped": base_clipped,
                "interpolation": interpolation_op,
                "ce": cert_equiv,
                "ce_op": ce_op,
            }

        return (
            float(utility_tree[0].reshape(-1)[0]),
            tape,
            diagnostics,
        )

    def _terminal_multiplier(self):
        log_growth = np.log(self.utility_model.growth_term)
        if self.utility_model.is_log_eis:
            continuation_log = self.utility_model.b * log_growth / (
                1.0 - self.utility_model.b
            )
        else:
            continuation_log = -np.log1p(
                -(self.utility_model.b / (1.0 - self.utility_model.b))
                * np.expm1(self.utility_model.r * log_growth)
            ) / self.utility_model.r
        return float(np.exp(continuation_log))

    def _certain_equivalence_forward(self, period, damage_period, utility_tree):
        next_u = utility_tree.get_next_period_array(period)
        if utility_tree.is_information_period(period):
            damage_nodes = self.tree.get_nodes_in_period(damage_period + 1)
            probs = self.tree.node_prob[damage_nodes[0]:damage_nodes[1] + 1]
            even_probs = probs[::2]
            odd_probs = probs[1::2]
            even_u = next_u[::2]
            odd_u = next_u[1::2]
            prob_sum = even_probs + odd_probs
            ave_u = (
                even_u**self.utility_model.a * even_probs
                + odd_u**self.utility_model.a * odd_probs
            ) / prob_sum
            ce = ave_u ** (1.0 / self.utility_model.a)
            ce_power = ce ** (1.0 - self.utility_model.a)
            even_weight = (
                ce_power * even_u ** (self.utility_model.a - 1.0)
                * even_probs / prob_sum
            )
            odd_weight = (
                ce_power * odd_u ** (self.utility_model.a - 1.0)
                * odd_probs / prob_sum
            )
            return ce, {
                "information": True,
                "even_weight": even_weight,
                "odd_weight": odd_weight,
            }
        return next_u, {"information": False}

    def _interpolation_forward(self, period, damage_period, base_consumption,
                               utility_tree, cons_tree, cost_tree, active_block,
                               diagnostics):
        next_period = period + self.period_len
        next_consumption = cons_tree.get_next_period_array(period)
        original_next = next_consumption.copy()
        segment = period - utility_tree.decision_times[damage_period]
        interval = segment + utility_tree.subinterval_len
        alpha = segment / float(interval)
        cost_adjustment = None

        if utility_tree.is_decision_period(next_period):
            if period < utility_tree.decision_times[-2]:
                next_cost = cost_tree[next_period]
                repeated_cost = np.repeat(active_block["cost"], 2)
                factor = (1.0 - repeated_cost) / (1.0 - next_cost)
                adjusted_next = next_consumption * factor
                adjusted_next, adjusted_clipped = self._clip_positive(adjusted_next)
                diagnostics["num_consumption_clipped"] += int(
                    np.sum(adjusted_clipped)
                )
                next_consumption = adjusted_next
                cost_adjustment = {
                    "original_next": original_next,
                    "next_cost": next_cost,
                    "repeated_cost": repeated_cost,
                    "factor": factor,
                    "clipped": adjusted_clipped,
                    "next_cost_key": next_period,
                }

        repeat_base = period < utility_tree.decision_times[-2]
        interpolation_base = (
            np.repeat(base_consumption, 2) if repeat_base else base_consumption
        )
        value = (
            (next_consumption / interpolation_base) ** alpha
            * interpolation_base
        )
        value, clipped = self._clip_positive(value)
        diagnostics["num_consumption_clipped"] += int(np.sum(clipped))
        return value, {
            "alpha": alpha,
            "base": interpolation_base,
            "next": next_consumption,
            "repeat_base": repeat_base,
            "clipped": clipped,
            "cost_adjustment": cost_adjustment,
            "next_period": next_period,
        }

    def _local_block(self, m, period, is_last, diagnostics):
        node_count = self.tree.get_num_nodes_period(period)
        node_range = self.tree.get_nodes_in_period(period)
        node_ids = np.arange(node_range[0], node_range[1] + 1)
        mitigation = m[node_ids]
        damage_values = np.zeros(node_count)
        average_values = np.zeros(node_count)
        states = []

        for state in range(node_count):
            node = self.tree.get_node(period, state)
            if node == 0:
                damage_values[state] = 0.0
                average_values[state] = 0.0
                states.append({
                    "path": np.zeros(0, dtype=int),
                    "cemit_coeff": np.zeros(0),
                    "conc_coeff": np.zeros(0),
                    "damage_slope": 0.0,
                    "penalty_slope": 0.0,
                    "average_denom": np.inf,
                })
                continue

            effective_period = self.tree.get_period(node) + (1 if is_last else 0)
            path = np.asarray(self.tree.get_path(node)[:effective_period], dtype=int)
            kernel = self._response_kernel(effective_period)
            mitigation_path = m[path]
            mitigated_gtco2 = np.zeros_like(kernel["baseline_gtco2"])
            for path_index, mitigation_value in enumerate(mitigation_path):
                low, high = kernel["interval_bounds"][path_index]
                mitigated_gtco2[low:high] = (
                    kernel["baseline_gtco2"][low:high]
                    * (1.0 - mitigation_value)
                )
            cemit = (
                self.damage.emit_baseline.CUMEMIT_BASE_YEAR
                + 1e-3
                * self.damage.emit_baseline._cumulative_trapezoid_preserve_jumps(
                    mitigated_gtco2, kernel["trunc_times"]
                )[-1]
            )
            conc = (
                kernel["conc_intercept"]
                + np.dot(kernel["conc_coeff"], mitigation_path)
            )
            if getattr(self.damage, "zero_damage", False):
                slope = 0.0
                penalty_slope = 0.0
                damage_values[state] = 0.0
            else:
                interp_damage, slope, clipped, knot = self._interp_damage_with_slope(
                    cemit, node, effective_period
                )
                diagnostics["num_damage_interp_clipped"] += int(clipped)
                diagnostics["num_damage_interp_knots"] += int(knot)
                penalty = 1.0 / (1.0 + np.exp(0.05 * (conc - 200.0)))
                penalty_slope = -0.05 * penalty * (1.0 - penalty)
                damage_values[state] = interp_damage + penalty

            period_ind = self.damage.emit_baseline.dec_times_ind[effective_period] + 1
            baseline_up_to_period = (
                self.damage.emit_baseline.baseline_cumemit[:period_ind]
            )
            denominator = (
                baseline_up_to_period[-1]
                - self.damage.emit_baseline.CUMEMIT_BASE_YEAR
            )
            average_values[state] = 1.0 - (
                cemit - self.damage.emit_baseline.CUMEMIT_BASE_YEAR
            ) / denominator
            states.append({
                "path": path,
                "cemit_coeff": kernel["cemit_coeff"],
                "conc_coeff": kernel["conc_coeff"],
                "damage_slope": float(slope),
                "penalty_slope": float(penalty_slope),
                "average_denom": float(denominator),
            })

        cost_values = self.cost.cost(period, mitigation, average_values)
        return {
            "period": period,
            "is_last": bool(is_last),
            "node_ids": node_ids,
            "mitigation": mitigation,
            "average": average_values,
            "damage": damage_values,
            "cost": cost_values,
            "states": states,
            "bar_damage": np.zeros_like(damage_values),
            "bar_cost": np.zeros_like(cost_values),
        }

    def _response_kernel(self, effective_period):
        effective_period = int(effective_period)
        if effective_period in self._response_kernels:
            return self._response_kernels[effective_period]

        emit = self.damage.emit_baseline
        trunc_times, flow_gtco2, interval_indices = \
            emit._decision_interval_samples(effective_period, "gtco2")
        ppm_times, flow_ppm, ppm_interval_indices = \
            emit._decision_interval_samples(effective_period, "ppm")
        if not np.array_equal(trunc_times, ppm_times) or not np.array_equal(
            interval_indices, ppm_interval_indices
        ):
            raise RuntimeError("emissions interval samples are inconsistent")
        conc_intercept = self._concentration_from_flow(flow_ppm, trunc_times)
        cemit_coeff = np.zeros(effective_period)
        conc_coeff = np.zeros(effective_period)
        interval_bounds = []

        for path_index in range(effective_period):
            positions = np.flatnonzero(interval_indices == path_index)
            low = int(positions[0])
            high = int(positions[-1]) + 1
            interval_bounds.append((low, high))
            d_gtco2 = np.zeros_like(flow_gtco2)
            d_ppm = np.zeros_like(flow_ppm)
            d_gtco2[low:high] = -flow_gtco2[low:high]
            d_ppm[low:high] = -flow_ppm[low:high]
            # Match the production float64 trapezoid convention used by the
            # primal cumulative-emissions integral.
            cemit_coeff[path_index] = 1e-3 * np.sum(
                0.5 * np.diff(trunc_times)
                * (d_gtco2[:-1] + d_gtco2[1:])
            )
            conc_coeff[path_index] = self._concentration_linear_response(
                d_ppm, trunc_times
            )

        kernel = {
            "conc_intercept": float(conc_intercept),
            "cemit_coeff": cemit_coeff,
            "conc_coeff": conc_coeff,
            "baseline_gtco2": flow_gtco2,
            "trunc_times": trunc_times,
            "interval_bounds": interval_bounds,
        }
        self._response_kernels[effective_period] = kernel
        return kernel

    def _concentration_from_flow(self, flow, trunc_times):
        final_time = int(trunc_times[-1] - self.damage.climate.base_year)
        if final_time <= 0:
            return float(self.damage.climate.C_BASE_YEAR)
        time = np.arange(-final_time, final_time, 1)
        time_after = time[final_time:]
        time_before = time[:final_time]
        source_x = trunc_times - self.damage.climate.base_year
        interp_flow = np.interp(time_after, source_x, flow)
        full_path = np.hstack((np.zeros_like(time_before), interp_flow))
        joos = self.damage.climate._make_joos_IRF(time_after, time_before)
        concentration = np.convolve(full_path[1:], joos, mode="same")[-1]
        preindustrial = (
            self.damage.climate.C_BASE_YEAR
            * (self.damage.climate.A_0 + self.damage.climate.A_1) ** (-1)
            * (
                self.damage.climate.A_0
                + self.damage.climate.A_1
                * np.exp(-time_after / self.damage.climate.TAU_1)
            )
        )
        return float(concentration + preindustrial[-1])

    def _concentration_linear_response(self, flow, trunc_times):
        final_time = int(trunc_times[-1] - self.damage.climate.base_year)
        if final_time <= 0:
            return 0.0
        time = np.arange(-final_time, final_time, 1)
        time_after = time[final_time:]
        time_before = time[:final_time]
        source_x = trunc_times - self.damage.climate.base_year
        interp_flow = np.interp(time_after, source_x, flow)
        full_path = np.hstack((np.zeros_like(time_before), interp_flow))
        joos = self.damage.climate._make_joos_IRF(time_after, time_before)
        return float(np.convolve(full_path[1:], joos, mode="same")[-1])

    def _interp_damage_with_slope(self, cemit, node, period):
        damage = self.damage
        mit_cumemit_per = damage.mitigation_cumulative_emissions_knots(period)
        end_states = self.tree.reachable_end_states(node)
        probs = self.tree.final_states_prob[end_states]
        d_per = np.array([
            (probs * damage.d_rcomb[i, end_states, period - 1]).sum()
            / probs.sum()
            for i in range(damage.dnum)
        ], dtype=float)
        knot = bool(np.any(np.isclose(
            cemit, mit_cumemit_per, rtol=0.0, atol=1e-8
        )))
        if not np.isfinite(cemit):
            return float(d_per[-1]), 0.0, True, knot
        candidates = np.where(mit_cumemit_per < cemit)[0]
        if not len(candidates):
            if cemit <= mit_cumemit_per[0]:
                return 0.0, 0.0, True, knot
            if cemit >= mit_cumemit_per[-1]:
                return float(d_per[-1]), 0.0, True, knot
            raise RuntimeError("Unable to locate damage interpolation interval")
        c_ind = int(candidates[-1])
        if c_ind >= damage.dnum - 1:
            return float(d_per[-1]), 0.0, True, knot
        cemit_less = mit_cumemit_per[c_ind]
        cemit_great = mit_cumemit_per[c_ind + 1]
        slope = (
            (d_per[c_ind + 1] - d_per[c_ind])
            / (cemit_great - cemit_less)
        )
        interpolated = slope * (cemit - cemit_less) + d_per[c_ind]
        return float(interpolated), float(slope), False, knot

    def _reverse_tape(self, m, tape):
        utility_tree = tape["utility_tree"]
        cons_tree = tape["cons_tree"]
        cost_tree = tape["cost_tree"]
        bar_utility = {
            period: np.zeros_like(values, dtype=float)
            for period, values in utility_tree.tree.items()
        }
        bar_consumption = {
            period: np.zeros_like(values, dtype=float)
            for period, values in cons_tree.tree.items()
        }
        bar_cost_tree = {
            period: np.zeros_like(values, dtype=float)
            for period, values in cost_tree.tree.items()
        }
        bar_utility[utility_tree.periods[0]][0] = 1.0

        for period in utility_tree.periods[:-1]:
            op = tape["period_ops"][period]
            utility = utility_tree[period]
            consumption = cons_tree[period]
            cert_equiv = op["ce"]
            utility_bar = bar_utility[period]
            if self.utility_model.is_log_eis:
                du_dc = (
                    (1.0 - self.utility_model.b) * utility / consumption
                )
                du_dce = self.utility_model.b * utility / cert_equiv
            else:
                du_dc = (
                    utility ** (1.0 - self.utility_model.r)
                    * (1.0 - self.utility_model.b)
                    * consumption ** (self.utility_model.r - 1.0)
                )
                du_dce = (
                    utility ** (1.0 - self.utility_model.r)
                    * self.utility_model.b
                    * cert_equiv ** (self.utility_model.r - 1.0)
                )
            bar_consumption[period] += utility_bar * du_dc
            bar_ce = utility_bar * du_dce
            next_period = period + self.period_len
            ce_op = op["ce_op"]
            if ce_op["information"]:
                bar_utility[next_period][::2] += (
                    bar_ce * ce_op["even_weight"]
                )
                bar_utility[next_period][1::2] += (
                    bar_ce * ce_op["odd_weight"]
                )
            else:
                bar_utility[next_period] += bar_ce

            base_bar = self._reverse_consumption_op(
                bar_consumption[period],
                op,
                bar_consumption,
                bar_cost_tree,
            )
            block = op["local_block"]
            potential = self.utility_model.potential_cons[op["damage_period"]]
            block["bar_damage"] += (
                -potential * (1.0 - block["cost"]) * base_bar
            )
            block["bar_cost"] += (
                -potential * (1.0 - block["damage"]) * base_bar
            )

        terminal_block = tape["terminal_block"]
        terminal_period = utility_tree.last_period
        terminal_bar = (
            bar_consumption[terminal_period]
            + bar_utility[terminal_period] * tape["terminal_multiplier"]
        )
        terminal_bar[tape["terminal_consumption_clipped"]] = 0.0
        terminal_block["bar_damage"] += (
            -self.utility_model.potential_cons[-1] * terminal_bar
        )

        for block in tape["local_blocks"]:
            period_key = (
                cost_tree.last_period if block["is_last"]
                else self.tree.decision_times[block["period"]]
            )
            if period_key in bar_cost_tree:
                block["bar_cost"] += bar_cost_tree[period_key]

        gradient = np.zeros(self.n_vars, dtype=float)
        for block in tape["local_blocks"]:
            self._reverse_local_block(m, block, gradient)
        return gradient

    def _reverse_consumption_op(self, output_bar, op, bar_consumption,
                                bar_cost_tree):
        base_bar = np.zeros_like(op["base_consumption"])
        interpolation = op["interpolation"]
        if interpolation is None:
            base_bar += output_bar
        else:
            active_bar = np.asarray(output_bar, dtype=float).copy()
            active_bar[interpolation["clipped"]] = 0.0
            value = (
                (interpolation["next"] / interpolation["base"])
                ** interpolation["alpha"]
                * interpolation["base"]
            )
            interpolation_base_bar = (
                active_bar
                * (1.0 - interpolation["alpha"])
                * value / interpolation["base"]
            )
            next_bar = (
                active_bar
                * interpolation["alpha"]
                * value / interpolation["next"]
            )
            if interpolation["repeat_base"]:
                base_bar += interpolation_base_bar.reshape(-1, 2).sum(axis=1)
            else:
                base_bar += interpolation_base_bar

            adjustment = interpolation["cost_adjustment"]
            if adjustment is None:
                bar_consumption[interpolation["next_period"]] += next_bar
            else:
                next_bar[adjustment["clipped"]] = 0.0
                bar_consumption[interpolation["next_period"]] += (
                    next_bar * adjustment["factor"]
                )
                factor_bar = next_bar * adjustment["original_next"]
                current_cost_bar_repeated = (
                    -factor_bar / (1.0 - adjustment["next_cost"])
                )
                op["local_block"]["bar_cost"] += (
                    current_cost_bar_repeated.reshape(-1, 2).sum(axis=1)
                )
                bar_cost_tree[adjustment["next_cost_key"]] += (
                    factor_bar
                    * (1.0 - adjustment["repeated_cost"])
                    / (1.0 - adjustment["next_cost"])**2
                )

        base_bar[op["base_clipped"]] = 0.0
        return base_bar

    def _reverse_local_block(self, m, block, gradient):
        period = block["period"]
        average = block["average"]
        mitigation = block["mitigation"]
        cost_bar = block["bar_cost"]

        years_since_base_year = self.tree.decision_times[period]
        abs_year = self.tree.base_year + years_since_base_year
        exponent = abs_year - self.cost.anchor_year
        tech_base = 1.0 - (
            self.cost.tech_const + self.cost.tech_scale * average
        ) / 100.0
        tech_term = tech_base**exponent
        dtech_daverage = (
            exponent * tech_base ** (exponent - 1.0)
            * (-self.cost.tech_scale / 100.0)
        )
        mitigation_cost = self.cost._raw_integrated_cost(mitigation)
        dmitigation_cost = self.cost._raw_marginal_cost(mitigation)
        gradient[block["node_ids"]] += (
            cost_bar * dmitigation_cost * tech_term / self.cost.cons_per_ton
        )
        average_bar = (
            cost_bar * mitigation_cost * dtech_daverage
            / self.cost.cons_per_ton
        )

        for state, state_data in enumerate(block["states"]):
            path = state_data["path"]
            if not len(path):
                continue
            cemit_bar = (
                block["bar_damage"][state] * state_data["damage_slope"]
                - average_bar[state] / state_data["average_denom"]
            )
            conc_bar = (
                block["bar_damage"][state] * state_data["penalty_slope"]
            )
            gradient[path] += (
                cemit_bar * state_data["cemit_coeff"]
                + conc_bar * state_data["conc_coeff"]
            )

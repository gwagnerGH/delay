"""Smooth-backstop, flat-SSP2, PCHIP, five-year-accounting EZDelay model.

The existing EZDelay adjoint handles the climate and preference derivatives;
the flow objective below adds the five-year abatement-cost derivatives.
"""

from functools import lru_cache
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import erf, hyp1f1

from .adjoint_objective import EZAdjointObjective
from .config.revised_baseline import BASELINE
from .climate import BPWClimate
from .cost import BPWCost
from .damage import BPWDamage
from .emit_baseline import BPWEmissionBaseline
from .storage_tree import BigStorageTree
from .tree import TreeModel
from .utility import EZUtility


class FlatAfter2100Emissions(BPWEmissionBaseline):
    """Follow SSP2 through 2100, then hold its annual flow fixed."""

    def _ssp_baseline_input_series(self):
        # The inherited reader resolves data/ against the current directory.
        # The revised default uses a repository-relative input from any cwd.
        source = Path(__file__).resolve().parents[1] / "data" / "SSP_baselines.csv"
        with source.open(newline="") as handle:
            rows = list(csv.reader(handle))
        if not 1 <= self.baseline_num <= 5:
            raise ValueError("Invalid baseline_num; expected SSP1 through SSP5")
        years = np.asarray(rows[0][1:], dtype=int)
        annual_emissions = np.asarray(rows[self.baseline_num][1:], dtype=float)
        return years, annual_emissions

    def _make_extension(self, time, baseline_gtco2_2100, extension_year):
        super()._make_extension(time, baseline_gtco2_2100, extension_year)
        level = float(np.interp(2100, time, baseline_gtco2_2100))
        self.baseline_gtco2[self.times > 2100] = level


class SmoothBackstopCost(BPWCost):
    """C1 removal curve with optional curvature above full abatement."""

    @property
    def cost_formulation(self):
        return "smooth_removal_curve_2500_curvature_v2"

    def __init__(self, *args, backstop_curvature=1.0, **kwargs):
        if not np.isfinite(backstop_curvature) or backstop_curvature <= 0:
            raise ValueError("backstop_curvature must be finite and positive")
        super().__init__(*args, **kwargs)
        self.tau_0 = 58.9
        self.power = 2.3
        self.backstop_curvature = float(backstop_curvature)
        self.removal_asymptote = 2500.0
        self.cost_at_one = self.tau_0*np.expm1(self.power)
        self.removal_amplitude = self.removal_asymptote-self.cost_at_one
        # Matching the first derivative also preserves the join when curvature varies.
        self.removal_linear_rate = (self.tau_0*self.power*np.exp(self.power)
                                    /self.removal_amplitude)

    def _unshifted_raw_marginal_cost(self, mitigation):
        x = np.maximum(np.asarray(mitigation, dtype=float), 0.0)
        u = np.maximum(x-1.0, 0.0)
        return np.where(x <= 1.0, self.tau_0*np.expm1(self.power*x),
                        self.removal_asymptote-self.removal_amplitude*np.exp(
                            -self.removal_linear_rate*u-self.backstop_curvature*u*u))

    def _unshifted_raw_integrated_cost(self, mitigation):
        x = np.maximum(np.asarray(mitigation, dtype=float), 0.0)
        u = np.maximum(x-1.0, 0.0)
        ordinary = self.tau_0*(np.expm1(self.power*np.minimum(x, 1.0))/self.power
                               -np.minimum(x, 1.0))
        theta = self.backstop_curvature
        root = np.sqrt(theta)
        offset = self.removal_linear_rate/(2.0*root)
        gaussian = (np.exp(offset*offset)*np.sqrt(np.pi)/(2.0*root)
                    *(erf(root*u+offset)-erf(offset)))
        return (ordinary+self.removal_asymptote*u
                -self.removal_amplitude*gaussian)

    def _raw_marginal_cost(self, mitigation):
        """Shift the entire revised MAC, including the smooth removal branch."""
        if self.mac_horizontal_shift == 0.0 and self.mac_vertical_shift == 0.0:
            return self._unshifted_raw_marginal_cost(mitigation)
        m = np.asarray(mitigation, dtype=float)
        shifted = (self._unshifted_raw_marginal_cost(
            np.maximum(m, 0.0)+self.mac_horizontal_shift)
            + self.mac_vertical_shift)
        return np.where(m >= 0.0, shifted, 0.0)

    def _raw_integrated_cost(self, mitigation):
        """Integral of MAC_base(m+h)+v from zero, with cost(0) exactly zero."""
        if self.mac_horizontal_shift == 0.0 and self.mac_vertical_shift == 0.0:
            return self._unshifted_raw_integrated_cost(mitigation)
        m = np.maximum(np.asarray(mitigation, dtype=float), 0.0)
        return (self._unshifted_raw_integrated_cost(m+self.mac_horizontal_shift)
                -self._unshifted_raw_integrated_cost(self.mac_horizontal_shift)
                +self.mac_vertical_shift*m)


class CubicBackstopCost(SmoothBackstopCost):
    """Optional sensitivity: exponential MAC plus gamma*(m-1)^3 in removal."""

    @property
    def cost_formulation(self):
        return "cubic_removal_sensitivity"

    def __init__(self, *args, gamma, **kwargs):
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)

    def _raw_marginal_cost(self, mitigation):
        m = np.maximum(np.asarray(mitigation, dtype=float), 0.0)
        return 58.9*np.expm1(2.3*m) + self.gamma*np.maximum(m-1.0, 0.0)**3

    def _raw_integrated_cost(self, mitigation):
        m = np.maximum(np.asarray(mitigation, dtype=float), 0.0)
        return (58.9*(np.expm1(2.3*m)/2.3-m)
                + self.gamma*np.maximum(m-1.0, 0.0)**4/4.0)


class ExponentialCubicBackstopCost(CubicBackstopCost):
    """Steep sensitivity: ordinary MAC plus gamma*u^3*exp(1000*u), u=(m-1)+.

    The integrated term is gamma*u^4/4 * 1F1(4;5;1000*u), which gives
    exactly this MAC on differentiation. Cluster jobs use an m=1.05 numerical
    guard to keep the exponential finite; the guard must be checked afterwards.
    """

    @property
    def cost_formulation(self):
        return "exponential_cubic_removal_sensitivity"

    def _raw_marginal_cost(self, mitigation):
        m = np.maximum(np.asarray(mitigation, dtype=float), 0.0)
        u = np.maximum(m-1.0, 0.0)
        return 58.9*np.expm1(2.3*m) + self.gamma*u**3*np.exp(1000.0*u)

    def _raw_integrated_cost(self, mitigation):
        m = np.maximum(np.asarray(mitigation, dtype=float), 0.0)
        u = np.maximum(m-1.0, 0.0)
        return (58.9*(np.expm1(2.3*m)/2.3-m)
                + self.gamma*u**4/4.0*hyp1f1(4.0, 5.0, 1000.0*u))


class PchipDamage(BPWDamage):
    """Smooth interpolation of the same saved damage-simulation grid."""

    @lru_cache(maxsize=None)
    def _pchip_curve(self, node, period):
        knots = np.asarray(self.mitigation_cumulative_emissions_knots(period), float)
        end_states = self.tree.reachable_end_states(node)
        probs = np.asarray(self.tree.final_states_prob[end_states], float)
        values = np.array([float((probs*self.d_rcomb[i, end_states, period-1]).sum()
                                  / probs.sum()) for i in range(self.dnum)])
        if not np.all(np.diff(knots) > 0) or not np.all(np.isfinite(values)):
            raise ValueError("Invalid PCHIP damage grid")
        return knots, values, PchipInterpolator(knots, values, extrapolate=False)

    def pchip_value_slope(self, cemit, node, period):
        knots, values, curve = self._pchip_curve(int(node), int(period))
        knot = bool(np.any(np.isclose(cemit, knots, rtol=0.0, atol=1e-8)))
        if not np.isfinite(cemit):
            return float(values[-1]), 0.0, True, knot
        if cemit <= knots[0]:
            return 0.0, 0.0, True, knot
        if cemit >= knots[-1]:
            return float(values[-1]), 0.0, True, knot
        return float(curve(cemit)), float(curve.derivative()(cemit)), False, knot

    def _get_interp_damage(self, cemit, node, period):
        return self.pchip_value_slope(cemit, node, period)[0]


class _ZeroCost:
    def __init__(self, original):
        self.original = original

    def cost(self, period, mitigation, average):
        return np.zeros_like(np.asarray(mitigation, dtype=float))

    def __getattr__(self, name):
        return getattr(self.original, name)


class _ClimateAdjoint(EZAdjointObjective):
    def _interp_damage_with_slope(self, cemit, node, period):
        return self.damage.pchip_value_slope(cemit, node, period)


def _climate_component_gradients(tape, n):
    """Split the climate adjoint into warming damages and low-GHG adjustment.

    This reads the same reverse-mode damage weights used by the optimizer;
    it holds the policy and continuation valuation fixed. The components
    sum to the total climate gradient, without extra finite differences.
    """
    warming = np.zeros(n, dtype=float)
    low_ghg = np.zeros(n, dtype=float)
    for block in tape["local_blocks"]:
        for state, state_data in enumerate(block["states"]):
            path = state_data["path"]
            if len(path):
                weight = block["bar_damage"][state]
                warming[path] += (weight*state_data["damage_slope"]
                                  *state_data["cemit_coeff"])
                if "tip_grad_path" in state_data:
                    warming[path] += weight*state_data["tip_grad_path"]
                low_ghg[path] += (weight*state_data["penalty_slope"]
                                  *state_data["conc_coeff"])
    return warming, low_ghg


class FiveYearFlowObjective:
    """EZ welfare and exact gradient with physical costs at each utility date.

    The original zero-cost consumption interpolation remains unchanged. Costs
    are multiplicative and exogenous technology improves at each five-year
    consumption date; learning is off. The inherited adjoint differentiates
    climate and cooling; this class reverse-differentiates preferences and
    the five-year cost shares.
    """

    def __init__(self, tree, damage, cost, emissions, *, reference_consumption=86252.0,
                 period_len=5.0, eis=BASELINE.eis, ra=BASELINE.ra, time_pref=BASELINE.prtp,
                 cons_growth=BASELINE.growth, common_damage_path=None,
                 cost_floor=0.0,
                 terminal_growth_mode=BASELINE.terminal_growth_mode,
                 terminal_growth=BASELINE.terminal_growth,
                 terminal_log_mean=BASELINE.terminal_log_mean,
                 terminal_log_sigma=BASELINE.terminal_log_sigma):
        if cost.tech_scale != 0:
            raise ValueError("The revised objective requires zero endogenous learning")
        if not isinstance(damage, PchipDamage):
            raise TypeError("The revised objective requires PchipDamage")
        if not np.isfinite(cost_floor) or not 0.0 <= cost_floor < 1.0:
            raise ValueError("cost_floor must be finite and in [0, 1)")
        self.cost_floor = float(cost_floor)
        self.tree, self.damage, self.cost, self.emit = tree, damage, cost, emissions
        self.zero = EZUtility(tree, damage, _ZeroCost(cost), period_len,
                              eis, ra, time_pref, cons_growth,
                              terminal_growth_mode=terminal_growth_mode,
                              terminal_growth=terminal_growth,
                              terminal_log_mean=terminal_log_mean,
                              terminal_log_sigma=terminal_log_sigma)
        self.zero._terminal_multiplier()  # fail before optimizing an inadmissible tail
        if common_damage_path is None:
            self.climate_adjoint = _ClimateAdjoint(self.zero, value_parity_mode="off")
        else:
            from .calendar_climate import CommonCalendarClimateAdjoint
            self.climate_adjoint = CommonCalendarClimateAdjoint(
                self.zero, common_damage_path)
        self.common_damage_path = common_damage_path
        self.period_len = period_len
        self.last_diagnostics = {}
        self._flow_factors = {}
        for year in np.arange(0, tree.decision_times[-1], period_len):
            period = int(np.searchsorted(tree.decision_times, year, side="right")-1)
            calendar_year = tree.base_year+year
            technology = self.technology_multiplier(calendar_year)
            emissions_level = float(np.interp(tree.base_year+year,
                emissions.times, emissions.baseline_gtco2))
            growth = (1.0+cons_growth)**float(year)
            self._flow_factors[int(year)] = (period,
                emissions_level/reference_consumption*technology/growth)

    def technology_multiplier(self, calendar_year):
        original = (1.0-self.cost.tech_const/100.0)**(
            calendar_year-self.cost.anchor_year)
        return (original if calendar_year < self.cost.anchor_year else
                self.cost_floor+(1.0-self.cost_floor)*original)

    def _forward(self, m):
        _, tape, diagnostics = self.climate_adjoint._forward_tape(m)
        base = tape["cons_tree"]
        consumption = BigStorageTree(self.period_len, self.tree.decision_times)
        shares = {}
        for year in consumption.periods[:-1]:
            period, factor = self._flow_factors[int(year)]
            first, last = self.tree.get_nodes_in_period(period)
            share = factor*self.cost._raw_integrated_cost(m[first:last+1])
            if len(base[year]) == 2*len(share):
                share = np.repeat(share, 2)
            if len(base[year]) != len(share):
                raise RuntimeError("Five-year cost states do not match consumption")
            flow = base[year]*(1.0-share)
            if np.any(flow <= 0) or not np.all(np.isfinite(flow)):
                raise FloatingPointError("Nonpositive five-year consumption")
            shares[year] = share
            consumption.set_value(year, flow)
        consumption.set_value(consumption.last_period, base.last.copy())

        utility = BigStorageTree(self.period_len, self.tree.decision_times)
        multiplier = self.climate_adjoint._terminal_multiplier()
        utility.set_value(utility.last_period, consumption.last*multiplier)
        ce_ops = {}
        for year in utility.periods[::-1][1:]:
            period = utility.between_decision_times(year)
            ce, ce_op = self.climate_adjoint._certain_equivalence_forward(
                year, period, utility)
            ce_ops[year] = (ce, ce_op)
            utility.set_value(year, self.zero._intertemporal_aggregate(
                consumption[year], ce))
        return float(utility[0][0]), tape, consumption, utility, shares, ce_ops, diagnostics

    def utility(self, mitigation):
        return self._forward(np.asarray(mitigation, dtype=float).reshape(-1))[0]

    def delay_window_compensated_utility(self, mitigation, delay_years, factor):
        """Value delayed policy with consumption scaled before policy re-entry.

        The same proportional increase applies at every five-year utility
        date strictly before the delay ends, as in EZDelay's delay-window DWL.
        Climate, mitigation, and costs are held fixed; only consumption used by
        preferences is adjusted for this welfare-equivalent calculation.
        """
        if delay_years < 0 or factor <= 0:
            raise ValueError("Require nonnegative delay and positive factor")
        _, _, consumption, _, _, _, _ = self._forward(
            np.asarray(mitigation, dtype=float).reshape(-1))
        for year in consumption.periods[:-1]:
            if year < delay_years:
                consumption.set_value(year, consumption[year]*factor)
        utility = BigStorageTree(self.period_len, self.tree.decision_times)
        utility.set_value(utility.last_period,
            consumption.last*self.climate_adjoint._terminal_multiplier())
        for year in utility.periods[::-1][1:]:
            period = utility.between_decision_times(year)
            ce, _ = self.climate_adjoint._certain_equivalence_forward(
                year, period, utility)
            utility.set_value(year, self.zero._intertemporal_aggregate(
                consumption[year], ce))
        return float(utility[0][0])

    def gradient(self, mitigation):
        return self.value_and_gradient(mitigation)[1]

    def value_and_gradient(self, mitigation):
        m = np.asarray(mitigation, dtype=float).reshape(-1)
        if len(m) != self.tree.num_decision_nodes:
            raise ValueError("Mitigation vector has wrong length")
        value, tape, cons, utility, shares, ce_ops, diagnostics = self._forward(m)
        bar_u = {t: np.zeros_like(v) for t, v in utility.tree.items()}
        bar_cons = {t: np.zeros_like(v) for t, v in cons.tree.items()}
        bar_u[0][0] = 1.0
        for year in utility.periods[:-1]:
            ce, ce_op = ce_ops[year]
            u, c, weight = utility[year], cons[year], bar_u[year]
            if self.zero.is_log_eis:
                du_dc = (1.0-self.zero.b)*u/c
                du_dce = self.zero.b*u/ce
            else:
                du_dc = (u**(1.0-self.zero.r)*(1.0-self.zero.b)
                         *c**(self.zero.r-1.0))
                du_dce = (u**(1.0-self.zero.r)*self.zero.b
                          *ce**(self.zero.r-1.0))
            bar_cons[year] += weight*du_dc
            bar_ce = weight*du_dce
            next_year = year+self.period_len
            if ce_op["information"]:
                bar_u[next_year][::2] += bar_ce*ce_op["even_weight"]
                bar_u[next_year][1::2] += bar_ce*ce_op["odd_weight"]
            else:
                bar_u[next_year] += bar_ce
        bar_cons[cons.last_period] += (
            bar_u[utility.last_period]*self.climate_adjoint._terminal_multiplier())

        base = tape["cons_tree"]
        bar_base = {t: np.zeros_like(v) for t, v in base.tree.items()}
        gradient = np.zeros_like(m)
        direct_cost_gradient = np.zeros_like(m)
        dollar_denominator = np.zeros_like(m)
        for year in cons.periods[:-1]:
            period, factor = self._flow_factors[int(year)]
            first, last = self.tree.get_nodes_in_period(period)
            share_bar = -bar_cons[year]*base[year]
            if len(share_bar) == 2*(last-first+1):
                share_bar = share_bar.reshape(-1, 2).sum(axis=1)
            calendar_year = self.tree.base_year+year
            technology = self.technology_multiplier(calendar_year)
            dollar_denominator[first:last+1] += -share_bar*factor/technology
            direct_cost_gradient[first:last+1] += (share_bar*factor
                *self.cost._raw_marginal_cost(m[first:last+1]))
            bar_base[year] += bar_cons[year]*(1.0-shares[year])
        bar_base[base.last_period] += bar_cons[cons.last_period]

        bar_cost_tree = {t: np.zeros_like(v) for t, v in tape["cost_tree"].tree.items()}
        for year in base.periods[:-1]:
            op = tape["period_ops"][year]
            base_bar = self.climate_adjoint._reverse_consumption_op(
                bar_base[year], op, bar_base, bar_cost_tree)
            block = op["local_block"]
            if block is not None:
                potential = op.get("potential_consumption")
                if potential is None:
                    potential = self.zero.potential_cons[op["damage_period"]]
                block["bar_damage"] += -potential*base_bar
        terminal_block = tape["terminal_block"]
        terminal_bar = bar_base[base.last_period].copy()
        terminal_bar[tape["terminal_consumption_clipped"]] = 0.0
        terminal_block["bar_damage"] += -tape.get(
            "terminal_potential_consumption", self.zero.potential_cons[-1])*terminal_bar
        for block in tape["local_blocks"]:
            block["bar_cost"].fill(0.0)
            self.climate_adjoint._reverse_local_block(m, block, gradient)
        warming_gradient, low_ghg_gradient = _climate_component_gradients(
            tape, len(m))
        climate_gradient = gradient.copy()
        gradient += direct_cost_gradient
        if np.any(dollar_denominator <= 0):
            raise FloatingPointError("Nonpositive interval ton-pricing denominator")
        interval_gap = gradient/dollar_denominator
        self.last_diagnostics = {**diagnostics,
            "gradient_backend": "reverse_adjoint_five_year_flows",
            "minimum_consumption": float(min(np.min(v) for v in cons.tree.values())),
            "maximum_mitigation": float(np.max(m)),
            "maximum_interval_foc_gap_dollars_per_ton": float(np.max(np.abs(interval_gap))),
            "interval_foc_gap_dollars_per_ton": interval_gap.tolist(),
            "dollar_denominator": dollar_denominator.tolist(),
            "direct_cost_gradient": direct_cost_gradient.tolist(),
            "learning_gradient": np.zeros_like(m).tolist(),
            "climate_gradient": climate_gradient.tolist(),
            "warming_gradient": warming_gradient.tolist(),
            "low_ghg_gradient": low_ghg_gradient.tolist()}
        return value, gradient

    def diagnostics(self):
        return dict(self.last_diagnostics)


class LearningFiveYearFlowObjective(FiveYearFlowObjective):
    """Five-year costs with technology updated from mitigation to each date.

    At five-year date u, the technology factor is
    [1-(tech_const+tech_scale*average_mitigation_to_u)/100]^(u-2030).
    The average uses BAU-emissions weights over all completed years, including
    years since the most recent decision. The decision control remains fixed
    within its interval, but the resulting learning continues to accumulate.
    """

    def __init__(self, tree, damage, cost, emissions, *, learning_reference_year=None, **kwargs):
        scale = float(cost.tech_scale)
        if scale <= 0:
            raise ValueError("Learning experiment requires tech_scale > 0")
        cost.tech_scale = 0.0
        try:
            super().__init__(tree, damage, cost, emissions, **kwargs)
        finally:
            cost.tech_scale = scale
        self.cons_growth = float(kwargs.get("cons_growth", 0.015))
        # Strip the exogenous factor inserted by the parent constructor;
        # _forward supplies the state-dependent technology factor instead.
        self._flow_factors = {
            year: (period, factor/self.technology_multiplier(tree.base_year+year))
            for year, (period, factor) in self._flow_factors.items()
        }
        self._learning_weight_matrices = {}
        baseline_at = lambda elapsed: float(np.interp(
            tree.base_year+elapsed, emissions.times, emissions.baseline_cumemit))
        self.learning_reference_year = learning_reference_year
        fixed_denominator = None
        if learning_reference_year is not None:
            if not tree.base_year < learning_reference_year <= tree.base_year + tree.decision_times[-1]:
                raise ValueError("Learning reference year must fall within the model horizon")
            fixed_denominator = baseline_at(learning_reference_year-tree.base_year)-baseline_at(0)
            if fixed_denominator <= 0:
                raise ValueError("Learning reference cumulative BAU emissions must be positive")
        for year, (period, _factor) in self._flow_factors.items():
            weights = np.array([
                max(0.0, baseline_at(min(year, tree.decision_times[i+1]))
                    -baseline_at(tree.decision_times[i]))
                for i in range(period+1)
            ])
            denominator = (fixed_denominator if fixed_denominator is not None
                           else float(weights.sum()))
            first, last = tree.get_nodes_in_period(period)
            matrix = np.zeros((last-first+1, tree.num_decision_nodes))
            if denominator > 0:
                for state, node in enumerate(range(first, last+1)):
                    path = tree.get_path(node)
                    matrix[state, path] = weights/denominator
            self._learning_weight_matrices[year] = matrix

    def technology_by_year(self, mitigation, with_derivatives=False, tape=None):
        m = np.asarray(mitigation, dtype=float)
        result = {}
        for elapsed, matrix in self._learning_weight_matrices.items():
            average = matrix @ m
            exponent = self.tree.base_year+elapsed-self.cost.anchor_year
            tech_base = 1.0-(self.cost.tech_const+self.cost.tech_scale*average)/100.0
            if np.any(tech_base <= 0):
                raise FloatingPointError("Nonpositive learning-technology base")
            original_technology = tech_base**exponent
            floor_weight = (1.0-self.cost_floor if
                            self.tree.base_year+elapsed >= self.cost.anchor_year else 1.0)
            technology = ((self.cost_floor+floor_weight*original_technology) if
                          self.tree.base_year+elapsed >= self.cost.anchor_year else
                          original_technology)
            if with_derivatives:
                dtechnology_daverage = (floor_weight*exponent*tech_base**(exponent-1.0)
                                       *(-self.cost.tech_scale/100.0))
                result[elapsed] = (technology, dtechnology_daverage, matrix)
            else:
                result[elapsed] = technology
        return result

    def technology_by_period(self, mitigation, with_derivatives=False, tape=None):
        """Technology at decision dates, retained for decision-node reporting."""
        by_year = self.technology_by_year(mitigation, with_derivatives, tape)
        return {period: by_year[int(elapsed)] for period, elapsed in
                enumerate(self.tree.decision_times[:-1])}

    def _forward(self, mitigation):
        m = np.asarray(mitigation, dtype=float).reshape(-1)
        _, tape, diagnostics = self.climate_adjoint._forward_tape(m)
        base = tape["cons_tree"]
        technology = self.technology_by_year(m, tape=tape)
        consumption = BigStorageTree(self.period_len, self.tree.decision_times)
        shares = {}
        for year in consumption.periods[:-1]:
            period, factor = self._flow_factors[int(year)]
            first, last = self.tree.get_nodes_in_period(period)
            share = (factor*technology[int(year)]
                     *self.cost._raw_integrated_cost(m[first:last+1]))
            if len(base[year]) == 2*len(share):
                share = np.repeat(share, 2)
            if len(base[year]) != len(share):
                raise RuntimeError("Five-year cost states do not match consumption")
            flow = base[year]*(1.0-share)
            if np.any(flow <= 0) or not np.all(np.isfinite(flow)):
                raise FloatingPointError("Nonpositive five-year consumption")
            shares[year] = share
            consumption.set_value(year, flow)
        consumption.set_value(consumption.last_period, base.last.copy())

        utility = BigStorageTree(self.period_len, self.tree.decision_times)
        multiplier = self.climate_adjoint._terminal_multiplier()
        utility.set_value(utility.last_period, consumption.last*multiplier)
        ce_ops = {}
        for year in utility.periods[::-1][1:]:
            period = utility.between_decision_times(year)
            ce, ce_op = self.climate_adjoint._certain_equivalence_forward(
                year, period, utility)
            ce_ops[year] = (ce, ce_op)
            utility.set_value(year, self.zero._intertemporal_aggregate(
                consumption[year], ce))
        return float(utility[0][0]), tape, consumption, utility, shares, ce_ops, diagnostics

    def value_and_gradient(self, mitigation):
        m = np.asarray(mitigation, dtype=float).reshape(-1)
        if len(m) != self.tree.num_decision_nodes:
            raise ValueError("Mitigation vector has wrong length")
        value, tape, cons, utility, shares, ce_ops, diagnostics = self._forward(m)
        tech_with_derivatives = self.technology_by_year(
            m, with_derivatives=True, tape=tape)
        bar_u = {t: np.zeros_like(v) for t, v in utility.tree.items()}
        bar_cons = {t: np.zeros_like(v) for t, v in cons.tree.items()}
        bar_u[0][0] = 1.0
        for year in utility.periods[:-1]:
            ce, ce_op = ce_ops[year]
            u, c, weight = utility[year], cons[year], bar_u[year]
            if self.zero.is_log_eis:
                du_dc = (1.0-self.zero.b)*u/c
                du_dce = self.zero.b*u/ce
            else:
                du_dc = (u**(1.0-self.zero.r)*(1.0-self.zero.b)
                         *c**(self.zero.r-1.0))
                du_dce = (u**(1.0-self.zero.r)*self.zero.b
                          *ce**(self.zero.r-1.0))
            bar_cons[year] += weight*du_dc
            bar_ce = weight*du_dce
            next_year = year+self.period_len
            if ce_op["information"]:
                bar_u[next_year][::2] += bar_ce*ce_op["even_weight"]
                bar_u[next_year][1::2] += bar_ce*ce_op["odd_weight"]
            else:
                bar_u[next_year] += bar_ce
        bar_cons[cons.last_period] += (
            bar_u[utility.last_period]*self.climate_adjoint._terminal_multiplier())

        base = tape["cons_tree"]
        bar_base = {t: np.zeros_like(v) for t, v in base.tree.items()}
        direct_cost_gradient = np.zeros_like(m)
        learning_gradient = np.zeros_like(m)
        dollar_denominator = np.zeros_like(m)
        for year in cons.periods[:-1]:
            period, factor = self._flow_factors[int(year)]
            first, last = self.tree.get_nodes_in_period(period)
            technology, dtechnology_daverage, daverage = tech_with_derivatives[int(year)]
            integrated_cost = self.cost._raw_integrated_cost(m[first:last+1])
            share_bar = -bar_cons[year]*base[year]
            if len(share_bar) == 2*(last-first+1):
                share_bar = share_bar.reshape(-1, 2).sum(axis=1)
            dollar_denominator[first:last+1] += -share_bar*factor
            direct_cost_gradient[first:last+1] += (share_bar*factor*technology
                *self.cost._raw_marginal_cost(m[first:last+1]))
            learning_gradient += ((share_bar*factor*integrated_cost
                                   *dtechnology_daverage) @ daverage)
            bar_base[year] += bar_cons[year]*(1.0-shares[year])
        bar_base[base.last_period] += bar_cons[cons.last_period]

        bar_cost_tree = {t: np.zeros_like(v) for t, v in tape["cost_tree"].tree.items()}
        for year in base.periods[:-1]:
            op = tape["period_ops"][year]
            base_bar = self.climate_adjoint._reverse_consumption_op(
                bar_base[year], op, bar_base, bar_cost_tree)
            block = op["local_block"]
            if block is not None:
                potential = op.get("potential_consumption")
                if potential is None:
                    potential = self.zero.potential_cons[op["damage_period"]]
                block["bar_damage"] += -potential*base_bar
        terminal_block = tape["terminal_block"]
        terminal_bar = bar_base[base.last_period].copy()
        terminal_bar[tape["terminal_consumption_clipped"]] = 0.0
        terminal_block["bar_damage"] += -tape.get(
            "terminal_potential_consumption", self.zero.potential_cons[-1])*terminal_bar
        climate_gradient = np.zeros_like(m)
        for block in tape["local_blocks"]:
            block["bar_cost"].fill(0.0)
            self.climate_adjoint._reverse_local_block(m, block, climate_gradient)
        warming_gradient, low_ghg_gradient = _climate_component_gradients(
            tape, len(m))
        if not np.allclose(climate_gradient, warming_gradient+low_ghg_gradient,
                           rtol=1e-10, atol=1e-10):
            raise RuntimeError("Climate-risk decomposition does not sum to adjoint gradient")
        gradient = direct_cost_gradient+learning_gradient+climate_gradient
        if np.any(dollar_denominator <= 0):
            raise FloatingPointError("Nonpositive interval ton-pricing denominator")
        self.last_diagnostics = {**diagnostics,
            "gradient_backend": "reverse_adjoint_five_year_flows_with_learning",
            "minimum_consumption": float(min(np.min(v) for v in cons.tree.values())),
            "maximum_mitigation": float(np.max(m)),
            "maximum_interval_foc_gap_dollars_per_ton": float(np.max(
                np.abs(gradient/dollar_denominator))),
            "interval_foc_gap_dollars_per_ton": (gradient/dollar_denominator).tolist(),
            "dollar_denominator": dollar_denominator.tolist(),
                        "direct_cost_gradient": direct_cost_gradient.tolist(),
            "learning_gradient": learning_gradient.tolist(),
            "climate_gradient": climate_gradient.tolist(),
            "warming_gradient": warming_gradient.tolist(),
            "low_ghg_gradient": low_ghg_gradient.tolist()}
        return value, gradient


def build_saved_revised_model(damage_path, *, decision_times=None,
                             prob_scale=1.0,
                             ra=BASELINE.ra, eis=BASELINE.eis, prtp=BASELINE.prtp,
                             growth=BASELINE.growth, tech_const=BASELINE.tech_const,
                             tech_scale=BASELINE.tech_scale, backstop=BASELINE.backstop,
                             learning_reference_year=None,
                             gamma=BASELINE.gamma,
                             backstop_curvature=BASELINE.backstop_curvature,
                             mac_horizontal_shift=0.0,
                             mac_vertical_shift=0.0,
                             cost_floor=0.0,
                             baseline_num=BASELINE.baseline_num,
                             common_damage_path=None,
                             terminal_growth_mode=BASELINE.terminal_growth_mode,
                             terminal_growth=BASELINE.terminal_growth,
                             terminal_log_mean=BASELINE.terminal_log_mean,
                             terminal_log_sigma=BASELINE.terminal_log_sigma):
    """Load an existing simulated damage grid without resimulating it."""
    if ra <= 0 or eis <= 0 or not 0 <= prtp < 1:
        raise ValueError("Require RA > 0, EIS > 0, and 0 <= PRTP < 1")
    if (mac_horizontal_shift or mac_vertical_shift) and backstop != "s_curve":
        raise ValueError("MAC shifts require the revised smooth S-curve backstop")
    if decision_times is None:
        decision_times = [0, 5, 10, 15, 35, 75, 125, 175, 225]
    if not np.isfinite(prob_scale) or prob_scale <= 0:
        raise ValueError("prob_scale must be finite and positive")
    tree = TreeModel(decision_times, prob_scale=prob_scale)
    emissions = FlatAfter2100Emissions(tree, baseline_num, emissions_time_step=5)
    emissions.baseline_emission_setup()
    with np.load(damage_path, allow_pickle=False) as saved:
        knots = np.asarray(saved["knots"], float)
        if "decision_times" in saved and not np.array_equal(
                saved["decision_times"], tree.decision_times):
            raise ValueError("Saved damage grid belongs to another decision tree")
        if "prob_scale" in saved and not np.isclose(
                float(saved["prob_scale"]), prob_scale, rtol=0, atol=1e-12):
            raise ValueError("Saved damage grid belongs to another probability scale")
        simulated = np.asarray(saved["damages"], float)
    climate = BPWClimate(tree, emissions, draws=16)
    damage = PchipDamage(tree, emissions, climate, knots, draws=16)
    damage.d = simulated
    damage._recombine_nodes()
    reference_emissions = float(np.interp(2030, emissions.times,
                                          emissions.baseline_gtco2))
    if backstop == "s_curve":
        cost = SmoothBackstopCost(tree, reference_emissions, baseline_num,
                                  tech_const, tech_scale, 86252.0, 0.0, True,
                                  backstop_curvature=backstop_curvature,
                                  mac_horizontal_shift=mac_horizontal_shift,
                                  mac_vertical_shift=mac_vertical_shift)
    elif backstop == "cubic":
        if backstop_curvature != 1.0:
            raise ValueError("backstop_curvature applies only to s_curve")
        cost = CubicBackstopCost(tree, reference_emissions, baseline_num,
                                 tech_const, tech_scale, 86252.0, 0.0, True,
                                 gamma=gamma)
    elif backstop == "exp_cubic":
        if backstop_curvature != 1.0:
            raise ValueError("backstop_curvature applies only to s_curve")
        cost = ExponentialCubicBackstopCost(
            tree, reference_emissions, baseline_num, tech_const, tech_scale,
            86252.0, 0.0, True, gamma=gamma)
    else:
        raise ValueError("backstop must be 's_curve', 'cubic', or 'exp_cubic'")
    objective_class = (LearningFiveYearFlowObjective if tech_scale != 0
                       else FiveYearFlowObjective)
    if learning_reference_year is not None and tech_scale == 0:
        raise ValueError("A fixed learning reference year requires endogenous learning")
    learning_options = ({"learning_reference_year": learning_reference_year}
                        if tech_scale != 0 else {})
    return objective_class(tree, damage, cost, emissions,
                           **learning_options,
                           ra=ra, eis=eis, time_pref=prtp,
                           cons_growth=growth,
                           common_damage_path=common_damage_path,
                           cost_floor=cost_floor,
                           terminal_growth_mode=terminal_growth_mode,
                           terminal_growth=terminal_growth,
                           terminal_log_mean=terminal_log_mean,
                           terminal_log_sigma=terminal_log_sigma)

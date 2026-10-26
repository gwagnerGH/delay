"""Common-calendar climate consumption for decision-tree comparisons.

All compared trees use one saved, finely resolved damage simulation. Physical
damages are interpolated between *fixed calendar anchors*, rather than between
the decision dates of each tree. A tree changes only the policy intervals and
the information partition used to condition those underlying damage states.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .climate import BPWClimate
from .storage_tree import BigStorageTree
from .tree import TreeModel
from .revised_experiment import FlatAfter2100Emissions, PchipDamage, _ClimateAdjoint


class CommonCalendarClimateAdjoint(_ClimateAdjoint):
    """Supply five-year climate consumption and its exact policy derivatives.

    The saved input is a full-tree damage simulation at common anchor years.
    Between anchors, damage fractions are linearly interpolated in calendar
    time for each prototype path and terminal climate state. The physical
    interpolation is therefore identical for every coarsened decision tree.
    The cooling adjustment and emissions/concentration response are evaluated
    directly at every five-year date, including dates inside decision intervals.
    """

    def __init__(self, utility, common_damage_path):
        super().__init__(utility, value_parity_mode="off")
        source = Path(common_damage_path)
        with np.load(source, allow_pickle=False) as saved:
            full_times = np.asarray(saved["decision_times"], int)
            full_values = np.asarray(saved["damages"], float)
            knots = np.asarray(saved["knots"], float)
        full_tree = TreeModel(full_times)
        if full_tree.base_year != self.tree.base_year:
            raise ValueError("Common damage grid has a different base year")
        if full_times[0] != 0 or full_times[-1] != self.tree.decision_times[-1]:
            raise ValueError("Common damage grid must cover the full model horizon")
        if any(t not in full_times for t in self.tree.decision_times):
            raise ValueError("Every decision date must be a common-grid anchor")
        if full_tree.num_final_states % self.tree.num_final_states:
            raise ValueError("Common damage states cannot be grouped into this tree")
        if full_values.shape != (len(knots), full_tree.num_final_states,
                                 len(full_times)-1):
            raise ValueError("Common damage grid has the wrong shape")
        full_emit = FlatAfter2100Emissions(full_tree,
            self.damage.emit_baseline.baseline_num, emissions_time_step=5)
        full_emit.baseline_emission_setup()
        full_climate = BPWClimate(full_tree, full_emit, draws=16)
        full_damage = PchipDamage(full_tree, full_emit, full_climate, knots, draws=16)
        full_damage.d = full_values
        full_damage._recombine_nodes()
        self.common_knots = knots
        self.common_years = full_tree.base_year + full_times[1:]
        self.full_damage = full_damage.d_rcomb
        self.full_probabilities = full_tree.final_states_prob.copy()
        self.group_size = full_tree.num_final_states // self.tree.num_final_states
        self._calendar_kernels = {}
        self._calendar_curves = {}
        self._calendar_curve_derivatives = {}
        self._calendar_full_values = {}
        self._calendar_membership = {}
        self._prepare_calendar()

    def _year_kernel(self, elapsed):
        """Carbon-cycle responses with actual decision-boundary jumps up to u."""
        year = int(self.tree.base_year + elapsed)
        boundaries = list(self.tree.base_year + self.tree.decision_times[
            self.tree.decision_times < elapsed]) + [year]
        emit = self.damage.emit_baseline
        times = []
        gtco2 = []
        ppm = []
        interval_ids = []
        for index, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            selection = (emit.times >= low) & (emit.times <= high)
            if not np.any(selection):
                raise ValueError("Missing five-year emissions samples")
            times.extend(emit.times[selection])
            gtco2.extend(emit.baseline_gtco2[selection])
            ppm.extend(emit.baseline_ppm[selection])
            interval_ids.extend([index]*int(np.sum(selection)))
        times = np.asarray(times, float)
        gtco2 = np.asarray(gtco2, float)
        ppm = np.asarray(ppm, float)
        interval_ids = np.asarray(interval_ids, int)
        n = len(boundaries)-1
        cemit_base = float(emit.CUMEMIT_BASE_YEAR)
        cemit_intercept = (cemit_base + 1e-3*np.sum(
            0.5*np.diff(times)*(gtco2[:-1]+gtco2[1:])))
        conc_intercept = self._concentration_from_flow(ppm, times)
        cemit_coeff = np.zeros(n)
        conc_coeff = np.zeros(n)
        for index in range(n):
            active = interval_ids == index
            d_gtco2 = np.where(active, -gtco2, 0.0)
            d_ppm = np.where(active, -ppm, 0.0)
            cemit_coeff[index] = 1e-3*np.sum(
                0.5*np.diff(times)*(d_gtco2[:-1]+d_gtco2[1:]))
            conc_coeff[index] = self._concentration_linear_response(d_ppm, times)
        return cemit_intercept, conc_intercept, cemit_coeff, conc_coeff

    def _full_values_at_year(self, year):
        if year <= self.tree.base_year:
            return np.zeros_like(self.full_damage[:, :, 0])
        upper = int(np.searchsorted(self.common_years, year, side="left"))
        if upper >= len(self.common_years):
            raise ValueError("Calendar damage grid ends before utility horizon")
        if year == self.common_years[upper]:
            return self.full_damage[:, :, upper]
        low_year = (self.tree.base_year if upper == 0
                    else self.common_years[upper-1])
        low = (np.zeros_like(self.full_damage[:, :, 0]) if upper == 0
               else self.full_damage[:, :, upper-1])
        alpha = (year-low_year)/(self.common_years[upper]-low_year)
        return (1-alpha)*low + alpha*self.full_damage[:, :, upper]

    def _prepare_calendar(self):
        years = list(range(0, int(self.tree.decision_times[-1])+1,
                           int(self.period_len)))
        storage = BigStorageTree(self.period_len, self.tree.decision_times)
        for elapsed in years[1:]:
            year = self.tree.base_year+elapsed
            period = int(np.searchsorted(self.tree.decision_times, elapsed,
                                         side="right")-1)
            period = min(period, self.tree.num_periods-1)
            state_count = len(storage[elapsed])
            if state_count == self.tree.get_num_nodes_period(period):
                evaluation_period = period
            elif (period+1 < self.tree.num_periods and
                  state_count == self.tree.get_num_nodes_period(period+1)):
                # BigStorageTree carries next-revelation contingencies at
                # intermediate utility dates. The current mitigation control
                # remains fixed in both future-contingent consumption states.
                evaluation_period = period+1
            else:
                raise ValueError("Unsupported calendar state layout")
            start, _ = self.tree.get_nodes_in_period(evaluation_period)
            full_values = self._full_values_at_year(year)
            cemit_bau, conc_bau, cemit_coeff, conc_coeff = self._year_kernel(elapsed)
            coordinates = (self.damage.emit_baseline.CUMEMIT_BASE_YEAR
                           + (1.0-self.common_knots)
                           * (cemit_bau-self.damage.emit_baseline.CUMEMIT_BASE_YEAR))
            if np.any(np.diff(coordinates) <= 0):
                raise ValueError("Common damage knots are not ordered")
            membership = []
            for state in range(state_count):
                node = start+state
                coarse_states = self.tree.reachable_end_states(node)
                full_states = (coarse_states[:, None]*self.group_size
                               + np.arange(self.group_size)[None, :]).ravel()
                weights = self.full_probabilities[full_states]
                membership.append((full_states, weights/weights.sum()))
            self._calendar_kernels[elapsed] = (
                period, evaluation_period, cemit_bau, conc_bau,
                cemit_coeff, conc_coeff,
                coordinates)
            # Interpolate *each physical state* before conditioning on a tree
            # node. PCHIP(conditional mean) would change unconditional fixed-
            # policy damages when the number of tree states changes.
            curve = PchipInterpolator(coordinates, full_values, axis=0,
                                      extrapolate=False)
            self._calendar_curves[elapsed] = curve
            self._calendar_curve_derivatives[elapsed] = curve.derivative()
            self._calendar_full_values[elapsed] = full_values
            self._calendar_membership[elapsed] = membership

    def _calendar_block(self, elapsed, mitigation):
        (period, evaluation_period, cemit_bau, conc_bau,
         cemit_coeff, conc_coeff, coordinates) = self._calendar_kernels[elapsed]
        state_count = len(self._calendar_membership[elapsed])
        first, last = self.tree.get_nodes_in_period(evaluation_period)
        control_first, control_last = self.tree.get_nodes_in_period(period)
        damages = np.empty(state_count)
        states = []
        for state, node in enumerate(range(first, last+1)):
            path = np.asarray(self.tree.get_path(node)[:len(cemit_coeff)], int)
            if len(path) != len(cemit_coeff):
                raise RuntimeError("Calendar policy path length mismatch")
            cemit = cemit_bau + np.dot(cemit_coeff, mitigation[path])
            conc = conc_bau + np.dot(conc_coeff, mitigation[path])
            indices, weights = self._calendar_membership[elapsed][state]
            if cemit <= coordinates[0]:
                warming, slope = 0.0, 0.0
            elif cemit >= coordinates[-1]:
                warming = float(weights @ self._calendar_full_values[elapsed][-1, indices])
                slope = 0.0
            else:
                warming = float(weights @ self._calendar_curves[elapsed](cemit)[indices])
                slope = float(weights @ self._calendar_curve_derivatives[elapsed](cemit)[indices])
            cooling = 1.0/(1.0+np.exp(np.clip(0.05*(conc-200.0), -700, 700)))
            damages[state] = warming+cooling
            states.append({
                "path": path,
                "cemit_coeff": cemit_coeff,
                "conc_coeff": conc_coeff,
                "damage_slope": slope,
                "penalty_slope": -0.05*cooling*(1.0-cooling),
                "average_denom": np.inf,
                "warming": warming,
                "cooling": cooling,
                "cemit": float(cemit),
                "conc": float(conc),
            })
        return {
            "period": period,
            "node_ids": np.arange(control_first, control_last+1),
            "mitigation": mitigation[control_first:control_last+1],
            "average": np.zeros(control_last-control_first+1),
            "damage": damages,
            "cost": np.zeros(control_last-control_first+1),
            "states": states,
            "bar_damage": np.zeros(state_count),
            "bar_cost": np.zeros(control_last-control_first+1),
        }

    def _forward_tape(self, mitigation):
        m = np.asarray(mitigation, float)
        base = BigStorageTree(self.period_len, self.tree.decision_times)
        tape = {"cons_tree": base, "cost_tree": BigStorageTree(
            self.period_len, self.tree.decision_times),
            "period_ops": {}, "local_blocks": [], "calendar_blocks": {}}
        for elapsed in base.periods:
            if elapsed == 0:
                base.set_value(0, np.ones(1))
                tape["period_ops"][0] = {
                    "interpolation": None, "base_consumption": np.ones(1),
                    "base_clipped": np.zeros(1, bool), "local_block": None,
                    "potential_consumption": 1.0,
                }
                continue
            block = self._calendar_block(int(elapsed), m)
            potential = float((1.0+self.utility_model.cons_growth)**elapsed)
            flow = potential*(1.0-block["damage"])
            if np.any(flow <= 0) or not np.all(np.isfinite(flow)):
                raise FloatingPointError("Nonpositive common-calendar climate consumption")
            base.set_value(elapsed, flow)
            tape["local_blocks"].append(block)
            tape["calendar_blocks"][elapsed] = block
            if elapsed == base.last_period:
                tape["terminal_block"] = block
                tape["terminal_consumption_clipped"] = np.zeros(len(flow), bool)
                tape["terminal_potential_consumption"] = potential
            else:
                tape["period_ops"][elapsed] = {
                    "interpolation": None, "base_consumption": flow,
                    "base_clipped": np.zeros(len(flow), bool),
                    "local_block": block,
                    "potential_consumption": potential,
                }
        diagnostics = {"num_damage_interp_clipped": 0,
                       "num_consumption_clipped": 0,
                       "num_overmitigation_nodes": int(np.sum(m > 1.0)),
                       "physical_damage_calendar": "common_five_year"}
        return 0.0, tape, diagnostics

    def _reverse_local_block(self, mitigation, block, gradient):
        """Differentiate only physical damages; resource costs live in flow objective."""
        for state, exposure in enumerate(block["states"]):
            path = exposure["path"]
            if len(path):
                weight = block["bar_damage"][state]
                gradient[path] += weight*(
                    exposure["damage_slope"]*exposure["cemit_coeff"]
                    + exposure["penalty_slope"]*exposure["conc_coeff"])

"""Checks for the revised cost-accounting and smooth-damage adjoint."""

import unittest

import numpy as np

from src.climate import BPWClimate
from src.revised_experiment import (FiveYearFlowObjective,
    FlatAfter2100Emissions, PchipDamage, SmoothBackstopCost)
from src.optimization import CandidateScreenedLBFGSB
from scripts.run_revised_adjoint import NormalizedObjective
from src.tree import TreeModel


def make_objective(eis=0.9, backstop_curvature=1.0,
                   mac_horizontal_shift=0.0, mac_vertical_shift=0.0):
    tree = TreeModel([0, 5, 10])
    emissions = FlatAfter2100Emissions(tree, 2, emissions_time_step=5)
    emissions.baseline_emission_setup()
    climate = BPWClimate(tree, emissions, draws=10)
    knots = np.round(np.arange(4.0, -0.0001, -0.2), 8)
    damage = PchipDamage(tree, emissions, climate, knots, draws=10)
    simulated = np.arange(len(knots), dtype=float)**1.4 * 0.0004
    damage.d_rcomb = np.broadcast_to(simulated[:, None, None],
        (len(knots), tree.num_final_states, tree.num_periods)).copy()
    reference_emissions = float(np.interp(2030, emissions.times,
                                          emissions.baseline_gtco2))
    cost = SmoothBackstopCost(tree, reference_emissions, 2, 1.5, 0.0,
                              86252.0, 0.0, True,
                              backstop_curvature=backstop_curvature,
                              mac_horizontal_shift=mac_horizontal_shift,
                              mac_vertical_shift=mac_vertical_shift)
    # Tests deliberately include high EIS; use a finite, explicit deterministic
    # tail so they test cost derivatives rather than terminal admissibility.
    return FiveYearFlowObjective(tree, damage, cost, emissions, eis=eis,
                                terminal_growth_mode="deterministic", terminal_growth=0.005)


class RevisedExperimentTests(unittest.TestCase):
    def test_smooth_cost_join_and_integral(self):
        cost = make_objective().cost
        self.assertAlmostEqual(cost._raw_integrated_cost(0.0), 0.0)
        # Removal retains the published shape while matching both value and
        # slope at m=1 exactly; rounded legacy coefficients differed slightly.
        self.assertLess(abs(cost._raw_marginal_cost(1.0-1e-9)
                            - cost._raw_marginal_cost(1.0+1e-9)), 1e-5)
        self.assertAlmostEqual(float(cost._raw_marginal_cost(0.5)),
                               58.9 * np.expm1(2.3 * 0.5), places=10)
        self.assertLess(abs(float(cost._raw_marginal_cost(1.5))
                            -(2500 - 1971.43*np.exp(-0.6854*0.5-0.5**2))), 0.01)
        ordinary_slope = 58.9*2.3*np.exp(2.3)
        h = 1e-5
        removal_slope = (cost._raw_marginal_cost(1.0+h)
                         -cost._raw_marginal_cost(1.0))/h
        np.testing.assert_allclose(removal_slope, ordinary_slope, rtol=2e-5)
        alternative_prices = []
        for curvature in (0.5, 1.0, 2.0):
            alternative = SmoothBackstopCost(cost.tree, cost.cons_per_ton,
                2, 1.5, 0.0, 86252.0, 0.0, True,
                backstop_curvature=curvature)
            self.assertAlmostEqual(float(alternative._raw_marginal_cost(0.8)),
                                   float(cost._raw_marginal_cost(0.8)), places=12)
            alternative_prices.append(float(alternative._raw_marginal_cost(1.5)))
            for m in (1.0001, 1.2, 1.5, 2.0):
                derivative = (alternative._raw_integrated_cost(m+h)
                              -alternative._raw_integrated_cost(m-h))/(2*h)
                np.testing.assert_allclose(derivative,
                    alternative._raw_marginal_cost(m), rtol=2e-7, atol=2e-4)
        self.assertLess(alternative_prices[0], alternative_prices[1])
        self.assertLess(alternative_prices[1], alternative_prices[2])
        for m in (0.35, 0.8, 1.2, 2.0):
            h = 1e-5
            derivative = (cost._raw_integrated_cost(m+h)
                          - cost._raw_integrated_cost(m-h))/(2*h)
            self.assertAlmostEqual(float(derivative),
                                   float(cost._raw_marginal_cost(m)), places=5)

    def test_curved_removal_adjoint_matches_finite_differences(self):
        mitigation = np.array([0.6, 1.4, 0.9])
        for curvature in (0.5, 2.0):
            with self.subTest(curvature=curvature):
                objective = make_objective(backstop_curvature=curvature)
                value, gradient = objective.value_and_gradient(mitigation)
                self.assertTrue(np.isfinite(value))
                self.assertGreater(objective.diagnostics()["minimum_consumption"], 0)
                for index in range(mitigation.size):
                    step = 2e-5
                    above = mitigation.copy()
                    below = mitigation.copy()
                    above[index] += step
                    below[index] -= step
                    finite = (objective.utility(above)-objective.utility(below))/(2*step)
                    np.testing.assert_allclose(gradient[index], finite,
                                               rtol=2e-5, atol=1e-8)

    def test_scaled_gradient_validation_uses_accurate_step(self):
        objective = make_objective(backstop_curvature=1.5)
        reference = objective.utility(np.array([0.6, 1.3, 0.7]))
        scaled = NormalizedObjective(objective, reference, 10000.0)
        solver = CandidateScreenedLBFGSB(
            scaled, np.zeros(3), np.full(3, 4.0),
            objective_with_gradient=scaled, gradient_mode="adjoint",
            validate_gradient=True)
        diagnostics = solver._validate_adjoint_gradient()
        self.assertEqual(diagnostics["gradient_validation_status"], "passed")
        self.assertLess(diagnostics["gradient_validation_max_rel_error"], 1e-6)

    def test_shifted_smooth_mac_and_integral(self):
        base = make_objective().cost
        zero = make_objective(mac_horizontal_shift=0.0,
                              mac_vertical_shift=0.0).cost
        shifted = make_objective(mac_horizontal_shift=0.2,
                                 mac_vertical_shift=50.0).cost
        self.assertEqual(float(shifted._raw_integrated_cost(0.0)), 0.0)
        for m in (0.0, 0.1, 0.7, 0.8, 1.0, 1.2, 2.0, 3.8, 4.0):
            with self.subTest(m=m):
                self.assertEqual(float(zero._raw_marginal_cost(m)),
                                 float(base._raw_marginal_cost(m)))
                self.assertEqual(float(zero._raw_integrated_cost(m)),
                                 float(base._raw_integrated_cost(m)))
                self.assertAlmostEqual(float(shifted._raw_marginal_cost(m)),
                    float(base._raw_marginal_cost(m+0.2))+50.0, places=9)
                self.assertAlmostEqual(float(shifted._raw_integrated_cost(m)),
                    float(base._raw_integrated_cost(m+0.2)
                          -base._raw_integrated_cost(0.2)+50.0*m), places=9)
                if 0.0 < m < 4.0:
                    step = 1e-5
                    finite = (shifted._raw_integrated_cost(m+step)
                              -shifted._raw_integrated_cost(m-step))/(2*step)
                    np.testing.assert_allclose(finite,
                        shifted._raw_marginal_cost(m), rtol=2e-7, atol=2e-4)

    def test_shifted_smooth_mac_adjoint(self):
        objective = make_objective(eis=2.2, mac_horizontal_shift=0.2,
                                   mac_vertical_shift=50.0)
        mitigation = np.array([0.6, 1.4, 0.9])
        value, gradient = objective.value_and_gradient(mitigation)
        self.assertTrue(np.isfinite(value))
        for index in range(mitigation.size):
            step = 2e-5
            above, below = mitigation.copy(), mitigation.copy()
            above[index] += step
            below[index] -= step
            finite = (objective.utility(above)-objective.utility(below))/(2*step)
            np.testing.assert_allclose(gradient[index], finite,
                                       rtol=2e-5, atol=1e-8)

    def test_adjoint_matches_finite_differences_across_preferences(self):
        m = np.array([0.6, 1.3, 0.75])
        for eis in (0.9, 1.0, 1.5):
            with self.subTest(eis=eis):
                objective = make_objective(eis)
                value, gradient = objective.value_and_gradient(m)
                self.assertAlmostEqual(value, objective.utility(m), places=12)
                self.assertGreater(objective.diagnostics()["minimum_consumption"], 0)
                self.assertTrue(np.all(np.isfinite(gradient)))
                for node in range(len(m)):
                    h = 1e-3
                    values = []
                    for multiple in (-2.0, -1.0, 1.0, 2.0):
                        point = m.copy()
                        point[node] += multiple*h
                        values.append(objective.utility(point))
                    finite_difference = (
                        values[0]-8.0*values[1]+8.0*values[2]-values[3]
                    )/(12.0*h)
                    np.testing.assert_allclose(
                        gradient[node], finite_difference,
                        rtol=1e-5, atol=1e-6,
                    )

    def test_flat_emissions_and_five_year_technology(self):
        objective = make_objective()
        emissions = objective.emit
        level = float(np.interp(2100, emissions.times, emissions.baseline_gtco2))
        self.assertTrue(np.allclose(emissions.baseline_gtco2[emissions.times > 2100],
                                    level))
        self.assertEqual(objective._flow_factors[0][0], 0)
        self.assertEqual(objective._flow_factors[5][0], 1)

    def test_delay_window_compensation_matches_unadjusted_utility(self):
        objective = make_objective()
        m = np.array([0.0, 0.7, 0.9])
        base = objective.utility(m)
        self.assertAlmostEqual(
            objective.delay_window_compensated_utility(m, 5, 1.0), base, places=12)
        self.assertGreater(
            objective.delay_window_compensated_utility(m, 5, 1.01), base)


if __name__ == "__main__":
    unittest.main()

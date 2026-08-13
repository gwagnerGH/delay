import unittest
import types

import numpy as np

from src.adjoint_objective import (
    EZAdjointObjective,
    EZForwardSensitivityObjective,
)
from src.climate import BPWClimate
from src.cost import BPWCost
from src.damage import BPWDamage
from src.emit_baseline import BPWEmissionBaseline
from src.tree import TreeModel
from src.tools import get_integral_var_ub
from src.utility import EZUtility
from src.analysis.climate_output import ClimateOutput


def make_tiny_real_utility(
        decision_times=None, backstop_smoothing_width=0.0,
        backstop_smoothing_mode="one_sided_huber"):

    tree = TreeModel(decision_times or [0, 5, 10])
    emit = BPWEmissionBaseline(tree=tree, baseline_num=2)
    emit.baseline_emission_setup()
    climate = BPWClimate(tree, emit, draws=10)
    damage = BPWDamage(
        tree=tree,
        emit_baseline=emit,
        climate=climate,
        mitigation_constants=np.array([0.9, 0.6, 0.0]),
        draws=10,
    )
    damage.d_rcomb = np.zeros(
        (damage.dnum, tree.num_final_states, tree.num_periods)
    )
    cost = BPWCost(
        tree=tree,
        emit_at_0=emit.baseline_gtco2[0],
        baseline_num=2,
        tech_const=1.5,
        tech_scale=1.5,
        cons_at_0=61880.0,
        backstop_premium=10000.0,
        no_free_lunch=False,
        backstop_smoothing_width=backstop_smoothing_width,
        backstop_smoothing_mode=backstop_smoothing_mode,
    )
    return EZUtility(
        tree=tree,
        damage=damage,
        cost=cost,
        period_len=5.0,
        eis=0.833,
        ra=10.0,
        time_pref=0.002,
        cons_growth=0.015,
    )


class TestEZAdjointObjective(unittest.TestCase):
    def test_value_matches_legacy_utility_on_tiny_real_stack(self):
        utility = make_tiny_real_utility()
        objective = EZAdjointObjective(utility)
        mitigation = np.array([0.3, 0.4, 0.5])

        value, gradient = objective.value_and_gradient(mitigation)
        legacy_value = objective.utility(mitigation)
        diagnostics = objective.diagnostics()

        self.assertTrue(np.isfinite(value))
        self.assertEqual(gradient.shape, mitigation.shape)
        self.assertAlmostEqual(value, legacy_value, places=10)
        self.assertLess(diagnostics["adjoint_value_rel_diff"], 1e-10)

    def test_variable_upper_bound_integral_preserves_float64_perturbations(self):
        times = np.array([0.0, 1.0, 2.0])
        integrand = np.ones_like(times)
        perturbed_integrand = integrand.copy()
        perturbed_integrand[1] += 1e-8

        baseline = get_integral_var_ub(integrand, times, times)
        perturbed = get_integral_var_ub(perturbed_integrand, times, times)

        self.assertEqual(baseline.dtype, np.dtype(np.float64))
        self.assertAlmostEqual(
            perturbed[-1] - baseline[-1], 1e-8, delta=1e-12
        )

    def test_mitigation_path_preserves_float64_perturbations(self):
        utility = make_tiny_real_utility()
        emit = utility.damage.emit_baseline
        mitigation = np.array([0.3, 0.4, 0.5])
        perturbed_mitigation = mitigation.copy()
        perturbed_mitigation[0] += 1e-8

        baseline, _ = emit.get_mitigated_baseline(
            mitigation, node=1, baseline="gtco2", is_last=True
        )
        perturbed, _ = emit.get_mitigated_baseline(
            perturbed_mitigation, node=1, baseline="gtco2", is_last=True
        )

        self.assertEqual(baseline.dtype, np.dtype(np.float64))
        self.assertGreater(np.max(np.abs(perturbed - baseline)), 1e-8)

    def test_small_perturbations_preserve_value_gradient_parity(self):
        utility = make_tiny_real_utility([0, 5, 15, 25])
        objective = EZAdjointObjective(utility, value_parity_mode="off")
        mitigation = np.linspace(
            0.2, 0.8, utility.tree.num_decision_nodes
        )
        direction = np.arange(
            1, utility.tree.num_decision_nodes + 1, dtype=float
        )
        direction /= np.linalg.norm(direction)
        step = 1e-7

        _, gradient = objective.value_and_gradient(mitigation)
        plus_value, _ = objective.value_and_gradient(
            mitigation + step * direction
        )
        minus_value, _ = objective.value_and_gradient(
            mitigation - step * direction
        )
        finite_difference = (plus_value - minus_value) / (2.0 * step)
        adjoint_dot = float(np.dot(gradient, direction))

        self.assertLess(abs(finite_difference - adjoint_dot), 5e-8)

    def test_directional_derivative_matches_finite_difference(self):
        utility = make_tiny_real_utility()
        objective = EZAdjointObjective(utility)
        mitigation = np.array([0.3, 0.4, 0.5])
        _, gradient = objective.value_and_gradient(mitigation)
        direction = np.array([0.2, -0.1, 0.3])
        direction = direction / np.linalg.norm(direction)
        eps = 1e-4

        finite_difference = (
            objective.utility(mitigation + eps * direction)
            - objective.utility(mitigation - eps * direction)
        ) / (2.0 * eps)
        adjoint_dot = float(np.dot(gradient, direction))

        self.assertLess(abs(finite_difference - adjoint_dot), 1e-7)

    def test_additive_removal_premium_is_continuous_and_marginal(self):
        cost = make_tiny_real_utility().cost
        step = 1e-7

        at_one = float(cost._raw_integrated_cost(1.0))
        just_above = float(cost._raw_integrated_cost(1.0 + step))
        calibrated_increment = cost.tau_0 * (
            np.expm1(cost.power * (1.0 + step)) / cost.power
            - (1.0 + step)
            - (np.expm1(cost.power) / cost.power - 1.0)
        )

        self.assertAlmostEqual(
            just_above - at_one,
            calibrated_increment + cost.backstop_premium * step,
            places=11,
        )
        marginal_above = float(cost._raw_marginal_cost(1.01))
        calibrated_marginal = cost.tau_0 * np.expm1(cost.power * 1.01)
        self.assertAlmostEqual(
            marginal_above - calibrated_marginal,
            cost.backstop_premium,
            places=10,
        )

    def test_backstop_premium_at_cap_one_point_five_applies_only_to_removal(self):
        """Guard against the legacy CAP6 whole-curve premium error."""

        cost = make_tiny_real_utility().cost
        mitigation = 1.5
        calibrated_integral = cost.tau_0 * (
            np.expm1(cost.power * mitigation) / cost.power - mitigation
        )
        expected_integral = (
            calibrated_integral
            + cost.backstop_premium * (mitigation - 1.0)
        )
        legacy_whole_curve_integral = (
            (cost.tau_0 + cost.backstop_premium) * (
                np.expm1(cost.power * mitigation) / cost.power - mitigation
            )
        )

        self.assertAlmostEqual(
            float(cost._raw_integrated_cost(mitigation)),
            expected_integral,
            places=10,
        )
        self.assertNotAlmostEqual(
            float(cost._raw_integrated_cost(mitigation)),
            legacy_whole_curve_integral,
            places=6,
        )
        self.assertAlmostEqual(
            float(cost._raw_marginal_cost(mitigation)),
            cost.tau_0 * np.expm1(cost.power * mitigation)
            + cost.backstop_premium,
            places=10,
        )

    def test_reported_prices_use_the_objective_learning_state(self):
        """The reported node price must use prior emissions-weighted mitigation."""

        utility = make_tiny_real_utility()
        mitigation = np.array([0.4, 0.8, 0.2])
        output = ClimateOutput(utility)
        output.calculate_output(mitigation)

        for node in (1, 2):
            average = utility.damage.average_mitigation_node(
                mitigation, node, period=1
            )
            expected_price = utility.cost.price(
                utility.tree.decision_times[1], mitigation[node], average
            )
            self.assertAlmostEqual(output.ave_mitigations[node], average)
            self.assertAlmostEqual(output.prices[node], expected_price)

    def test_huber_smoothed_removal_premium_is_c1_and_economically_bounded(self):
        delta = 0.002
        sharp = make_tiny_real_utility().cost
        smooth = make_tiny_real_utility(
            backstop_smoothing_width=delta
        ).cost
        points = np.array([0.8, 1.0, 1.0 + delta, 1.2])

        np.testing.assert_allclose(
            smooth._raw_integrated_cost(points[:2]),
            sharp._raw_integrated_cost(points[:2]),
            rtol=0.0, atol=1e-14,
        )
        self.assertEqual(
            smooth.cost_formulation, "additive_removal_premium_huber_v1"
        )
        self.assertEqual(
            sharp.cost_formulation, "additive_removal_premium_v1"
        )
        self.assertAlmostEqual(
            float(sharp._raw_integrated_cost(1.0 + delta))
            - float(smooth._raw_integrated_cost(1.0 + delta)),
            sharp.backstop_premium * delta / 2.0,
            places=10,
        )
        self.assertAlmostEqual(
            float(smooth._raw_marginal_cost(1.0)),
            float(sharp._raw_marginal_cost(1.0)),
            places=12,
        )
        self.assertAlmostEqual(
            float(smooth._raw_marginal_cost(1.0 + delta))
            - float(sharp._raw_marginal_cost(1.0 + delta - 1e-10)),
            0.0,
            places=4,
        )

        # The joins are C1 but not C2, so a central difference at a join
        # has a curvature error. Boundary differentiability is checked above;
        # check numerical differentiation inside the transition instead.
        eps = delta * 1e-5
        for point in (1.0 + 0.5 * delta,):
            finite_difference = (
                float(smooth._raw_integrated_cost(point + eps))
                - float(smooth._raw_integrated_cost(point - eps))
            ) / (2.0 * eps)
            self.assertAlmostEqual(
                finite_difference,
                float(smooth._raw_marginal_cost(point)),
                places=4,
            )

    def test_symmetric_huber_removal_premium_is_c1_and_matches_sharp_outside_band(self):
        half_width = 0.004
        sharp = make_tiny_real_utility().cost
        smooth = make_tiny_real_utility(
            backstop_smoothing_width=half_width,
            backstop_smoothing_mode="symmetric_huber",
        ).cost
        points = np.array([0.8, 1.0 - half_width, 1.0 + half_width, 1.2])

        np.testing.assert_allclose(
            smooth._raw_integrated_cost(points),
            sharp._raw_integrated_cost(points),
            rtol=0.0, atol=1e-12,
        )
        self.assertEqual(
            smooth.cost_formulation,
            "additive_removal_premium_symmetric_huber_v1",
        )
        self.assertAlmostEqual(
            float(smooth._raw_integrated_cost(1.0))
            - float(sharp._raw_integrated_cost(1.0)),
            sharp.backstop_premium * half_width / 4.0,
            places=10,
        )
        calibrated_at_one = sharp.tau_0 * np.expm1(sharp.power)
        self.assertAlmostEqual(
            float(smooth._raw_marginal_cost(1.0)) - calibrated_at_one,
            sharp.backstop_premium / 2.0,
            places=10,
        )

        eps = half_width * 1e-5
        for point in (1.0,):
            finite_difference = (
                float(smooth._raw_integrated_cost(point + eps))
                - float(smooth._raw_integrated_cost(point - eps))
            ) / (2.0 * eps)
            self.assertAlmostEqual(
                finite_difference,
                float(smooth._raw_marginal_cost(point)),
                places=4,
            )

    def test_symmetric_huber_overmitigation_gradient_matches_finite_difference(self):
        half_width = 0.004
        utility = make_tiny_real_utility(
            backstop_smoothing_width=half_width,
            backstop_smoothing_mode="symmetric_huber",
        )
        objective = EZAdjointObjective(utility, value_parity_mode="always")
        eps = 1e-6
        for mitigation_level in (1.0,):
            mitigation = np.array([0.3, mitigation_level, 0.5])
            _, gradient = objective.value_and_gradient(mitigation)
            finite_difference = (
                objective.utility(mitigation + np.array([0.0, eps, 0.0]))
                - objective.utility(mitigation - np.array([0.0, eps, 0.0]))
            ) / (2.0 * eps)

            self.assertLess(abs(finite_difference - gradient[1]), 2e-6)
            self.assertEqual(objective.diagnostics()["num_cost_kink_nodes"], 0)

    def test_huber_smoothed_overmitigation_gradient_matches_finite_difference(self):
        utility = make_tiny_real_utility(backstop_smoothing_width=0.002)
        objective = EZAdjointObjective(utility, value_parity_mode="always")
        eps = 1e-6
        for mitigation_level in (1.001,):
            mitigation = np.array([0.3, mitigation_level, 0.5])
            _, gradient = objective.value_and_gradient(mitigation)
            finite_difference = (
                objective.utility(mitigation + np.array([0.0, eps, 0.0]))
                - objective.utility(mitigation - np.array([0.0, eps, 0.0]))
            ) / (2.0 * eps)

            self.assertLess(abs(finite_difference - gradient[1]), 2e-6)
            self.assertEqual(objective.diagnostics()["num_cost_kink_nodes"], 0)

    def test_overmitigation_value_and_gradient_match_legacy_and_forward(self):
        utility = make_tiny_real_utility()
        reference = EZForwardSensitivityObjective(utility)
        objective = EZAdjointObjective(utility, value_parity_mode="always")
        mitigation = np.array([0.3, 1.01, 0.5])

        reference_value, reference_gradient = reference.value_and_gradient(
            mitigation
        )
        value, gradient = objective.value_and_gradient(mitigation)
        legacy_value = objective.utility(mitigation)
        direction = np.array([0.0, 1.0, 0.0])
        eps = 1e-6

        finite_difference = (
            objective.utility(mitigation + eps * direction)
            - objective.utility(mitigation - eps * direction)
        ) / (2.0 * eps)

        self.assertAlmostEqual(value, legacy_value, places=10)
        self.assertAlmostEqual(reference_value, legacy_value, places=10)
        np.testing.assert_allclose(
            gradient, reference_gradient, rtol=1e-8, atol=1e-10
        )
        self.assertGreater(abs(finite_difference), 1e-4)
        self.assertLess(abs(finite_difference - gradient[1]), 2e-6)
        self.assertEqual(
            objective.diagnostics()["adjoint_value_parity_status"], "passed"
        )
        self.assertGreaterEqual(
            objective.diagnostics()["num_overmitigation_nodes"], 1
        )

    def test_sparse_decision_time_interpolation_is_differentiated(self):
        utility = make_tiny_real_utility([0, 5, 15, 25])
        objective = EZAdjointObjective(utility)
        mitigation = np.linspace(0.2, 0.8, utility.tree.num_decision_nodes)
        _, gradient = objective.value_and_gradient(mitigation)
        direction = np.arange(1, utility.tree.num_decision_nodes + 1, dtype=float)
        direction = direction / np.linalg.norm(direction)
        eps = 1e-4

        finite_difference = (
            objective.utility(mitigation + eps * direction)
            - objective.utility(mitigation - eps * direction)
        ) / (2.0 * eps)
        adjoint_dot = float(np.dot(gradient, direction))

        self.assertLess(objective.diagnostics()["adjoint_value_rel_diff"], 1e-10)
        self.assertLess(abs(finite_difference - adjoint_dot), 2e-7)

    def test_reverse_matches_forward_reference_on_fixed_grid(self):
        utility = make_tiny_real_utility([0, 5, 10, 15, 35, 75, 125, 175, 225])
        mitigation = np.linspace(
            0.2, 0.8, utility.tree.num_decision_nodes
        )
        reference = EZForwardSensitivityObjective(utility)
        objective = EZAdjointObjective(utility, value_parity_mode="off")

        reference_value, reference_gradient = reference.value_and_gradient(
            mitigation
        )
        value, gradient = objective.value_and_gradient(mitigation)

        self.assertLess(abs(value - reference_value), 1e-9)
        np.testing.assert_allclose(
            gradient, reference_gradient, rtol=1e-8, atol=1e-10
        )
        self.assertEqual(
            objective.diagnostics()["gradient_backend"], "reverse_adjoint"
        )
        self.assertFalse(
            objective.diagnostics()["adjoint_has_state_control_matrix"]
        )

    def test_eis_one_log_limit_matches_reference(self):
        utility = make_tiny_real_utility([0, 5, 15, 25])
        utility.r = 0.0
        utility.is_log_eis = True
        mitigation = np.linspace(
            0.25, 0.75, utility.tree.num_decision_nodes
        )
        reference = EZForwardSensitivityObjective(utility)
        objective = EZAdjointObjective(utility, value_parity_mode="always")

        reference_value, reference_gradient = reference.value_and_gradient(
            mitigation
        )
        value, gradient = objective.value_and_gradient(mitigation)

        self.assertLess(abs(value - reference_value), 1e-9)
        np.testing.assert_allclose(
            gradient, reference_gradient, rtol=1e-8, atol=1e-10
        )
        self.assertEqual(
            objective.diagnostics()["adjoint_value_parity_status"], "passed"
        )

    def test_parity_first_runs_only_once_and_kink_is_diagnosed(self):
        utility = make_tiny_real_utility()
        objective = EZAdjointObjective(utility, value_parity_mode="first")
        mitigation = np.array([1.0, 0.4, 0.5])

        objective.value_and_gradient(mitigation)
        first = objective.diagnostics()
        objective.value_and_gradient(mitigation)
        second = objective.diagnostics()

        self.assertEqual(first["adjoint_value_parity_status"], "passed")
        self.assertEqual(second["adjoint_value_parity_status"], "skipped")
        self.assertEqual(first["num_cost_kink_nodes"], 1)

    def test_cost_kink_diagnostic_uses_absolute_tolerance_only(self):
        utility = make_tiny_real_utility()
        cases = (
            (0.0, 1), (5e-9, 1), (-5e-9, 1),
            (5e-6, 0), (-5e-6, 0),
        )
        for delta, expected in cases:
            with self.subTest(delta=delta):
                objective = EZAdjointObjective(
                    utility, value_parity_mode="off"
                )
                objective.value_and_gradient(
                    np.array([1.0 + delta, 0.4, 0.5])
                )
                self.assertEqual(
                    objective.diagnostics()["num_cost_kink_nodes"], expected
                )

    def test_no_endogenous_learning_matches_reference(self):
        utility = make_tiny_real_utility([0, 5, 15, 25])
        utility.cost.tech_scale = 0.0
        mitigation = np.linspace(
            0.15, 0.85, utility.tree.num_decision_nodes
        )
        reference_gradient = EZForwardSensitivityObjective(utility).gradient(
            mitigation
        )
        gradient = EZAdjointObjective(
            utility, value_parity_mode="off"
        ).gradient(mitigation)

        np.testing.assert_allclose(
            gradient, reference_gradient, rtol=1e-8, atol=1e-10
        )

    def test_nonzero_interpolated_damages_match_reference(self):
        utility = make_tiny_real_utility([0, 5, 15, 25])
        damage_levels = np.array([0.01, 0.035, 0.09])
        for damage_index, damage_level in enumerate(damage_levels):
            utility.damage.d_rcomb[damage_index, :, :] = damage_level
        mitigation = np.linspace(
            0.2, 0.75, utility.tree.num_decision_nodes
        )

        reference_value, reference_gradient = (
            EZForwardSensitivityObjective(utility).value_and_gradient(mitigation)
        )
        objective = EZAdjointObjective(utility, value_parity_mode="always")
        value, gradient = objective.value_and_gradient(mitigation)

        self.assertLess(abs(value - reference_value), 1e-9)
        np.testing.assert_allclose(
            gradient, reference_gradient, rtol=1e-8, atol=1e-10
        )

    def test_zero_damage_override_removes_damage_channel(self):
        utility = make_tiny_real_utility([0, 5, 15, 25])
        utility.damage.zero_damage = True

        def zero_damage(self, m, period, is_last=False):
            return np.zeros(self.tree.get_num_nodes_period(period))

        utility.damage.damage_function = types.MethodType(
            zero_damage, utility.damage
        )
        mitigation = np.linspace(
            0.2, 0.7, utility.tree.num_decision_nodes
        )
        reference = EZForwardSensitivityObjective(utility)
        objective = EZAdjointObjective(utility, value_parity_mode="always")

        reference_value, reference_gradient = reference.value_and_gradient(
            mitigation
        )
        value, gradient = objective.value_and_gradient(mitigation)

        self.assertLess(abs(value - reference_value), 1e-9)
        np.testing.assert_allclose(
            gradient, reference_gradient, rtol=1e-8, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()

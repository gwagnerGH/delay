import unittest

import numpy as np

from src.cost import BPWCost
from src.tree import TreeModel


def make_cost(horizontal=0.0, vertical=0.0):
    return BPWCost(
        tree=TreeModel([0, 5, 10]),
        emit_at_0=40.0,
        baseline_num=2,
        tech_const=1.5,
        tech_scale=1.5,
        cons_at_0=86252.0,
        backstop_premium=10000.0,
        no_free_lunch=False,
        mac_horizontal_shift=horizontal,
        mac_vertical_shift=vertical,
    )


class TestMacShiftCost(unittest.TestCase):
    def test_zero_shifts_reproduce_legacy_curve_exactly(self):
        legacy = make_cost()
        explicit_zero = make_cost(0.0, 0.0)
        mitigation = np.array([0.0, 0.2, 0.7, 1.0])
        np.testing.assert_array_equal(
            legacy._raw_integrated_cost(mitigation),
            explicit_zero._raw_integrated_cost(mitigation),
        )
        np.testing.assert_array_equal(
            legacy._raw_marginal_cost(mitigation),
            explicit_zero._raw_marginal_cost(mitigation),
        )

    def test_shifted_cost_is_zero_at_zero_and_has_shifted_right_derivative(self):
        cost = make_cost(horizontal=0.2, vertical=50.0)
        self.assertEqual(float(cost._raw_integrated_cost(0.0)), 0.0)
        expected = cost.tau_0 * np.expm1(cost.power * 0.2) + 50.0
        self.assertAlmostEqual(float(cost._raw_marginal_cost(0.0)), expected)

    def test_shifted_marginal_cost_is_derivative_of_integrated_cost(self):
        cost = make_cost(horizontal=0.15, vertical=75.0)
        mitigation = 0.45
        epsilon = 1e-6
        finite_difference = (
            cost._raw_integrated_cost(mitigation + epsilon)
            - cost._raw_integrated_cost(mitigation - epsilon)
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(
            float(finite_difference),
            float(cost._raw_marginal_cost(mitigation)),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()

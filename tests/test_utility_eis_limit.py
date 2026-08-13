import unittest

import numpy as np

from src.utility import EZUtility
from src.tree import TreeModel


def utility_for_eis(eis, beta=0.93, growth=1.015):
    utility = EZUtility.__new__(EZUtility)
    utility.r = 1.0 - 1.0 / eis
    utility.is_log_eis = abs(utility.r) <= utility._LOG_EIS_TOL
    utility.b = beta
    utility.growth_term = growth
    return utility


class TestEISLimit(unittest.TestCase):
    def test_exact_log_limit(self):
        utility = utility_for_eis(1.0)
        consumption = np.array([2.0, 4.0])
        cert_equiv = np.array([8.0, 1.0])

        actual = utility._intertemporal_aggregate(consumption, cert_equiv)
        expected = consumption ** (1.0 - utility.b) * cert_equiv ** utility.b

        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)

    def test_intertemporal_aggregate_is_continuous_around_one(self):
        consumption = np.array([0.8, 1.2, 3.5])
        cert_equiv = np.array([1.7, 0.9, 2.2])
        center = utility_for_eis(1.0)._intertemporal_aggregate(
            consumption, cert_equiv
        )

        for eis in (0.9999, 1.0001):
            nearby = utility_for_eis(eis)._intertemporal_aggregate(
                consumption, cert_equiv
            )
            self.assertTrue(np.isfinite(nearby).all())
            np.testing.assert_allclose(nearby, center, rtol=2e-5, atol=0.0)

    def test_terminal_utility_is_continuous_around_one(self):
        consumption = np.array([0.8, 1.2, 3.5])
        center = utility_for_eis(1.0)._terminal_utility(consumption)

        for eis in (0.9999, 1.0001):
            nearby = utility_for_eis(eis)._terminal_utility(consumption)
            self.assertTrue(np.isfinite(nearby).all())
            np.testing.assert_allclose(nearby, center, rtol=2e-5, atol=0.0)

    def test_stable_formula_matches_legacy_formula_away_from_one(self):
        utility = utility_for_eis(0.8)
        consumption = np.array([0.8, 1.2, 3.5])
        cert_equiv = np.array([1.7, 0.9, 2.2])

        actual = utility._intertemporal_aggregate(consumption, cert_equiv)
        expected = (
            (1.0 - utility.b) * consumption ** utility.r
            + utility.b * cert_equiv ** utility.r
        ) ** (1.0 / utility.r)

        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)

    def test_full_utility_recursion_is_finite_and_continuous(self):
        tree = TreeModel([0, 5, 10])
        mitigation = np.zeros(tree.num_decision_nodes)

        class ZeroDamage:
            def average_mitigation(self, m, period, is_last=False):
                return np.zeros(tree.get_num_nodes_period(period))

            def damage_function(self, m, period, is_last=False):
                return np.zeros(tree.get_num_nodes_period(period))

        class ZeroCost:
            def cost(self, period, period_mitigation, period_ave_mitigation):
                return np.zeros_like(period_mitigation)

        values = []
        for eis in (0.9999, 1.0, 1.0001):
            utility = EZUtility(
                tree=tree,
                damage=ZeroDamage(),
                cost=ZeroCost(),
                period_len=5.0,
                eis=eis,
                ra=7.0,
                time_pref=0.005,
                cons_growth=0.015,
            )
            value = float(utility.utility(mitigation)[0])
            self.assertTrue(np.isfinite(value))
            values.append(value)

        np.testing.assert_allclose(values[0], values[1], rtol=3e-5, atol=0.0)
        np.testing.assert_allclose(values[2], values[1], rtol=3e-5, atol=0.0)

    def test_exact_compensation_adjustments_work_at_eis_one(self):
        tree = TreeModel([0, 5, 10])
        mitigation = np.zeros(tree.num_decision_nodes)

        class ZeroDamage:
            def average_mitigation(self, m, period, is_last=False):
                return np.zeros(tree.get_num_nodes_period(period))

            def damage_function(self, m, period, is_last=False):
                return np.zeros(tree.get_num_nodes_period(period))

        class ZeroCost:
            def cost(self, period, period_mitigation, period_ave_mitigation):
                return np.zeros_like(period_mitigation)

        utility = EZUtility(
            tree=tree,
            damage=ZeroDamage(),
            cost=ZeroCost(),
            period_len=5.0,
            eis=1.0,
            ra=7.0,
            time_pref=0.005,
            cons_growth=0.015,
        )
        base = float(utility.utility(mitigation)[0])
        first_period = float(
            utility.adjusted_utility(
                mitigation, first_period_consadj=0.01
            )[0]
        )
        first_five_years = float(
            utility.adjusted_utility(
                mitigation, period_consadj=np.array([0.01, 0.0, 0.0])
            )[0]
        )

        self.assertTrue(np.isfinite(first_period))
        self.assertTrue(np.isfinite(first_five_years))
        self.assertGreater(first_period, base)
        self.assertGreater(first_five_years, base)


if __name__ == "__main__":
    unittest.main()

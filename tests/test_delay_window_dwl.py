import unittest

import numpy as np

from src.analysis.delayed_action import find_consumption_equivalence
from src.tree import TreeModel
from src.utility import EZUtility


class ZeroDamage:
    def __init__(self, tree):
        self.tree = tree

    def average_mitigation(self, m, period, is_last=False):
        return np.zeros(self.tree.get_num_nodes_period(period))

    def damage_function(self, m, period, is_last=False):
        return np.zeros(self.tree.get_num_nodes_period(period))


class ZeroCost:
    def cost(self, period, period_mitigation, period_ave_mitigation):
        return np.zeros_like(period_mitigation)


def make_utility():
    tree = TreeModel([0, 5, 10, 15])
    utility = EZUtility(
        tree=tree,
        damage=ZeroDamage(tree),
        cost=ZeroCost(),
        period_len=5.0,
        eis=0.9,
        ra=7.0,
        time_pref=0.005,
        cons_growth=0.015,
    )
    return tree, utility, np.zeros(tree.num_decision_nodes)


class TargetUtility:
    def __init__(self, utility, multiplier):
        self.utility_object = utility
        self.multiplier = multiplier

    def utility(self, mitigation):
        return self.utility_object.adjusted_utility(
            mitigation, period_consmult=self.multiplier
        )


class TestDelayWindowDWL(unittest.TestCase):
    def test_multiplier_applies_to_selected_periods_and_nodes_only(self):
        tree, utility, mitigation = make_utility()
        base_consumption = utility.utility(mitigation, return_trees=True)['Consumption']
        multiplier = np.array([1.1, 1.1, 1.0, 1.0])
        _, adjusted_consumption, _, _ = utility.adjusted_utility(
            mitigation, period_consmult=multiplier, return_trees=True
        )

        for period in (0, 5):
            np.testing.assert_allclose(
                adjusted_consumption[period], 1.1 * base_consumption[period]
            )
        for period in (10, 15):
            np.testing.assert_allclose(
                adjusted_consumption[period], base_consumption[period]
            )

    def test_proportional_root_matches_target_for_all_delay_windows(self):
        tree, utility, mitigation = make_utility()
        n_periods = int(round(tree.decision_times[-1] / utility.period_len)) + 1
        for years, n_comp_periods in ((5, 1), (10, 2), (15, 3)):
            multiplier = np.ones(n_periods)
            multiplier[:n_comp_periods] = 1.1
            target = TargetUtility(utility, multiplier)
            result = find_consumption_equivalence(
                mitigation, mitigation, utility, target,
                method='delay_window_proportional', compensation_years=years,
            )
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result, 0.1, places=7)

    def test_legacy_first_period_additive_adjustment_is_unchanged(self):
        tree, utility, mitigation = make_utility()
        n_periods = int(round(tree.decision_times[-1] / utility.period_len)) + 1
        legacy = utility.adjusted_utility(
            mitigation, first_period_consadj=0.01
        )
        explicit = utility.adjusted_utility(
            mitigation, period_consadj=np.r_[0.01, np.zeros(n_periods - 1)]
        )
        np.testing.assert_allclose(legacy, explicit, rtol=1e-13, atol=0.0)

    def test_invalid_multiplier_is_rejected(self):
        _, utility, mitigation = make_utility()
        with self.assertRaises(ValueError):
            utility.adjusted_utility(
                mitigation, period_consmult=np.array([1.0, 0.0])
            )


if __name__ == '__main__':
    unittest.main()

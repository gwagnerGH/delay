import unittest

import numpy as np

from src.analysis.delayed_action import (
    FIXED_DELAY_DECISION_TIMES,
    FIXED_DELAY_EMISSIONS_TIME_STEP,
    fixed_delay_decision_times,
    get_delay_nodes_for_year,
    get_delay_periods_for_year,
)
from src.emit_baseline import BPWEmissionBaseline
from src.tree import TreeModel


class TestFixedLearningDelay(unittest.TestCase):
    def test_fixed_delay_grid_is_copied(self):
        decision_times = fixed_delay_decision_times()

        self.assertEqual(decision_times, FIXED_DELAY_DECISION_TIMES)
        self.assertIsNot(decision_times, FIXED_DELAY_DECISION_TIMES)

    def test_delay_periods_and_node_counts(self):
        tree = TreeModel(fixed_delay_decision_times())

        expected = {
            0: (0, 0),
            5: (1, 1),
            10: (2, 3),
            15: (3, 7),
        }
        for delay_year, (periods, node_count) in expected.items():
            with self.subTest(delay_year=delay_year):
                self.assertEqual(
                    get_delay_periods_for_year(tree.decision_times, delay_year),
                    periods,
                )
                self.assertEqual(
                    len(get_delay_nodes_for_year(tree, delay_year)),
                    node_count,
                )

    def test_delay_nodes_are_pre_reentry_only(self):
        tree = TreeModel(fixed_delay_decision_times())

        nodes = get_delay_nodes_for_year(tree, 15)
        node_years = [
            int(tree.decision_times[tree.get_period(node)])
            for node in nodes
        ]

        self.assertTrue(node_years)
        self.assertLess(max(node_years), 15)
        self.assertEqual(sorted(set(node_years)), [0, 5, 10])

    def test_invalid_delay_year_is_rejected(self):
        tree = TreeModel(fixed_delay_decision_times())

        with self.assertRaises(ValueError):
            get_delay_nodes_for_year(tree, 12)

    def test_emissions_baseline_contains_fixed_decision_years(self):
        tree = TreeModel(fixed_delay_decision_times())
        baseline = BPWEmissionBaseline(
            tree=tree,
            baseline_num=2,
            emissions_time_step=FIXED_DELAY_EMISSIONS_TIME_STEP,
        )

        baseline.baseline_emission_setup()

        np.testing.assert_array_equal(
            baseline.decision_years,
            tree.calendar_years,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from src.climate import BPWClimate
from src.damage import BPWDamage
from src.emit_baseline import BPWEmissionBaseline
from src.tree import TreeModel


class TestEmissionsIntervalAccounting(unittest.TestCase):
    def setUp(self):
        self.tree = TreeModel([0, 5, 10], base_year=2025)
        self.emit = BPWEmissionBaseline(
            tree=self.tree, baseline_num=2, emissions_time_step=5
        )
        self.emit.baseline_emission_setup()

    def _avoided_through_terminal(self, mitigation):
        baseline, _ = self.emit.get_mitigated_baseline(
            np.zeros(3), node=1, baseline="cumemit", is_last=True
        )
        mitigated, _ = self.emit.get_mitigated_baseline(
            mitigation, node=1, baseline="cumemit", is_last=True
        )
        return 1000.0 * (baseline[-1] - mitigated[-1])

    def test_each_control_receives_exactly_its_own_interval(self):
        avoided_root = self._avoided_through_terminal(np.array([1.0, 0.0, 0.0]))
        avoided_child = self._avoided_through_terminal(np.array([0.0, 1.0, 0.0]))

        indices = self.emit.dec_times_ind
        expected_root = np.trapezoid(
            self.emit.baseline_gtco2[indices[0]:indices[1] + 1],
            self.emit.times[indices[0]:indices[1] + 1],
        )
        expected_child = np.trapezoid(
            self.emit.baseline_gtco2[indices[1]:indices[2] + 1],
            self.emit.times[indices[1]:indices[2] + 1],
        )
        self.assertAlmostEqual(avoided_root, expected_root, places=10)
        self.assertAlmostEqual(avoided_child, expected_child, places=10)

    def test_policy_jump_is_represented_at_shared_boundary(self):
        flow, times = self.emit.get_mitigated_baseline(
            np.array([1.0, 0.0, 0.0]),
            node=1,
            baseline="gtco2",
            is_last=True,
        )
        boundary = np.flatnonzero(times == 2030)
        self.assertEqual(len(boundary), 2)
        self.assertEqual(flow[boundary[0]], 0.0)
        self.assertEqual(
            flow[boundary[1]],
            self.emit.baseline_gtco2[self.emit.dec_times_ind[1]],
        )

    def test_constant_policy_hits_its_damage_interpolation_knot(self):
        constants = np.array([0.9, 0.7, 0.0])
        damage = BPWDamage(
            tree=self.tree,
            emit_baseline=self.emit,
            climate=BPWClimate(self.tree, self.emit, draws=1),
            mitigation_constants=constants,
            draws=1,
        )
        knots = damage.mitigation_cumulative_emissions_knots(period=1)
        for index, mitigation in enumerate(constants):
            policy = np.full(self.tree.num_decision_nodes, mitigation)
            cumulative, _ = self.emit.get_mitigated_baseline(
                policy, node=1, baseline="cumemit"
            )
            self.assertAlmostEqual(cumulative[-1], knots[index], places=12)

    def test_constant_policy_does_not_mitigate_pre_model_carbon_stock(self):
        mitigation = 0.7
        cumulative = self.emit.get_mitigated_baseline(
            mitigation, node=None, baseline="cumemit"
        )
        expected = self.emit.CUMEMIT_BASE_YEAR + (1.0 - mitigation) * (
            self.emit.baseline_cumemit - self.emit.CUMEMIT_BASE_YEAR
        )

        np.testing.assert_allclose(cumulative, expected, rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(cumulative[0], self.emit.CUMEMIT_BASE_YEAR)


if __name__ == "__main__":
    unittest.main()

"""Common physical calendar checks for EZDelay tree-timing comparisons."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.revised_experiment import build_saved_revised_model
from src.storage_tree import BigStorageTree


class CommonCalendarClimateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.full_times = [0, 5, 10, 15, 35, 75, 125, 175, 225]
        cls.coarse_times = [0, 10, 35, 75, 125, 175, 225]
        cls.knots = np.linspace(4.0, 0.0, 21)
        k = np.arange(21)[:, None, None]
        state = np.arange(128)[None, :, None]
        period = np.arange(8)[None, None, :]+1
        cls.damage = (0.002+0.00015*(20-k)*period
                      +0.00004*(127-state)*period)
        cls.full_path = cls.root/"full.npz"
        cls.coarse_path = cls.root/"coarse.npz"
        np.savez(cls.full_path, damages=cls.damage, knots=cls.knots,
                 decision_times=cls.full_times)
        columns = [cls.full_times.index(t)-1 for t in cls.coarse_times[1:]]
        reduced = cls.damage[:, :, columns].reshape(21, 32, 4, 6).mean(axis=2)
        np.savez(cls.coarse_path, damages=reduced, knots=cls.knots,
                 decision_times=cls.coarse_times)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def model(cls, times, learning, common=True):
        return build_saved_revised_model(
            cls.full_path if times == cls.full_times else cls.coarse_path,
            decision_times=times, ra=15, eis=1.5, prtp=0.005,
            growth=0.015, tech_const=1.5, tech_scale=learning,
            common_damage_path=cls.full_path if common else None)

    def test_full_tree_decision_dates_match_inherited_physical_damages(self):
        standard = self.model(self.full_times, 0.0, common=False)
        calendar = self.model(self.full_times, 0.0, common=True)
        # At a simulated damage knot the physical-state and original
        # node-conditional interpolants must agree exactly.
        policy = np.full(calendar.tree.num_decision_nodes, 0.6)
        _, old_tape, *_ = standard._forward(policy)
        _, new_tape, *_ = calendar._forward(policy)
        for year in self.full_times:
            error = np.max(np.abs(old_tape["cons_tree"][year]
                                  -new_tape["cons_tree"][year]))
            self.assertLess(error, 1e-12)

    def test_fixed_policy_unconditional_consumption_is_calendar_invariant(self):
        averages = []
        for times in (self.full_times, self.coarse_times):
            model = self.model(times, 0.0)
            policy = np.full(model.tree.num_decision_nodes, 0.63)
            _, tape, *_ = model._forward(policy)
            flows = []
            for elapsed in BigStorageTree(5, model.tree.decision_times).periods:
                period = min(int(np.searchsorted(times, elapsed, side="right")-1),
                             model.tree.num_periods-1)
                values = tape["cons_tree"][elapsed]
                if len(values) == 2*model.tree.get_num_nodes_period(period):
                    period += 1
                first, _ = model.tree.get_nodes_in_period(period)
                probabilities = model.tree.node_prob[first:first+len(values)]
                flows.append(float(probabilities @ values))
            averages.append(flows)
        np.testing.assert_allclose(averages[0], averages[1], atol=1e-12, rtol=0)

    def test_climate_and_learning_gradients_match_finite_differences(self):
        for learning in (0.0, 1.5):
            model = self.model(self.coarse_times, learning)
            policy = np.full(model.tree.num_decision_nodes, 0.63)
            _, gradient = model.value_and_gradient(policy)
            for node in (0, 1, len(policy)//2, len(policy)-1):
                step = 1e-3
                values = []
                for multiple in (-2.0, -1.0, 1.0, 2.0):
                    point = policy.copy()
                    point[node] += multiple*step
                    values.append(model.utility(point))
                finite = (
                    values[0]-8.0*values[1]+8.0*values[2]-values[3]
                )/(12.0*step)
                np.testing.assert_allclose(
                    finite, gradient[node], rtol=1e-5, atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()

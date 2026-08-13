import unittest

import numpy as np

from src.analysis.delayed_action import get_delay_nodes
from src.climate import BPWClimate
from src.config import get_base_year_reference
from src.emit_baseline import BPWEmissionBaseline
from src.tree import TreeModel


HISTORICAL_DECISION_TIMES = [0, 15, 50, 55, 85, 125, 175, 225, 275]
HISTORICAL_DECISION_YEARS = [1975, 1990, 2025, 2030, 2060, 2100, 2150, 2200, 2250]


class TestHistoricalDelaySetup(unittest.TestCase):
    def historical_tree(self):
        return TreeModel(HISTORICAL_DECISION_TIMES, base_year=1975)

    def historical_baseline(self):
        baseline = BPWEmissionBaseline(
            tree=self.historical_tree(),
            baseline_num=2,
            emissions_time_step=1,
            baseline_source="historical_splice",
        )
        baseline.baseline_emission_setup()
        return baseline

    def test_1975_base_year_reference_exists(self):
        reference = get_base_year_reference(1975)

        self.assertEqual(reference.cumemit_reference_year, 1974)
        self.assertAlmostEqual(reference.cumemit_value, 1.096959078076)
        self.assertEqual(reference.co2_reference_year, 1975)
        self.assertAlmostEqual(reference.co2_concentration, 331.13)

    def test_historical_baseline_contains_exact_decision_years(self):
        baseline = self.historical_baseline()

        self.assertEqual(list(baseline.decision_years), HISTORICAL_DECISION_YEARS)
        self.assertTrue(
            set(HISTORICAL_DECISION_YEARS).issubset(set(baseline.times.astype(int)))
        )

    def test_historical_baseline_splices_observed_history_to_ssp2(self):
        baseline = self.historical_baseline()

        self.assertAlmostEqual(
            float(np.interp(1975, baseline.times, baseline.baseline_gtco2)),
            23.111973,
            places=6,
        )
        self.assertAlmostEqual(
            float(np.interp(2025, baseline.times, baseline.baseline_gtco2)),
            42.2,
            places=6,
        )
        self.assertAlmostEqual(
            float(np.interp(2030, baseline.times, baseline.baseline_gtco2)),
            43.476063,
            places=6,
        )

    def test_climate_uses_1975_concentration_reference(self):
        baseline = self.historical_baseline()
        climate = BPWClimate(baseline.tree, baseline, draws=10)

        self.assertEqual(climate.base_year, 1975)
        self.assertAlmostEqual(climate.C_BASE_YEAR, 331.13)

    def test_delay_nodes_constrain_1975_and_1990_blocks_only(self):
        tree = self.historical_tree()

        np.testing.assert_array_equal(get_delay_nodes(tree, 2), np.array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()

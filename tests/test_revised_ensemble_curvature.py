"""Check the revised eight-column ensemble and terminal-safe support."""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import main_ensemble_delayed_cluster as ensemble
from src.config.parameter_priors import PARAMETER_PRIOR_STDS, PARAMETER_PRIOR_INDEX


class RevisedEnsembleCurvatureTests(unittest.TestCase):
    def test_fixed_spreads_and_admissible_support(self):
        self.assertEqual(ensemble.DIMS, 8)
        self.assertEqual(ensemble.param_names[-1], "backstop_curvature")
        self.assertEqual(tuple(PARAMETER_PRIOR_STDS[[
            PARAMETER_PRIOR_INDEX[name] for name in ("RA", "EIS", "PRTP", "growth")
        ]]), (3.75, 0.5625, 0.00225, 0.00375))
        i = {name: ensemble.param_names.index(name) for name in ensemble.param_names}
        self.assertEqual((ensemble.lbs[i["RA"]], ensemble.ubs[i["RA"]]),
                         (1.0, 15.0))
        self.assertIn("RAmax15_v5", ensemble.get_sample_filename())
        self.assertEqual((ensemble.lbs[i["PRTP"]], ensemble.ubs[i["PRTP"]]),
                         (0.004, 0.02))
        from src.config.terminal_growth import TerminalGrowthSettings
        # Bounds alone no longer guarantee finiteness after fixing timing.
        self.assertFalse(TerminalGrowthSettings().admissible_mask(
            ensemble.lbs[i["RA"]], ensemble.ubs[i["EIS"]], ensemble.lbs[i["PRTP"]]))

    def test_cluster_sample_file_contains_effective_curvature_draws(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ensemble, "DATA_DIR", tmpdir), \
                 patch.object(ensemble, "N_SAMPLES", 64):
                draws = ensemble.load_or_generate_gaussian_samples()
                again = ensemble.load_or_generate_gaussian_samples()
        np.testing.assert_array_equal(draws, again)
        self.assertEqual(draws.shape, (64, 8))
        i = {name: ensemble.param_names.index(name) for name in ensemble.param_names}
        self.assertEqual(np.unique(draws[:, i["bs_premium"]]).size, 1)
        self.assertTrue(np.all(draws[:, i["RA"]] <= 15.0))
        self.assertGreater(np.std(draws[:, i["backstop_curvature"]]), 0.1)
        self.assertTrue(np.all(draws[:, i["backstop_curvature"]] >= 0.5))
        self.assertTrue(np.all(draws[:, i["backstop_curvature"]] <= 2.0))
        from src.config.terminal_growth import TerminalGrowthSettings
        self.assertTrue(np.all(TerminalGrowthSettings().admissible_mask(
            draws[:, i["RA"]], draws[:, i["EIS"]], draws[:, i["PRTP"]])))


if __name__ == "__main__":
    unittest.main()

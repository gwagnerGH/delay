import unittest

import numpy as np

from src.optimization import CandidateScreenedLBFGSB


class QuadraticUtility:
    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def utility(self, mitigation):
        mitigation = np.asarray(mitigation, dtype=float)
        return -float(np.sum((mitigation - self.target) ** 2))


class TestMandatoryStarts(unittest.TestCase):
    def _optimizer(self, mandatory_starts, n_local_starts=1):
        utility = QuadraticUtility([0.5, 0.5])
        return CandidateScreenedLBFGSB(
            utility=utility,
            lower_bounds=np.zeros(2),
            upper_bounds=np.ones(2),
            mandatory_starts=mandatory_starts,
            warm_starts=[np.array([0.5, 0.5])],
            n_candidates=1,
            n_local_starts=n_local_starts,
            max_candidates=1,
            max_local_starts=n_local_starts,
            maxiter=40,
            structured_start_count=0,
            warm_start_perturbations=0,
            start_boundary_epsilon=0.1,
            preserve_diverse_starts=False,
            local_start_max_utility_gap=0.0,
            local_start_max_relative_utility_gap=0.0,
            escalate_on_dispersion=False,
            perturbation_check=False,
            kkt_check=False,
            print_progress=False,
        )

    def test_exact_boundary_start_is_selected_before_ranking_and_filters(self):
        optimizer = self._optimizer([np.array([0.0, 1.0])])
        required, candidates = optimizer._candidate_starts(1)
        mandatory = [
            start for source, start in candidates
            if source == "mandatory_start_0"
        ]

        self.assertEqual(len(mandatory), 1)
        np.testing.assert_array_equal(mandatory[0], [0.0, 1.0])

        selected, utilities, selected_utilities, selected_sources = (
            optimizer._screen_candidates(
                candidates,
                n_local_starts=1,
                required_count=len(required),
            )
        )

        self.assertEqual(selected_sources, ["mandatory_start_0"])
        np.testing.assert_array_equal(selected[0], [0.0, 1.0])
        self.assertGreater(
            float(np.max(utilities)),
            float(selected_utilities[0]),
        )

    def test_run_records_that_mandatory_start_reached_local_solver(self):
        optimizer = self._optimizer([np.array([0.0, 1.0])])

        _, _, diagnostics = optimizer.run()

        self.assertEqual(diagnostics["n_mandatory_starts_requested"], 1)
        self.assertEqual(diagnostics["n_mandatory_start_candidates"], 1)
        self.assertEqual(diagnostics["n_mandatory_local_starts"], 1)
        self.assertTrue(diagnostics["mandatory_starts_selected"])
        self.assertEqual(
            diagnostics["_local_results"][0]["start_source"],
            "mandatory_start_0",
        )
        self.assertAlmostEqual(
            diagnostics["_local_results"][0]["start_utility"],
            -0.5,
        )

    def test_capacity_error_is_explicit(self):
        optimizer = self._optimizer(
            [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
            n_local_starts=1,
        )

        with self.assertRaisesRegex(
                ValueError, "cannot accommodate 2 mandatory starts"):
            optimizer.run()

    def test_invalid_mandatory_start_fails_at_construction(self):
        with self.assertRaisesRegex(ValueError, "within the optimizer bounds"):
            self._optimizer([np.array([0.0, 1.1])])


if __name__ == "__main__":
    unittest.main()

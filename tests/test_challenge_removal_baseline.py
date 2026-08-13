import unittest
from unittest.mock import patch

import numpy as np

from scripts.challenge_removal_baseline import (
    KINK_TOL,
    _masked_best_point_resolved,
    _run_masked_polish,
    classify_challenge,
    generate_removal_masks,
    generate_screen_seeds,
    node_period,
    period_nodes,
)


class RemovalMaskTests(unittest.TestCase):
    def setUp(self):
        self.incumbent = np.ones(31, dtype=float)
        self.incumbent[0] = 0.6
        self.incumbent[3] = 1.2
        self.ranked = [node for node in range(30, -1, -1)]

    def test_tree_helpers(self):
        self.assertEqual(node_period(0), 0)
        self.assertEqual(node_period(14), 3)
        self.assertEqual(period_nodes(2, 31), (3, 4, 5, 6))

    def test_removal_masks_are_deterministic_and_structured(self):
        first = generate_removal_masks(self.incumbent, self.ranked, [3])
        second = generate_removal_masks(self.incumbent, self.ranked, [3])
        self.assertEqual(first, second)
        families = {record["family"] for record in first}
        self.assertTrue({"topk", "period_block", "sibling", "path", "deep"} <= families)
        self.assertTrue(all(tuple(sorted(set(item["nodes"]))) == item["nodes"] for item in first))


    def test_removal_masks_preserve_probe_gain_ranking(self):
        incumbent = np.ones(15, dtype=float)
        ranked = [14, 2, 9, 1, 12]
        masks = generate_removal_masks(incumbent, ranked)
        top_two = next(item for item in masks if item["name"] == "top2_at_1.01")
        deep_top_one = next(item for item in masks if item["name"] == "deep_top1_at_1.20")
        self.assertEqual(top_two["nodes"], (2, 14))
        self.assertEqual(deep_top_one["nodes"], (14,))


class ScreenSeedTests(unittest.TestCase):
    def test_seeds_are_capped_deterministic_and_preserve_active_values(self):
        incumbent = np.ones(63, dtype=float)
        incumbent[:4] = [0.4, 0.7, 0.9, 1.25]
        backgrounds = [np.linspace(0.1, 1.0, 63), np.full(63, 0.8)]
        ranked = list(range(62, -1, -1))
        first = generate_screen_seeds(incumbent, backgrounds, ranked, [3], 40)
        second = generate_screen_seeds(incumbent, backgrounds, ranked, [3], 40)
        self.assertLessEqual(len(first), 40)
        self.assertEqual(
            [record["vector_sha256"] for record in first],
            [record["vector_sha256"] for record in second],
        )
        self.assertEqual(len(first), len({record["vector_sha256"] for record in first}))
        for record in first:
            self.assertEqual(record["vector"][3], incumbent[3])
            if record["family"] == "below_kink":
                non_active = np.delete(record["vector"], 3)
                self.assertLessEqual(float(np.max(non_active)), 1.0 + KINK_TOL)


class ChallengeClassificationTests(unittest.TestCase):
    def test_pass_requires_no_material_gain_and_all_resolved(self):
        result = classify_challenge(
            10.0,
            [
                {"name": "a", "utility": 10.0 + 0.5e-10, "resolved": True},
                {"name": "b", "utility": 9.0, "resolved": True},
            ],
        )
        self.assertEqual(result["status"], "pass")

    def test_ambiguous_gain_or_unresolved_branch_is_inconclusive(self):
        gain = classify_challenge(
            10.0,
            [{"name": "a", "utility": 10.0 + 5.0e-10, "resolved": True}],
        )
        unresolved = classify_challenge(
            10.0,
            [{"name": "a", "utility": 9.0, "resolved": False}],
        )
        self.assertEqual(gain["status"], "inconclusive")
        self.assertEqual(unresolved["status"], "inconclusive")

    def test_constructive_win_takes_precedence_over_unresolved_branch(self):
        result = classify_challenge(
            10.0,
            [
                {"name": "winner", "utility": 10.0 + 2.0e-8, "resolved": True},
                {"name": "failed", "utility": None, "resolved": False},
            ],
        )
        self.assertEqual(result["status"], "win")
        self.assertEqual(result["best_source"], "winner")


class MaskedPolishTests(unittest.TestCase):
    def test_challenge_and_incumbent_are_both_mandatory_starts(self):
        incumbent = np.array([0.5, 1.0, 1.0, 1.2])
        challenge = incumbent.copy()
        challenge[2] = 1.01
        seed = {
            "name": "challenge",
            "family": "path",
            "mask_nodes": (2,),
            "vector": challenge,
        }
        captured = {}

        class FakeOptimizer:
            def __init__(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

            def run(self):
                return incumbent.copy(), 10.0, {
                    "lbfgsb_best_result_accepted": True,
                    "gradient_validation_status": "passed",
                    "mandatory_starts_selected": True,
                    "stationarity_failed": False,
                    "mitigation_kink_failed": False,
                    "perturbation_pass": True,
                    "perturbation_failed": False,
                    "dispersion_failed": True,
                }

        with patch(
            "scripts.challenge_removal_baseline.EZAdjointObjective",
            return_value=object(),
        ), patch(
            "scripts.challenge_removal_baseline.CandidateScreenedLBFGSB",
            FakeOptimizer,
        ):
            result = _run_masked_polish(
                object(), seed, incumbent, [3], 1.5, 600, 7, True, 1.0e-8
            )

        mandatory = captured["kwargs"]["mandatory_starts"]
        self.assertEqual(len(mandatory), 2)
        np.testing.assert_array_equal(mandatory[0], challenge)
        np.testing.assert_array_equal(mandatory[1], incumbent)
        self.assertEqual(captured["kwargs"]["n_candidates"], 2)
        self.assertEqual(captured["kwargs"]["n_local_starts"], 2)
        self.assertEqual(captured["kwargs"]["utility_spread_tol"], 1.0e-6)
        self.assertTrue(result["resolved"])
        self.assertEqual(
            result["resolution_basis"],
            "best_point_certified_ignoring_cross_start_dispersion",
        )


    def test_best_point_gate_does_not_ignore_stationarity_or_perturbations(self):
        diagnostics = {
            "lbfgsb_best_result_accepted": True,
            "gradient_validation_status": "passed",
            "mandatory_starts_selected": True,
            "stationarity_failed": False,
            "mitigation_kink_failed": False,
            "perturbation_pass": True,
            "perturbation_failed": False,
            "dispersion_failed": True,
        }
        self.assertTrue(_masked_best_point_resolved(diagnostics))
        diagnostics["stationarity_failed"] = True
        self.assertFalse(_masked_best_point_resolved(diagnostics))
        diagnostics["stationarity_failed"] = False
        diagnostics["perturbation_failed"] = True
        self.assertFalse(_masked_best_point_resolved(diagnostics))


if __name__ == "__main__":
    unittest.main()

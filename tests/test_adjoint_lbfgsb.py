import os
import unittest
from unittest import mock

import numpy as np

from src.optimization import (
    CandidateScreenedLBFGSB,
    ObjectiveWithGradient,
    objective_with_gradient_from_utility,
    prolong_policy_nearest_ancestor,
)
from src.tree import TreeModel


class QuadraticObjective(ObjectiveWithGradient):
    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def utility(self, m):
        m = np.asarray(m, dtype=float)
        return -float(np.sum((m - self.target) ** 2))

    def gradient(self, m):
        m = np.asarray(m, dtype=float)
        return -2.0 * (m - self.target)


class BadGradientObjective(QuadraticObjective):
    def gradient(self, m):
        return np.zeros_like(np.asarray(m, dtype=float))


class NonFiniteGradientObjective(QuadraticObjective):
    def gradient(self, m):
        return np.full_like(np.asarray(m, dtype=float), np.nan)


class HighMitigationDomainObjective(QuadraticObjective):
    """Objective defined only in a high-mitigation interior region."""

    def utility(self, m):
        m = np.asarray(m, dtype=float)
        if np.any(m < 0.92):
            return -np.inf
        return super().utility(m)


class CountingQuadraticObjective(QuadraticObjective):
    def __init__(self, target):
        super().__init__(target)
        self.combined_calls = 0

    def value_and_gradient(self, m):
        self.combined_calls += 1
        return self.utility(m), self.gradient(m)


class KinkDiagnosticObjective(ObjectiveWithGradient):
    """One-dimensional objective with configurable cusp diagnostics."""

    def __init__(self, utility, gradient, damage_knot_count=0,
                 cost_kink_count=0):
        self._utility = utility
        self._gradient = float(gradient)
        self._damage_knot_count = int(damage_knot_count)
        self._cost_kink_count = int(cost_kink_count)

    def utility(self, m):
        return float(self._utility(float(np.asarray(m, dtype=float)[0])))

    def gradient(self, m):
        return np.array([self._gradient], dtype=float)

    def diagnostics(self):
        return {
            "num_damage_interp_knots": self._damage_knot_count,
            "num_cost_kink_nodes": self._cost_kink_count,
        }


class TestAdjointLBFGSB(unittest.TestCase):
    def _kkt_optimizer(self, objective, **kwargs):
        options = {
            "utility": objective,
            "lower_bounds": np.zeros(1),
            "upper_bounds": np.ones(1),
            "objective_with_gradient": objective,
            "gradient_mode": "adjoint",
            "kkt_check": True,
            "projected_gradient_tol": 1e-5,
            "nonsmooth_kkt_check": True,
            "nonsmooth_kkt_step": 1e-4,
            "nonsmooth_kkt_utility_gain_tol": 1e-12,
            "nonsmooth_kkt_max_coordinates": 4,
            "validate_gradient": False,
            "perturbation_check": False,
            "print_progress": False,
        }
        options.update(kwargs)
        return CandidateScreenedLBFGSB(**options)

    def test_nonsmooth_kkt_accepts_cusp_with_no_feasible_improvement(self):
        objective = KinkDiagnosticObjective(
            utility=lambda value: -abs(value - 0.5),
            gradient=2e-3,
            damage_knot_count=1,
        )
        diagnostics = self._kkt_optimizer(objective)._kkt_diagnostics(
            np.array([0.5])
        )

        self.assertFalse(diagnostics["smooth_projected_gradient_pass"])
        self.assertTrue(diagnostics["nonsmooth_kkt_pass"])
        self.assertTrue(diagnostics["nonsmooth_kkt_override_applied"])
        self.assertTrue(diagnostics["projected_gradient_pass"])
        self.assertEqual(
            diagnostics["nonsmooth_kkt_status"],
            "passed_no_improving_direction",
        )
        self.assertEqual(diagnostics["nonsmooth_kkt_detected_knots"], 1)
        self.assertEqual(
            diagnostics["nonsmooth_kkt_detected_damage_knots"], 1
        )
        self.assertEqual(
            diagnostics["nonsmooth_kkt_detected_cost_kink_nodes"], 0
        )
        self.assertEqual(diagnostics["nonsmooth_kkt_tested_coordinates"], 1)
        self.assertLessEqual(diagnostics["nonsmooth_kkt_max_utility_gain"], 0.0)

    def test_nonsmooth_kkt_accepts_reported_cost_kink_without_damage_knot(self):
        objective = KinkDiagnosticObjective(
            utility=lambda value: -abs(value - 0.5),
            gradient=2e-3,
            damage_knot_count=0,
            cost_kink_count=1,
        )
        diagnostics = self._kkt_optimizer(objective)._kkt_diagnostics(
            np.array([0.5])
        )

        self.assertFalse(diagnostics["smooth_projected_gradient_pass"])
        self.assertTrue(diagnostics["nonsmooth_kkt_pass"])
        self.assertTrue(diagnostics["nonsmooth_kkt_override_applied"])
        self.assertTrue(diagnostics["projected_gradient_pass"])
        self.assertEqual(
            diagnostics["nonsmooth_kkt_status"],
            "passed_no_improving_direction",
        )
        self.assertEqual(
            diagnostics["nonsmooth_kkt_detected_damage_knots"], 0
        )
        self.assertEqual(
            diagnostics["nonsmooth_kkt_detected_cost_kink_nodes"], 1
        )
        self.assertEqual(diagnostics["nonsmooth_kkt_tested_coordinates"], 1)
        self.assertLessEqual(diagnostics["nonsmooth_kkt_max_utility_gain"], 0.0)

    def test_nonsmooth_kkt_rejects_true_improving_direction(self):
        objective = KinkDiagnosticObjective(
            utility=lambda value: value,
            gradient=2e-3,
            damage_knot_count=0,
            cost_kink_count=1,
        )
        diagnostics = self._kkt_optimizer(objective)._kkt_diagnostics(
            np.array([0.5])
        )

        self.assertFalse(diagnostics["smooth_projected_gradient_pass"])
        self.assertFalse(diagnostics["nonsmooth_kkt_pass"])
        self.assertFalse(diagnostics["nonsmooth_kkt_override_applied"])
        self.assertFalse(diagnostics["projected_gradient_pass"])
        self.assertEqual(
            diagnostics["nonsmooth_kkt_status"],
            "failed_improving_direction",
        )
        self.assertGreater(
            diagnostics["nonsmooth_kkt_max_utility_gain"], 1e-12
        )

    def test_nonsmooth_kkt_cannot_override_without_reported_knot(self):
        objective = KinkDiagnosticObjective(
            utility=lambda value: -abs(value - 0.5),
            gradient=2e-3,
            damage_knot_count=0,
        )
        diagnostics = self._kkt_optimizer(objective)._kkt_diagnostics(
            np.array([0.5])
        )

        self.assertFalse(diagnostics["smooth_projected_gradient_pass"])
        self.assertFalse(diagnostics["nonsmooth_kkt_pass"])
        self.assertFalse(diagnostics["nonsmooth_kkt_override_applied"])
        self.assertFalse(diagnostics["projected_gradient_pass"])
        self.assertEqual(
            diagnostics["nonsmooth_kkt_status"],
            "skipped_no_damage_or_cost_kinks",
        )

    def test_validated_nonsmooth_cost_kink_can_pass_final_gate(self):
        diagnostics = {
            "final_utility_spread": 0.0,
            "effective_utility_spread_tol": 1e-8,
            "lbfgsb_best_result_accepted": True,
            "n_active_variables": 1,
            "kkt_check": True,
            "projected_gradient_pass": True,
            "lbfgsb_scipy_success": True,
            "all_active_free_at_mitigation_kink": True,
            "nonsmooth_kkt_pass": True,
            "perturbation_failed": False,
        }

        result = CandidateScreenedLBFGSB._finalize_lbfgsb_diagnostics(
            diagnostics
        )

        self.assertFalse(result["mitigation_kink_failed"])
        self.assertTrue(result["lbfgsb_converged"])
        self.assertTrue(result["lbfgsb_success"])

    def test_exact_gradient_optimizer_solves_quadratic(self):
        target = np.array([0.2, 0.7, 1.2, 0.4])
        objective = QuadraticObjective(target)
        optimizer = CandidateScreenedLBFGSB(
            utility=objective,
            lower_bounds=np.zeros(len(target)),
            upper_bounds=np.full(len(target), 1.5),
            objective_with_gradient=objective,
            gradient_mode="adjoint",
            n_candidates=16,
            n_local_starts=4,
            max_candidates=16,
            max_local_starts=4,
            structured_start_count=4,
            validate_gradient=True,
            perturbation_check=True,
            kkt_check=True,
            projected_gradient_tol=1e-5,
            print_progress=False,
        )

        m, utility, diag = optimizer.run()

        np.testing.assert_allclose(m, target, atol=1e-5, rtol=0.0)
        self.assertGreaterEqual(utility, -1e-10)
        self.assertEqual(diag["optimizer"], "adjoint_lbfgsb")
        self.assertEqual(diag["gradient_mode"], "adjoint")
        self.assertEqual(diag["gradient_validation_status"], "passed")
        self.assertTrue(diag["lbfgsb_success"])
        self.assertTrue(diag["success_diagnostics"])
        self.assertLessEqual(diag["projected_grad_inf_norm"], 1e-5)

    def test_gradient_validation_fails_bad_gradient(self):
        target = np.array([0.2, 0.7, 1.2])
        objective = BadGradientObjective(target)
        optimizer = CandidateScreenedLBFGSB(
            utility=objective,
            lower_bounds=np.zeros(len(target)),
            upper_bounds=np.full(len(target), 1.5),
            objective_with_gradient=objective,
            gradient_mode="adjoint",
            n_candidates=8,
            n_local_starts=2,
            structured_start_count=2,
            validate_gradient=True,
            print_progress=False,
        )

        with self.assertRaises(RuntimeError):
            optimizer.run()

    def test_gradient_validation_searches_high_finite_interior_region(self):
        target = np.array([0.9, 0.9, 0.9])
        objective = HighMitigationDomainObjective(target)
        optimizer = CandidateScreenedLBFGSB(
            utility=objective,
            lower_bounds=np.zeros(len(target)),
            upper_bounds=np.ones(len(target)),
            objective_with_gradient=objective,
            gradient_mode="adjoint",
            validate_gradient=True,
            perturbation_check=False,
            print_progress=False,
        )

        point = optimizer._gradient_validation_point()
        self.assertIsNotNone(point)
        self.assertTrue(np.all(point >= 0.95))
        self.assertEqual(
            optimizer._validate_adjoint_gradient()["gradient_validation_status"],
            "passed",
        )

    def test_nonfinite_adjoint_gradient_rejects_only_the_local_start(self):
        target = np.array([0.2, 0.7, 1.2])
        objective = NonFiniteGradientObjective(target)
        optimizer = CandidateScreenedLBFGSB(
            utility=objective,
            lower_bounds=np.zeros(len(target)),
            upper_bounds=np.full(len(target), 1.5),
            objective_with_gradient=objective,
            gradient_mode="adjoint",
            n_candidates=8,
            n_local_starts=2,
            max_candidates=8,
            max_local_starts=2,
            structured_start_count=2,
            validate_gradient=False,
            perturbation_check=False,
            kkt_check=True,
            print_progress=False,
        )

        mitigation, utility, diagnostics = optimizer.run()

        self.assertTrue(np.all(np.isfinite(mitigation)))
        self.assertTrue(np.isfinite(utility))
        self.assertGreaterEqual(
            diagnostics["n_nonfinite_adjoint_gradient_rejections"], 1
        )
        self.assertTrue(diagnostics["fallback_used"])
        self.assertFalse(diagnostics["lbfgsb_success"])
        self.assertEqual(
            diagnostics["kkt_status"], "skipped_nonfinite_adjoint_gradient"
        )

    def test_fixed_nodes_are_not_active_gradient_variables(self):
        target = np.array([0.2, 0.7, 1.2])
        objective = QuadraticObjective(target)
        lower = np.array([0.0, 0.4, 0.0])
        upper = np.array([1.5, 0.4, 1.5])
        optimizer = CandidateScreenedLBFGSB(
            utility=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            objective_with_gradient=objective,
            gradient_mode="adjoint",
            n_candidates=8,
            n_local_starts=2,
            structured_start_count=2,
            validate_gradient=True,
            perturbation_check=False,
            print_progress=False,
        )

        m, _, diag = optimizer.run()

        self.assertEqual(diag["n_active_variables"], 2)
        self.assertEqual(diag["n_fixed_variables"], 1)
        self.assertAlmostEqual(m[1], 0.4)

    def test_combined_callback_respects_partial_upper_bounds(self):
        target = np.array([0.8, 1.2, 0.6])
        objective = CountingQuadraticObjective(target)
        upper = np.array([0.4, 1.5, 0.3])
        optimizer = CandidateScreenedLBFGSB(
            utility=objective,
            lower_bounds=np.zeros(len(target)),
            upper_bounds=upper,
            objective_with_gradient=objective,
            gradient_mode="adjoint",
            n_candidates=8,
            n_local_starts=2,
            max_candidates=8,
            max_local_starts=2,
            structured_start_count=2,
            local_start_workers=2,
            validate_gradient=True,
            perturbation_check=False,
            kkt_check=True,
            print_progress=False,
        )

        mitigation, _, diagnostics = optimizer.run()

        np.testing.assert_allclose(
            mitigation, np.minimum(target, upper), atol=1e-5, rtol=0.0
        )
        self.assertGreater(objective.combined_calls, 0)
        self.assertTrue(diagnostics["lbfgsb_success"])

    def test_missing_gradient_fails_fast(self):
        class UtilityOnly:
            def utility(self, m):
                return np.array([1.0])

        with self.assertRaises(NotImplementedError):
            objective_with_gradient_from_utility(UtilityOnly())

    def test_coarse_to_fine_mapping_is_deterministic(self):
        source_tree = TreeModel([0, 10, 20])
        target_tree = TreeModel([0, 5, 10, 15, 20])
        source_policy = np.arange(source_tree.num_decision_nodes, dtype=float)

        mapped = prolong_policy_nearest_ancestor(
            source_policy, source_tree, target_tree
        )

        self.assertEqual(len(mapped), target_tree.num_decision_nodes)
        self.assertAlmostEqual(mapped[0], source_policy[0])
        np.testing.assert_allclose(
            mapped,
            prolong_policy_nearest_ancestor(source_policy, source_tree, target_tree),
        )


    def test_main_policy_cap_is_uniform_and_fixed_nodes_are_preserved(self):
        import main_ensemble_delayed_cluster as ensemble

        captured = {}

        class FakeOptimizer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self):
                return np.zeros(4), 0.0, {}

        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "0.9",
            "RANDOM_SEED_BASE": "12345",
            "LBFGSB_NONSMOOTH_KKT_CHECK": "0",
            "LBFGSB_NONSMOOTH_KKT_STEP": "0.0003",
            "LBFGSB_NONSMOOTH_KKT_UTILITY_GAIN_TOL": "4e-11",
            "LBFGSB_NONSMOOTH_KKT_MAX_COORDINATES": "9",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
                ensemble, "CandidateScreenedLBFGSB", FakeOptimizer):
            _, _, diagnostics = ensemble.run_lbfgsb_policy(
                utility=object(),
                num_nodes=4,
                scenario_name="optimal",
                fixed_indices=[1],
                fixed_values=[1.2],
                upper_bounds=[1.4, 1.4, 0.7, 2.0],
                seed_parts=("benchmark",),
            )

        np.testing.assert_allclose(captured["lower_bounds"], [0.0, 1.2, 0.0, 0.0])
        np.testing.assert_allclose(captured["upper_bounds"], [0.9, 1.2, 0.7, 0.9])
        self.assertEqual(diagnostics["lbfgsb_policy_upper_bound"], 0.9)
        self.assertFalse(captured["nonsmooth_kkt_check"])
        self.assertEqual(captured["nonsmooth_kkt_step"], 0.0003)
        self.assertEqual(captured["nonsmooth_kkt_utility_gain_tol"], 4e-11)
        self.assertEqual(captured["nonsmooth_kkt_max_coordinates"], 9)
        self.assertEqual(
            diagnostics["lbfgsb_seed"],
            ensemble.stable_seed("12345", "optimal", "benchmark"),
        )

        with mock.patch.dict(os.environ, {"LBFGSB_POLICY_UPPER_BOUND": ""}):
            _, default_upper = ensemble.make_policy_bounds(2)
        np.testing.assert_allclose(default_upper, [1.5, 1.5])

        configured_env = {
            "OPTIMIZER": "ga_adjoint_lbfgsb",
            "N_CANDIDATES": "32",
            "N_LOCAL_STARTS": "3",
            "MAX_CANDIDATES": "64",
            "MAX_LOCAL_STARTS": "5",
            "LBFGSB_MAXITER": "80",
            "LBFGSB_FTOL": "1e-8",
            "LBFGSB_GTOL": "2e-6",
            "LBFGSB_N_WORKERS": "6",
            "LBFGSB_SCREENING_WORKERS": "5",
            "LBFGSB_GRADIENT_WORKERS": "4",
            "LBFGSB_LOCAL_START_WORKERS": "2",
            "ESCALATE_ON_DISPERSION": "0",
            "LBFGSB_KKT_CHECK": "0",
            "LBFGSB_NONSMOOTH_KKT_CHECK": "1",
            "LBFGSB_NONSMOOTH_KKT_STEP": "0.002",
            "LBFGSB_NONSMOOTH_KKT_UTILITY_GAIN_TOL": "3e-9",
            "LBFGSB_NONSMOOTH_KKT_MAX_COORDINATES": "7",
            "ADJOINT_VALIDATE_GRADIENT": "1",
            "GA_ADJOINT_POP_AMOUNT": "120",
            "GA_ADJOINT_GENERATIONS": "90",
            "GA_ADJOINT_TOP_STARTS": "7",
            "GA_ADJOINT_DIVERSE_STARTS": "6",
            "LBFGSB_POLICY_UPPER_BOUND": "0.99",
        }
        with mock.patch.dict(os.environ, configured_env):
            configured = ensemble.configured_optimizer_diagnostics(200, 150, 8)
        self.assertEqual(configured["configured_n_candidates"], 32)
        self.assertEqual(configured["configured_n_local_starts"], 3)
        self.assertEqual(configured["configured_max_candidates"], 64)
        self.assertEqual(configured["configured_max_local_starts"], 5)
        self.assertEqual(configured["configured_lbfgsb_maxiter"], 80)
        self.assertEqual(configured["configured_lbfgsb_n_workers"], 6)
        self.assertEqual(configured["configured_lbfgsb_screening_workers"], 5)
        self.assertEqual(configured["configured_lbfgsb_gradient_workers"], 4)
        self.assertEqual(configured["configured_lbfgsb_local_start_workers"], 2)
        self.assertFalse(configured["configured_escalate_on_dispersion"])
        self.assertFalse(configured["configured_lbfgsb_kkt_check"])
        self.assertTrue(configured["configured_lbfgsb_nonsmooth_kkt_check"])
        self.assertEqual(configured["configured_lbfgsb_nonsmooth_kkt_step"], 0.002)
        self.assertEqual(
            configured["configured_lbfgsb_nonsmooth_kkt_utility_gain_tol"],
            3e-9,
        )
        self.assertEqual(
            configured["configured_lbfgsb_nonsmooth_kkt_max_coordinates"], 7
        )
        self.assertTrue(configured["configured_adjoint_validate_gradient"])
        self.assertEqual(configured["configured_ga_adjoint_pop_amount"], 120)
        self.assertEqual(configured["configured_ga_adjoint_generations"], 90)
        self.assertEqual(configured["configured_ga_adjoint_top_starts"], 7)
        self.assertEqual(configured["configured_ga_adjoint_diverse_starts"], 6)
        self.assertEqual(configured["configured_lbfgsb_policy_upper_bound"], 0.99)

    def test_baseline_outputs_append_results_and_node_prices(self):
        import csv as csv_module
        import hashlib as hashlib_module
        import json as json_module
        import tempfile

        import main_ensemble_delayed_cluster as ensemble

        class FakeTree:
            base_year = 2025
            decision_times = [0, 5]
            num_decision_nodes = 2
            node_prob = np.array([1.0, 0.5])

            def get_period(self, node):
                return node

            def get_state(self, node, period):
                return 0

        class FakeDamage:
            def _damage_function_node(self, mitigation, node):
                return [0.01, 0.02][node]

            def climate_damage_node(self, mitigation, node):
                return [0.005, 0.015][node]

        class FakeUtility:
            damage = FakeDamage()

            def utility(self, mitigation, return_trees=False):
                class FakeConsumptionTree:
                    tree = {
                        0: np.array([1.0]),
                        5: np.array([2.0]),
                    }

                if return_trees:
                    return {"Consumption": FakeConsumptionTree()}
                return 1.25

        class FakeClimateOutput:
            prices = np.array([10.0, 20.0])
            ave_mitigations = np.array([0.2, 0.4])
            ghg_levels = np.array([400.0, 410.0])
            utility = FakeUtility()

        model_params = {
            "ra": 7.0,
            "eis": 0.9,
            "pref": 0.005,
            "tech_chg": 1.5,
            "tech_scale": 0.0,
            "bs_premium": 1.0,
            "growth": 0.015,
            "baseline_num": 2,
            "dam_func": 0,
            "tip_on": 1,
            "d_unc": 1,
            "t_unc": 1,
            "no_free_lunch": False,
            "period_len": 5.0,
            "emissions_time_step": 5.0,
            "damage_file_tag": "TEST",
            "zero_climate_damages": False,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            damage_dir = os.path.join(temp_dir, "data")
            os.makedirs(damage_dir)
            artifact_path = os.path.join(damage_dir, "artifact.csv")
            artifact_bytes = b"exact damage artifact\n"
            with open(artifact_path, "wb") as handle:
                handle.write(artifact_bytes)
            expected_sha = hashlib_module.sha256(artifact_bytes).hexdigest()
            output_dir = os.path.join(temp_dir, "outputs")

            with mock.patch.object(ensemble, "PROJECT_ROOT", temp_dir), mock.patch.object(
                    ensemble, "DATA_DIR", output_dir), mock.patch.dict(
                    os.environ, {
                        "OPTIMIZER": "ga_gs",
                        "RANDOM_SEED_BASE": "777",
                        "JOB_ID": "42",
                    }):
                for _ in range(2):
                    ensemble.persist_baseline_outputs(
                        sample_id="main",
                        delay_year=5,
                        task_id="1",
                        out_folder="bench",
                        run_type="delay_frontier",
                        comparison_type="fixed_learning_grid",
                        tree_spec="default",
                        decision_times_label="0|5",
                        tree=FakeTree(),
                        climate_output=FakeClimateOutput(),
                        mitigation=np.array([0.2, 0.4]),
                        utility_value=1.25,
                        model_params=model_params,
                        prob_scale_baseline=1.0,
                        output_metadata={"experiment": "append_test"},
                        solver_diagnostics={
                            "n_generations_ga": 10,
                            "seed_optimal": np.int64(123),
                        },
                        damage_filename="artifact",
                        runtime_seconds=2.5,
                    )

            analysis_dir = os.path.join(output_dir, "bench", "analysis")
            with open(os.path.join(analysis_dir, "bench_baseline_results.csv"), newline="") as handle:
                result_rows = list(csv_module.DictReader(handle))
            self.assertEqual(len(result_rows), 2)
            self.assertEqual(result_rows[0]["random_seed_base"], "777")
            self.assertEqual(result_rows[0]["optimizer"], "ga_gs")
            self.assertEqual(result_rows[0]["decision_times"], "0|5")
            self.assertEqual(result_rows[0]["damage_artifact_sha256"], expected_sha)
            self.assertIn("code_worktree_dirty", result_rows[0])
            self.assertIn("code_tracked_diff_sha256", result_rows[0])
            self.assertEqual(result_rows[0]["mitigation"], "0.20000000000000001|0.40000000000000002")
            diagnostics = json_module.loads(result_rows[0]["solver_diagnostics_json"])
            self.assertEqual(diagnostics["seed_optimal"], 123)

            with open(os.path.join(analysis_dir, "bench_baseline_node_prices.csv"), newline="") as handle:
                node_rows = list(csv_module.DictReader(handle))
            self.assertEqual(len(node_rows), 4)
            self.assertTrue(all(row["scenario"] == "optimal" for row in node_rows))
            self.assertTrue(all(row["damage_artifact_sha256"] == expected_sha for row in node_rows))
            self.assertEqual(
                [float(row["consumption"]) for row in node_rows[:2]],
                [1.0, 2.0],
            )
            self.assertEqual(
                [float(row["damage"]) for row in node_rows[:2]],
                [0.01, 0.02],
            )
            self.assertEqual(
                [float(row["climate_damage"]) for row in node_rows[:2]],
                [0.005, 0.015],
            )


    def test_required_damage_import_configuration_is_opt_in(self):
        import main_ensemble_delayed_cluster as ensemble

        with mock.patch.dict(os.environ, {"REQUIRE_DAMAGE_IMPORT": "0"}):
            ensemble.validate_damage_import_configuration(False)
            self.assertFalse(ensemble.require_damage_import_enabled())

        with mock.patch.dict(os.environ, {"REQUIRE_DAMAGE_IMPORT": "1"}):
            self.assertTrue(ensemble.require_damage_import_enabled())
            with self.assertRaisesRegex(
                    ValueError, "requires IMPORT_DAMAGES=1"):
                ensemble.validate_damage_import_configuration(False)
            ensemble.validate_damage_import_configuration(True)

    def test_required_damage_import_failure_is_fail_hard(self):
        import main_ensemble_delayed_cluster as ensemble

        import_error = IOError("missing artifact")
        with mock.patch.dict(os.environ, {"REQUIRE_DAMAGE_IMPORT": "0"}):
            self.assertIsNone(
                ensemble.raise_required_damage_import_failure(
                    import_error, "baseline artifact"
                )
            )

        with mock.patch.dict(os.environ, {"REQUIRE_DAMAGE_IMPORT": "1"}):
            with self.assertRaisesRegex(
                    RuntimeError, "refusing to fall back") as caught:
                ensemble.raise_required_damage_import_failure(
                    import_error, "coarse artifact"
                )
        self.assertIs(caught.exception.__cause__, import_error)


if __name__ == "__main__":
    unittest.main()

import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import main_ensemble_delayed_cluster as ensemble
from src.gen_samples import generate_gaussian_samples


class QuadraticPolicyUtility:
    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def utility(self, mitigation):
        mitigation = np.asarray(mitigation, dtype=float)
        return -float(np.sum((mitigation - self.target) ** 2))


class ConfigurableQuadraticPolicyUtility(QuadraticPolicyUtility):
    class Cost:
        backstop_smoothing_width = 0.0
        backstop_smoothing_mode = "one_sided_huber"

    def __init__(self, target):
        super().__init__(target)
        self.cost = self.Cost()


def successful_diagnostics(mitigation, source="fake"):
    return {
        "lbfgsb_success": True,
        "success_diagnostics": True,
        "mandatory_starts_selected": True,
        "runtime_seconds": 1.0,
        "_local_results": [{
            "m": np.asarray(mitigation, dtype=float),
            "start_source": source,
            "utility": 0.0,
            "start_utility": 0.0,
            "success": True,
            "scipy_success": True,
            "guarded_start_kept": False,
            "nfev": 1,
            "ngev": 1,
            "nit": 1,
            "runtime_seconds": 1.0,
        }],
    }


class TestRemovalActiveSet(unittest.TestCase):
    
    def _valid_replay_rows(self):
        diagnostics = {
            "configured_lbfgsb_removal_active_set": True,
            "configured_lbfgsb_policy_upper_bound": 1.5,
            "optimal_lbfgsb_success": True,
            "optimal_success_diagnostics": True,
            "optimal_gradient_mode": "adjoint",
            "optimal_gradient_validation_status": "passed",
            "optimal_lbfgsb_policy_upper_bound": 1.5,
            "optimal_removal_active_set_enabled": True,
            "optimal_removal_active_set_status": "passed",
            "optimal_removal_active_set_pass": True,
            "optimal_removal_active_set_accepted_stages_success": True,
            "optimal_removal_active_set_all_probe_evals_finite": True,
            "optimal_removal_active_set_full_probe_coverage_complete": True,
            "optimal_removal_active_set_final_audit_complete": True,
            "optimal_removal_active_set_final_coverage_complete": True,
            "optimal_removal_active_set_final_all_scales_tested": True,
            "optimal_removal_active_set_no_improving_inactive_nodes": True,
            "optimal_removal_active_set_all_mandatory_starts_selected": True,
            "optimal_removal_active_set_final_cap": 1.5,
            "optimal_removal_active_set_full_domain_upper_bound_max": 1.5,
            "optimal_removal_active_set_final_max_inactive_gain": -1e-9,
            "optimal_removal_active_set_gain_tol": 1e-10,
            "optimal_removal_active_set_final_gain_exceedance_count": 0,
        }
        common = {
            "scenario": "optimal",
            "delay_year": "5",
            "decision_times": "0|5",
            "sample_index": "run0",
            "tree_spec": "default",
            "task_id": "1",
            "job_id": "123",
            "baseline_only": "True",
            "lbfgsb_policy_upper_bound": "1.5",
            "cost_formulation": "additive_removal_premium_v1",
            "bs_premium": "10000",
            "code_revision": "revision",
            "code_worktree_dirty": "True",
            "code_tracked_diff_sha256": "diff-sha",
            "damage_artifact_filename": "damage.csv",
            "damage_artifact_sha256": "damage-sha",
            "damage_artifact_size_bytes": "123",
            "solver_diagnostics_json": json.dumps(diagnostics),
        }
        return [
            dict(common, node="0", mitigation="0.25"),
            dict(common, node="1", mitigation="1.5"),
        ]

    
    def _write_rows(self, directory, rows):
        path = Path(directory) / "node_prices.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    
    def _load_strict(
            self, path, extra_env=None, expected_smoothing_width=None,
            expected_smoothing_mode=None):
        env = {
            "EXTERNAL_OPTIMAL_WARM_START_NODE_PRICES": path,
            "EXTERNAL_OPTIMAL_WARM_START_TASK_ID": "1",
            "REPLAY_EXTERNAL_OPTIMAL_BASELINE": "1",
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
        }
        if extra_env:
            env.update(extra_env)
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            ensemble, "current_code_revision", return_value="revision"
        ), mock.patch.object(
            ensemble,
            "current_code_worktree_metadata",
            return_value={
                "code_worktree_dirty": True,
                "code_tracked_diff_sha256": "diff-sha",
            },
        ), mock.patch.object(
            ensemble,
            "damage_artifact_metadata",
            return_value={
                "damage_artifact_filename": "damage.csv",
                "damage_artifact_sha256": "damage-sha",
                "damage_artifact_size_bytes": 123,
            },
        ):
            return ensemble.load_external_optimal_warm_start(
                2,
                "0|5",
                5,
                sample_id="run0",
                tree_spec="default",
                expected_backstop_premium=10000.0,
                expected_backstop_smoothing_width=expected_smoothing_width,
                expected_backstop_smoothing_mode=expected_smoothing_mode,
                damage_filename="damage",
            )

    def test_smooth_replay_accepts_a_smooth_certificate(self):
        rows = self._valid_replay_rows()
        diagnostics = json.loads(rows[0]["solver_diagnostics_json"])
        diagnostics.update({
            "configured_lbfgsb_removal_active_set": False,
            "optimal_removal_active_set_enabled": False,
            "optimal_removal_active_set_pass": True,
            "optimal_removal_active_set_not_required": True,
            "optimal_removal_active_set_status": "not_required_smooth_premium",
        })
        for row in rows:
            row["cost_formulation"] = "additive_removal_premium_huber_v1"
            row["backstop_smoothing_width"] = "0.002"
            row["backstop_smoothing_mode"] = "one_sided_huber"
            row["solver_diagnostics_json"] = json.dumps(diagnostics)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(directory, rows)
            loaded = self._load_strict(
                path,
                extra_env={"BACKSTOP_SMOOTHING_WIDTH": "0.002"},
                expected_smoothing_width=0.002,
                expected_smoothing_mode="one_sided_huber",
            )

        np.testing.assert_allclose(loaded, [0.25, 1.5])

    def test_symmetric_smooth_replay_accepts_a_symmetric_certificate(self):
        rows = self._valid_replay_rows()
        diagnostics = json.loads(rows[0]["solver_diagnostics_json"])
        diagnostics.update({
            "configured_lbfgsb_removal_active_set": False,
            "optimal_removal_active_set_enabled": False,
            "optimal_removal_active_set_pass": True,
            "optimal_removal_active_set_not_required": True,
            "optimal_removal_active_set_status": "not_required_smooth_premium",
        })
        for row in rows:
            row["cost_formulation"] = (
                "additive_removal_premium_symmetric_huber_v1"
            )
            row["backstop_smoothing_width"] = "0.02"
            row["backstop_smoothing_mode"] = "symmetric_huber"
            row["solver_diagnostics_json"] = json.dumps(diagnostics)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(directory, rows)
            loaded = self._load_strict(
                path,
                extra_env={
                    "BACKSTOP_SMOOTHING_WIDTH": "0.02",
                    "BACKSTOP_SMOOTHING_MODE": "symmetric_huber",
                },
                expected_smoothing_width=0.02,
                expected_smoothing_mode="symmetric_huber",
            )

        np.testing.assert_allclose(loaded, [0.25, 1.5])

    def test_explicit_stage_cap_preserves_fixed_values_last(self):
        lower, upper = ensemble.make_policy_bounds(
            3,
            fixed_indices=[1],
            fixed_values=[1.2],
            upper_bounds=[1.5, 1.5, 0.7],
            policy_upper_bound=1.0,
        )

        np.testing.assert_allclose(lower, [0.0, 1.2, 0.0])
        np.testing.assert_allclose(upper, [1.0, 1.2, 0.7])

    def test_smooth_premium_skips_the_removal_active_set(self):
        mitigation = np.array([1.1, 0.4])
        diagnostics = successful_diagnostics(mitigation)
        with mock.patch.dict(
            os.environ,
            {
                "BACKSTOP_SMOOTHING_WIDTH": "0.002",
                "BACKSTOP_SMOOTHING_MODE": "symmetric_huber",
                "LBFGSB_POLICY_UPPER_BOUND": "1.5",
                "LBFGSB_REMOVAL_ACTIVE_SET": "1",
            },
            clear=False,
        ), mock.patch.object(
            ensemble,
            "run_lbfgsb_policy",
            return_value=(mitigation, 1.0, diagnostics),
        ) as smooth_solver, mock.patch.object(
            ensemble,
            "run_lbfgsb_policy_removal_active_set",
        ) as active_set_solver:
            _, _, result = ensemble.solve_lbfgsb_policy(
                QuadraticPolicyUtility([1.1, 0.4]),
                2,
                "smooth_test",
                gradient_mode="adjoint",
            )
            formulation = ensemble.cost_formulation_name()

        smooth_solver.assert_called_once()
        active_set_solver.assert_not_called()
        self.assertTrue(result["removal_active_set_requested"])
        self.assertFalse(result["removal_active_set_enabled"])
        self.assertEqual(
            formulation,
            "additive_removal_premium_symmetric_huber_v1",
        )
        self.assertTrue(result["removal_active_set_not_required"])
        self.assertEqual(
            result["removal_active_set_status"],
            "not_required_smooth_premium",
        )

    def test_probe_audit_identifies_only_robust_removal_node(self):
        utility = QuadraticPolicyUtility([1.5, 0.8])
        audit = ensemble._removal_probe_audit(
            utility=utility,
            mitigation=np.array([1.0, 1.0]),
            eligible_nodes=np.array([0, 1]),
            active_nodes=set(),
            final_upper=np.array([1.5, 1.5]),
            probe_steps=(1e-4, 1e-3, 1e-2),
            screen_steps=(1e-4, 1e-2),
            gain_tol=1e-10,
            kink_tol=1e-8,
            min_positive_scales=3,
        )

        self.assertTrue(audit["all_finite"])
        self.assertEqual([row["node"] for row in audit["candidates"]], [0])
        self.assertEqual(audit["candidates"][0]["positive_scales"], 3)
        self.assertEqual(audit["best_node"], 0)

    def test_probe_audit_covers_fine_scales_when_screen_is_negative(self):
        utility = QuadraticPolicyUtility([1.000001])
        audit = ensemble._removal_probe_audit(
            utility=utility,
            mitigation=np.array([1.0]),
            eligible_nodes=np.array([0]),
            active_nodes=set(),
            final_upper=np.array([1.5]),
            probe_steps=(1e-6, 1e-5, 1e-4),
            screen_steps=(1e-4,),
            gain_tol=1e-14,
            kink_tol=1e-8,
            min_positive_scales=3,
        )

        self.assertTrue(audit["full_probe_coverage_complete"])
        self.assertEqual(audit["probe_evals"], 3)
        self.assertGreater(audit["best_gain"], 1e-14)
        self.assertEqual(audit["candidates"], [])

    def test_active_set_rejects_inconsistent_probe_configuration(self):
        utility = QuadraticPolicyUtility([1.5])
        configurations = (
            {
                "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-2",
                "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
                "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
            },
            {
                "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
                "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-5,1e-2",
                "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
            },
            {"LBFGSB_REMOVAL_KINK_TOL": "-1e-8"},
            {"LBFGSB_REMOVAL_GAIN_TOL": "nan"},
            {"LBFGSB_REMOVAL_STAGE0_POLISH_RESTARTS": "0"},
        )
        for configuration in configurations:
            env = {"LBFGSB_POLICY_UPPER_BOUND": "1.5"}
            env.update(configuration)
            with self.subTest(configuration=configuration), mock.patch.dict(
                os.environ, env
            ):
                with self.assertRaises(ValueError):
                    ensemble.run_lbfgsb_policy_removal_active_set(
                        utility,
                        num_nodes=1,
                        scenario_name="optimal",
                        gradient_mode="adjoint",
                    )

    def test_active_set_rejects_nonrobust_positive_final_probe(self):
        utility = QuadraticPolicyUtility([1.5])
        stage0_m = np.array([1.0])
        stage0_u = float(utility.utility(stage0_m))
        stage0_diag = successful_diagnostics(stage0_m, "stage0")
        ambiguous_audit = {
            "base_utility": stage0_u,
            "candidates": [],
            "probe_evals": 3,
            "tested_nodes": 1,
            "all_finite": True,
            "full_probe_coverage_complete": True,
            "full_probe_scale_count": 3,
            "best_gain": 1e-8,
            "best_node": 0,
            "best_step": 1e-4,
        }
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
            "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
            "LBFGSB_REMOVAL_GAIN_TOL": "1e-10",
            "LBFGSB_REMOVAL_ESCALATE_AMBIGUOUS": "0",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble,
            "run_lbfgsb_policy",
            return_value=(stage0_m, stage0_u, stage0_diag),
        ), mock.patch.object(
            ensemble, "_removal_probe_audit", return_value=ambiguous_audit
        ):
            with self.assertRaisesRegex(RuntimeError, "inconclusive"):
                ensemble.run_lbfgsb_policy_removal_active_set(
                    utility,
                    num_nodes=1,
                    scenario_name="optimal",
                    gradient_mode="adjoint",
                )

    def test_active_set_escalates_a_one_scale_exact_branch_then_certifies(self):
        utility = QuadraticPolicyUtility([1.2])
        stage0_m = np.array([1.0])
        stage0_u = float(utility.utility(stage0_m))
        branch_m = np.array([1.2])
        branch_u = float(utility.utility(branch_m))
        stage0_diag = successful_diagnostics(stage0_m, "stage0")
        branch_diag = successful_diagnostics(branch_m, "ambiguous_branch")
        ambiguous_audit = {
            "base_utility": stage0_u,
            "candidates": [],
            "ambiguous_candidates": [{
                "node": 0, "max_gain": 0.01, "seed_step": 1e-2,
                "positive_scales": 1,
            }],
            "probe_evals": 3,
            "tested_nodes": 1,
            "all_finite": True,
            "full_probe_coverage_complete": True,
            "full_probe_scale_count": 3,
            "gain_exceedance_count": 1,
            "best_gain": 0.01,
            "best_node": 0,
            "best_step": 1e-2,
        }
        clean_audit = {
            "base_utility": branch_u,
            "candidates": [],
            "ambiguous_candidates": [],
            "probe_evals": 0,
            "tested_nodes": 0,
            "all_finite": True,
            "full_probe_coverage_complete": True,
            "full_probe_scale_count": 3,
            "gain_exceedance_count": 0,
            "best_gain": -1e-8,
            "best_node": -1,
            "best_step": float("nan"),
        }
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
            "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
            "LBFGSB_REMOVAL_GAIN_TOL": "1e-8",
            "LBFGSB_REMOVAL_ESCALATE_AMBIGUOUS": "1",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble, "run_lbfgsb_policy",
            side_effect=[(stage0_m, stage0_u, stage0_diag),
                         (branch_m, branch_u, branch_diag)],
        ), mock.patch.object(
            ensemble, "_removal_probe_audit",
            side_effect=[ambiguous_audit, clean_audit],
        ):
            mitigation, _, diagnostics = (
                ensemble.run_lbfgsb_policy_removal_active_set(
                    utility, num_nodes=1, scenario_name="optimal",
                    gradient_mode="adjoint",
                )
            )

        np.testing.assert_allclose(mitigation, branch_m)
        self.assertTrue(diagnostics["removal_active_set_pass"])
        self.assertEqual(
            diagnostics["removal_active_set_ambiguous_branch_activations"],
            "round1:0",
        )

    def test_active_set_accepts_subtolerance_nonrobust_final_probe(self):
        """A gain below the materiality threshold is not unresolved."""
        utility = QuadraticPolicyUtility([1.0])
        stage0_m = np.array([1.0])
        stage0_u = float(utility.utility(stage0_m))
        stage0_diag = successful_diagnostics(stage0_m, "stage0")
        sub_tolerance_audit = {
            "base_utility": stage0_u,
            "candidates": [],
            "probe_evals": 3,
            "tested_nodes": 1,
            "all_finite": True,
            "full_probe_coverage_complete": True,
            "full_probe_scale_count": 3,
            "gain_exceedance_count": 0,
            "best_gain": 9.7e-9,
            "best_node": 0,
            "best_step": 1e-2,
        }
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
            "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
            "LBFGSB_REMOVAL_GAIN_TOL": "1e-8",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble, "run_lbfgsb_policy",
            return_value=(stage0_m, stage0_u, stage0_diag),
        ), mock.patch.object(
            ensemble, "_removal_probe_audit", return_value=sub_tolerance_audit
        ):
            mitigation, _, diagnostics = (
                ensemble.run_lbfgsb_policy_removal_active_set(
                    utility, num_nodes=1, scenario_name="optimal",
                    gradient_mode="adjoint",
                )
            )

        np.testing.assert_allclose(mitigation, stage0_m)
        self.assertTrue(diagnostics["removal_active_set_pass"])
        self.assertEqual(diagnostics["removal_active_set_gain_tol"], 1e-8)

    def test_active_set_uses_full_removal_feasibility_fallback(self):
        utility = QuadraticPolicyUtility([1.5, 1.5])
        cap_one_m = np.array([1.0, 1.0])
        cap_one_diag = successful_diagnostics(cap_one_m, "cap_one")
        cap_one_diag.update({
            "lbfgsb_success": False,
            "success_diagnostics": False,
            "gradient_validation_status": "failed",
            "gradient_validation_message": (
                "no finite interior objective point found for gradient validation"
            ),
        })
        full_m = np.array([1.5, 1.5])
        full_u = float(utility.utility(full_m))
        full_diag = successful_diagnostics(full_m, "full_removal")
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_REMOVAL_MAX_ROUNDS": "4",
            "LBFGSB_REMOVAL_BATCH_SIZE": "2",
            "LBFGSB_REMOVAL_GAIN_TOL": "1e-8",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble,
            "run_lbfgsb_policy",
            side_effect=[
                RuntimeError(
                    "adjoint_lbfgsb gradient validation failed: "
                    "no finite interior objective point found for gradient validation"
                ),
                (full_m, full_u, full_diag),
            ],
        ) as solve:
            mitigation, _, diagnostics = (
                ensemble.run_lbfgsb_policy_removal_active_set(
                    utility, num_nodes=2, scenario_name="optimal",
                    gradient_mode="adjoint",
                )
            )

        self.assertEqual(solve.call_count, 2)
        self.assertEqual(solve.call_args_list[0].kwargs["policy_upper_bound"], 1.0)
        self.assertEqual(solve.call_args_list[1].kwargs["policy_upper_bound"], 1.5)
        np.testing.assert_allclose(
            solve.call_args_list[1].kwargs["mandatory_starts"][0], full_m
        )
        np.testing.assert_allclose(mitigation, full_m)
        self.assertTrue(diagnostics["removal_active_set_pass"])
        self.assertTrue(diagnostics["removal_active_set_cap_one_infeasible"])
        self.assertTrue(
            diagnostics["removal_active_set_full_removal_feasibility_fallback"]
        )
        self.assertEqual(
            diagnostics["removal_active_set_full_removal_feasibility_support_count"],
            2,
        )

    def test_active_set_uses_boundary_fallback_when_full_interior_is_empty(self):
        utility = QuadraticPolicyUtility([1.5])
        full_m = np.array([1.5])
        full_diag = successful_diagnostics(full_m, "boundary_removal")
        no_interior = RuntimeError(
            "adjoint_lbfgsb gradient validation failed: "
            "no finite interior objective point found for gradient validation"
        )
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_REMOVAL_MAX_ROUNDS": "4",
            "LBFGSB_REMOVAL_GAIN_TOL": "1e-8",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble, "run_lbfgsb_policy",
            side_effect=[no_interior, no_interior, (full_m, 0.0, full_diag)],
        ) as solve:
            _, _, diagnostics = ensemble.run_lbfgsb_policy_removal_active_set(
                utility, num_nodes=1, scenario_name="optimal",
                gradient_mode="adjoint",
            )

        self.assertEqual(solve.call_count, 3)
        options = solve.call_args_list[2].kwargs["optimizer_options"]
        self.assertFalse(options["validate_gradient"])
        self.assertEqual(options["start_boundary_epsilon"], 0.0)
        self.assertTrue(
            diagnostics["removal_active_set_full_removal_boundary_validation_override"]
        )
        self.assertTrue(diagnostics["removal_active_set_pass"])

    def test_stage0_nonretryable_failure_aborts_even_when_abort_disabled(self):
        utility = QuadraticPolicyUtility([0.5])
        stage0_m = np.array([0.5])
        stage0_u = float(utility.utility(stage0_m))
        stage0_diag = successful_diagnostics(stage0_m, "stage0")
        stage0_diag.update({
            "lbfgsb_success": False,
            "success_diagnostics": False,
        })
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_ABORT_ON_DIAGNOSTIC_FAILURE": "0",
            "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
            "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble,
            "run_lbfgsb_policy",
            return_value=(stage0_m, stage0_u, stage0_diag),
        ):
            with self.assertRaises(RuntimeError):
                ensemble.run_lbfgsb_policy_removal_active_set(
                    utility,
                    num_nodes=1,
                    scenario_name="optimal",
                    gradient_mode="adjoint",
                )

    def test_stage0_stationarity_failure_is_polished_in_same_run(self):
        utility = QuadraticPolicyUtility([0.5])
        initial_m = np.array([0.4])
        final_m = np.array([0.5])
        initial_u = float(utility.utility(initial_m))
        final_u = float(utility.utility(final_m))
        failed_diag = successful_diagnostics(initial_m, "initial")
        failed_diag.update({
            "lbfgsb_success": False,
            "success_diagnostics": False,
            "lbfgsb_best_result_accepted": True,
            "stationarity_failed": True,
            "dispersion_failed": False,
            "perturbation_failed": False,
            "gradient_validation_status": "passed",
        })
        final_diag = successful_diagnostics(final_m, "polish")
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_REMOVAL_STAGE0_POLISH_RESTARTS": "2",
            "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
            "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble,
            "run_lbfgsb_policy",
            side_effect=[
                (initial_m, initial_u, failed_diag),
                (final_m, final_u, final_diag),
            ],
        ) as solver:
            mitigation, _, diagnostics = (
                ensemble.run_lbfgsb_policy_removal_active_set(
                    utility,
                    num_nodes=1,
                    scenario_name="optimal",
                    gradient_mode="adjoint",
                )
            )

        self.assertEqual(solver.call_count, 2)
        self.assertNotIn("mandatory_starts", solver.call_args_list[0].kwargs)
        np.testing.assert_array_equal(
            solver.call_args_list[1].kwargs["mandatory_starts"][0], initial_m
        )
        self.assertEqual(
            solver.call_args_list[1].kwargs["policy_upper_bound"], 1.0
        )
        np.testing.assert_array_equal(mitigation, final_m)
        self.assertTrue(diagnostics["removal_active_set_pass"])
        self.assertEqual(
            diagnostics["removal_active_set_stage0_polish_attempts"], 1
        )
        self.assertEqual(diagnostics["removal_stage0_failed_attempts"], 1)
        self.assertTrue(diagnostics["removal_stage0_lbfgsb_success"])
        self.assertTrue(
            diagnostics["removal_stage0_all_mandatory_starts_selected"]
        )
        self.assertEqual(
            [
                row["removal_active_set_stage"]
                for row in diagnostics["_local_results"]
            ],
            ["cap1_stage", "cap1_stage_polish_1"],
        )

    def test_optimal_baseline_seed_does_not_depend_on_delay_year(self):
        parts = ensemble.optimal_baseline_seed_parts(
            "delay_frontier", "run0", "default"
        )
        self.assertEqual(
            parts,
            ("delay_frontier", "run0", "default", "optimal_baseline"),
        )

    def test_active_set_activates_a_batch_of_profitable_removal_nodes(self):
        utility = QuadraticPolicyUtility([1.5, 1.4, 0.5])
        stage0_m = np.array([1.0, 1.0, 0.5])
        final_m = np.array([1.5, 1.4, 0.5])
        stage0_u = float(utility.utility(stage0_m))
        final_u = float(utility.utility(final_m))
        stage0_diag = successful_diagnostics(stage0_m, "stage0")
        final_diag = successful_diagnostics(final_m, "batch")
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
            "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
            "LBFGSB_REMOVAL_MAX_ROUNDS": "1",
            "LBFGSB_REMOVAL_BATCH_SIZE": "2",
            "LBFGSB_REMOVAL_GAIN_TOL": "1e-10",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble,
            "run_lbfgsb_policy",
            side_effect=[
                (stage0_m, stage0_u, stage0_diag),
                (final_m, final_u, final_diag),
            ],
        ) as solver:
            mitigation, utility_value, diagnostics = (
                ensemble.run_lbfgsb_policy_removal_active_set(
                    utility,
                    num_nodes=3,
                    scenario_name="batch",
                    gradient_mode="adjoint",
                )
            )

        np.testing.assert_allclose(mitigation, final_m)
        self.assertEqual(utility_value, final_u)
        self.assertEqual(solver.call_count, 2)
        np.testing.assert_allclose(
            solver.call_args_list[1].kwargs["mandatory_starts"][0],
            [1.01, 1.01, 0.5],
        )
        np.testing.assert_allclose(
            solver.call_args_list[1].kwargs["upper_bounds"],
            [1.5, 1.5, 1.0],
        )
        self.assertTrue(diagnostics["removal_active_set_pass"])
        self.assertEqual(diagnostics["removal_active_set_strategy"], "batch")
        self.assertEqual(diagnostics["removal_active_set_batch_size"], 2)
        self.assertEqual(diagnostics["removal_active_set_activated_nodes"], "0,1")
        self.assertEqual(diagnostics["removal_active_set_nodes_by_round"], "0+1")
        self.assertEqual(diagnostics["removal_active_set_rounds"], 1)

    def test_active_set_prunes_a_batch_node_that_returns_to_the_kink(self):
        utility = QuadraticPolicyUtility([1.1, 1.4, 0.5])
        stage0_m = np.array([1.0, 1.0, 0.5])
        pruned_m = np.array([1.0, 1.4, 0.5])
        final_m = np.array([1.1, 1.4, 0.5])
        stage0_u = float(utility.utility(stage0_m))
        pruned_u = float(utility.utility(pruned_m))
        final_u = float(utility.utility(final_m))
        failed_at_kink = successful_diagnostics(pruned_m, "batch")
        failed_at_kink.update({
            "lbfgsb_success": False,
            "success_diagnostics": False,
            "lbfgsb_best_result_accepted": True,
            "stationarity_failed": True,
            "gradient_validation_status": "passed",
        })
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
            "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
            "LBFGSB_REMOVAL_MAX_ROUNDS": "2",
            "LBFGSB_REMOVAL_BATCH_SIZE": "2",
            "LBFGSB_REMOVAL_GAIN_TOL": "1e-10",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble,
            "run_lbfgsb_policy",
            side_effect=[
                (stage0_m, stage0_u, successful_diagnostics(stage0_m, "stage0")),
                (pruned_m, pruned_u, failed_at_kink),
                (pruned_m, pruned_u, successful_diagnostics(pruned_m, "pruned")),
                (final_m, final_u, successful_diagnostics(final_m, "reactivated")),
            ],
        ) as solver:
            mitigation, utility_value, diagnostics = (
                ensemble.run_lbfgsb_policy_removal_active_set(
                    utility,
                    num_nodes=3,
                    scenario_name="prune",
                    gradient_mode="adjoint",
                )
            )

        self.assertEqual(solver.call_count, 4)
        np.testing.assert_allclose(mitigation, final_m)
        self.assertEqual(utility_value, final_u)
        np.testing.assert_allclose(
            solver.call_args_list[1].kwargs["upper_bounds"], [1.5, 1.5, 1.0]
        )
        np.testing.assert_allclose(
            solver.call_args_list[2].kwargs["upper_bounds"], [1.0, 1.5, 1.0]
        )
        np.testing.assert_allclose(
            solver.call_args_list[3].kwargs["upper_bounds"], [1.5, 1.5, 1.0]
        )
        self.assertTrue(diagnostics["removal_active_set_pass"])
        self.assertEqual(diagnostics["removal_active_set_deactivated_count"], 1)
        self.assertEqual(diagnostics["removal_active_set_deactivated_nodes"], "0")
        self.assertEqual(
            diagnostics["removal_active_set_deactivation_events"],
            "round1:kink:0",
        )
        self.assertEqual(diagnostics["removal_active_set_nodes_by_round"], "1+0,0")

    def test_active_set_retries_stationarity_and_retains_incumbent(self):
        utility = QuadraticPolicyUtility([1.5, 0.5])
        stage0_m = np.array([1.0, 0.5])
        final_m = np.array([1.5, 0.5])
        stage0_u = float(utility.utility(stage0_m))
        final_u = float(utility.utility(final_m))
        stage0_diag = successful_diagnostics(stage0_m, "stage0")
        failed_diag = successful_diagnostics(final_m, "first_branch")
        failed_diag.update({
            "lbfgsb_success": False,
            "success_diagnostics": False,
            "lbfgsb_best_result_accepted": True,
            "stationarity_failed": True,
            "dispersion_failed": False,
            "perturbation_failed": False,
            "gradient_validation_status": "passed",
        })
        final_diag = successful_diagnostics(final_m, "polish")

        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
            "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
            "LBFGSB_REMOVAL_MAX_ROUNDS": "4",
            "LBFGSB_REMOVAL_POLISH_RESTARTS": "2",
            "LBFGSB_REMOVAL_GAIN_TOL": "1e-10",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble,
            "run_lbfgsb_policy",
            side_effect=[
                (stage0_m, stage0_u, stage0_diag),
                (final_m, final_u, failed_diag),
                (final_m, final_u, final_diag),
            ],
        ) as solver:
            mitigation, utility_value, diagnostics = (
                ensemble.run_lbfgsb_policy_removal_active_set(
                    utility,
                    num_nodes=2,
                    scenario_name="optimal",
                    gradient_mode="adjoint",
                )
            )

        stage0_call, first_call, retry_call = solver.call_args_list
        self.assertEqual(stage0_call.kwargs["policy_upper_bound"], 1.0)
        self.assertNotIn("mandatory_starts", stage0_call.kwargs)
        np.testing.assert_allclose(
            first_call.kwargs["mandatory_starts"][0], [1.01, 0.5]
        )
        np.testing.assert_allclose(
            first_call.kwargs["upper_bounds"], [1.5, 1.0]
        )
        self.assertEqual(first_call.kwargs["policy_upper_bound"], 1.5)
        self.assertEqual(
            first_call.kwargs["optimizer_options"]["n_candidates"], 1
        )
        self.assertEqual(
            first_call.kwargs["optimizer_options"]["n_local_starts"], 1
        )
        np.testing.assert_array_equal(
            retry_call.kwargs["mandatory_starts"][0], final_m
        )
        self.assertEqual(retry_call.kwargs["seed_parts"][-2:], (1, 1))
        np.testing.assert_array_equal(mitigation, final_m)
        self.assertEqual(utility_value, final_u)
        self.assertEqual(solver.call_count, 3)
        self.assertTrue(diagnostics["lbfgsb_success"])
        self.assertTrue(diagnostics["removal_active_set_pass"])
        self.assertEqual(diagnostics["removal_active_set_activated_nodes"], "0")
        self.assertEqual(diagnostics["removal_active_set_rounds"], 1)
        self.assertGreater(diagnostics["removal_active_set_utility_gain"], 0.0)
        self.assertTrue(diagnostics["removal_active_set_no_improving_inactive_nodes"])
        self.assertEqual(
            diagnostics["removal_active_set_failed_solver_attempts"], 1
        )
        self.assertFalse(
            diagnostics["removal_active_set_all_solver_attempts_success"]
        )
        self.assertTrue(
            diagnostics["removal_active_set_accepted_stages_success"]
        )
        stages = [
            row["removal_active_set_stage"]
            for row in diagnostics["_local_results"]
        ]
        self.assertEqual(
            stages,
            ["cap1_stage", "removal_round_1_polish_0", "removal_round_1_polish_1"],
        )

    def test_removal_branch_nonretryable_failure_aborts_when_abort_disabled(self):
        utility = QuadraticPolicyUtility([1.5, 0.5])
        stage0_m = np.array([1.0, 0.5])
        branch_m = np.array([1.5, 0.5])
        stage0_u = float(utility.utility(stage0_m))
        branch_u = float(utility.utility(branch_m))
        stage0_diag = successful_diagnostics(stage0_m, "stage0")
        failed_diag = successful_diagnostics(branch_m, "nonretryable_branch")
        failed_diag.update({
            "lbfgsb_success": False,
            "success_diagnostics": False,
            "lbfgsb_best_result_accepted": True,
            "stationarity_failed": True,
            "dispersion_failed": False,
            "perturbation_failed": False,
            "gradient_validation_status": "failed",
        })
        unexpected_success = successful_diagnostics(
            branch_m, "unexpected_retry"
        )
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_ABORT_ON_DIAGNOSTIC_FAILURE": "0",
            "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
            "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
            "LBFGSB_REMOVAL_POLISH_RESTARTS": "2",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble,
            "run_lbfgsb_policy",
            side_effect=[
                (stage0_m, stage0_u, stage0_diag),
                (branch_m, branch_u, failed_diag),
                (branch_m, branch_u, unexpected_success),
            ],
        ) as solver:
            with self.assertRaisesRegex(
                RuntimeError, "removal branch was not accepted"
            ):
                ensemble.run_lbfgsb_policy_removal_active_set(
                    utility,
                    num_nodes=2,
                    scenario_name="optimal",
                    gradient_mode="adjoint",
                )

        self.assertEqual(solver.call_count, 2)

    def test_fixed_kink_node_is_not_eligible_for_removal(self):
        utility = QuadraticPolicyUtility([1.5, 0.5])
        stage0_m = np.array([1.0, 0.5])
        stage0_u = float(utility.utility(stage0_m))
        stage0_diag = successful_diagnostics(stage0_m, "stage0")
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
            "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble,
            "run_lbfgsb_policy",
            return_value=(stage0_m, stage0_u, stage0_diag),
        ) as solver:
            mitigation, _, diagnostics = (
                ensemble.run_lbfgsb_policy_removal_active_set(
                    utility,
                    num_nodes=2,
                    scenario_name="delayed",
                    fixed_indices=[0],
                    fixed_values=[1.0],
                    gradient_mode="adjoint",
                )
            )

        np.testing.assert_array_equal(mitigation, stage0_m)
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(diagnostics["removal_active_set_eligible_nodes"], 1)
        self.assertEqual(diagnostics["removal_active_set_activated_count"], 0)


    def test_smooth_proposal_is_a_copy_not_a_change_to_the_sharp_utility(self):
        utility = ConfigurableQuadraticPolicyUtility([1.5, 0.5])

        proposal = ensemble._smoothed_removal_proposal_utility(
            utility, 0.02, "symmetric_huber"
        )

        self.assertIsNot(proposal, utility)
        self.assertIsNot(proposal.cost, utility.cost)
        self.assertEqual(utility.cost.backstop_smoothing_width, 0.0)
        self.assertEqual(utility.cost.backstop_smoothing_mode, "one_sided_huber")
        self.assertEqual(proposal.cost.backstop_smoothing_width, 0.02)
        self.assertEqual(proposal.cost.backstop_smoothing_mode, "symmetric_huber")

    def test_smooth_proposal_prioritizes_only_sharp_verified_candidates(self):
        utility = ConfigurableQuadraticPolicyUtility([1.5, 1.4, 0.5])
        stage0_m = np.array([1.0, 1.0, 0.5])
        proposal_m = np.array([1.0, 1.4, 0.5])
        first_exact_m = proposal_m.copy()
        final_m = np.array([1.5, 1.4, 0.5])
        stage0_diag = successful_diagnostics(stage0_m, "stage0")
        proposal_diag = successful_diagnostics(proposal_m, "smooth_proposal")
        first_diag = successful_diagnostics(first_exact_m, "first_exact")
        final_diag = successful_diagnostics(final_m, "final_exact")
        env = {
            "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            "LBFGSB_REMOVAL_PROBE_STEPS": "1e-4,1e-3,1e-2",
            "LBFGSB_REMOVAL_SCREEN_STEPS": "1e-4,1e-2",
            "LBFGSB_REMOVAL_MIN_POSITIVE_SCALES": "3",
            "LBFGSB_REMOVAL_MAX_ROUNDS": "3",
            "LBFGSB_REMOVAL_BATCH_SIZE": "1",
            "LBFGSB_REMOVAL_PROPOSAL_SMOOTHING_WIDTH": "0.02",
            "LBFGSB_REMOVAL_PROPOSAL_SMOOTHING_MODE": "symmetric_huber",
            "LBFGSB_REMOVAL_PROPOSAL_MAXITER": "37",
        }
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ensemble,
            "run_lbfgsb_policy",
            side_effect=[
                (stage0_m, utility.utility(stage0_m), stage0_diag),
                (proposal_m, utility.utility(proposal_m), proposal_diag),
                (first_exact_m, utility.utility(first_exact_m), first_diag),
                (final_m, utility.utility(final_m), final_diag),
            ],
        ) as solver:
            _, _, diagnostics = ensemble.run_lbfgsb_policy_removal_active_set(
                utility, num_nodes=3, scenario_name="optimal",
                gradient_mode="adjoint",
            )

        self.assertEqual(solver.call_count, 4)
        self.assertTrue(diagnostics["removal_active_set_pass"])
        self.assertTrue(diagnostics["removal_active_set_smooth_proposal_ran"])
        self.assertTrue(diagnostics["removal_active_set_smooth_proposal_finite"])
        self.assertEqual(
            diagnostics["removal_active_set_smooth_proposal_support_nodes"], "1"
        )
        self.assertEqual(diagnostics["removal_active_set_nodes_by_round"], "1,0")
        self.assertEqual(
            solver.call_args_list[1].kwargs["optimizer_options"]["maxiter"], 37
        )
        np.testing.assert_allclose(
            solver.call_args_list[2].kwargs["upper_bounds"], [1.0, 1.5, 1.0]
        )

    def test_strict_replay_accepts_complete_certificate(self):
        rows = self._valid_replay_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(directory, rows)
            mitigation = self._load_strict(path)
        np.testing.assert_array_equal(mitigation, [0.25, 1.5])

    def test_strict_replay_rejects_ambiguous_or_failed_artifact(self):
        base_rows = self._valid_replay_rows()
        cases = []
        duplicate = copy.deepcopy(base_rows)
        duplicate.append(copy.deepcopy(duplicate[0]))
        cases.append(("duplicate", duplicate))

        failed = copy.deepcopy(base_rows)
        diagnostics = json.loads(failed[0]["solver_diagnostics_json"])
        diagnostics["optimal_removal_active_set_pass"] = False
        for row in failed:
            row["solver_diagnostics_json"] = json.dumps(diagnostics)
        cases.append(("failed", failed))

        positive = copy.deepcopy(base_rows)
        diagnostics = json.loads(positive[0]["solver_diagnostics_json"])
        diagnostics["optimal_removal_active_set_final_max_inactive_gain"] = 1e-8
        for row in positive:
            row["solver_diagnostics_json"] = json.dumps(diagnostics)
        cases.append(("positive_gain", positive))

        nonfinite = copy.deepcopy(base_rows)
        nonfinite[0]["mitigation"] = "nan"
        cases.append(("nonfinite", nonfinite))

        wrong_cap = copy.deepcopy(base_rows)
        for row in wrong_cap:
            row["lbfgsb_policy_upper_bound"] = "1.0"
        cases.append(("wrong_cap", wrong_cap))

        for label, rows in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = self._write_rows(directory, rows)
                with self.assertRaises(ValueError):
                    self._load_strict(path)

    def test_replay_requires_path_but_warm_start_allows_identical_duplicates(self):
        with mock.patch.dict(os.environ, {
            "EXTERNAL_OPTIMAL_WARM_START_NODE_PRICES": "",
            "REPLAY_EXTERNAL_OPTIMAL_BASELINE": "1",
        }, clear=False):
            with self.assertRaises(ValueError):
                ensemble.load_external_optimal_warm_start(2, "0|5", 5)

        rows = self._valid_replay_rows()
        rows.extend(copy.deepcopy(rows))
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_rows(directory, rows)
            with mock.patch.dict(os.environ, {
                "EXTERNAL_OPTIMAL_WARM_START_NODE_PRICES": path,
                "EXTERNAL_OPTIMAL_WARM_START_TASK_ID": "",
                "REPLAY_EXTERNAL_OPTIMAL_BASELINE": "0",
                "LBFGSB_POLICY_UPPER_BOUND": "1.5",
            }, clear=False):
                mitigation = ensemble.load_external_optimal_warm_start(
                    2, "0|5", 5, sample_id="run0", tree_spec="default"
                )
        np.testing.assert_array_equal(mitigation, [0.25, 1.5])

    def test_csv_append_preserves_existing_columns_when_diagnostics_vary(self):
        with tempfile.TemporaryDirectory() as directory:
            results_path = str(Path(directory) / "results.csv")
            self.assertTrue(ensemble.append_results_to_csv(
                {"task_id": "1", "stable": "first"}, results_path
            ))
            self.assertTrue(ensemble.append_results_to_csv(
                {"task_id": "2", "branch_only": "second"}, results_path
            ))
            with open(results_path, newline="") as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(handle.seek(0) or next(csv.reader(handle)), [
                    "task_id", "stable", "branch_only"
                ])
            self.assertEqual(rows, [
                {"task_id": "1", "stable": "first", "branch_only": ""},
                {"task_id": "2", "stable": "", "branch_only": "second"},
            ])

            rows_path = str(Path(directory) / "rows.csv")
            self.assertTrue(ensemble.append_rows_to_csv(
                [{"node": "0", "price": "10"}], rows_path
            ))
            self.assertTrue(ensemble.append_rows_to_csv(
                [{"node": "1", "diagnostic": "kink"}], rows_path
            ))
            with open(rows_path, newline="") as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(handle.seek(0) or next(csv.reader(handle)), [
                    "node", "price", "diagnostic"
                ])
            self.assertEqual(rows, [
                {"node": "0", "price": "10", "diagnostic": ""},
                {"node": "1", "price": "", "diagnostic": "kink"},
            ])


    def test_eis_truncated_gaussian_support_has_distinct_identity(self):
        with mock.patch.dict(
            os.environ, {"GAUSSIAN_EIS_UPPER_BOUND": "0.833"}, clear=False
        ):
            self.assertEqual(ensemble.gaussian_eis_upper_bound(), 0.833)
            self.assertEqual(ensemble.gaussian_support_tag(), "EISmax0p833")
            self.assertIn("EISmax0p833", ensemble.get_sample_filename())
            metadata = ensemble.gaussian_support_metadata()
            self.assertTrue(metadata["gaussian_eis_truncated"])
            self.assertEqual(metadata["gaussian_eis_upper_bound"], 0.833)

    def test_truncated_gaussian_sample_file_is_generated_once_and_reused(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"GAUSSIAN_EIS_UPPER_BOUND": "0.833"}, clear=False
        ), mock.patch.object(ensemble, "DATA_DIR", directory), mock.patch.object(
            ensemble, "N_SAMPLES", 3
        ):
            first = ensemble.load_or_generate_gaussian_samples()
            second = ensemble.load_or_generate_gaussian_samples()
            self.assertEqual(first.shape, (3, ensemble.DIMS))
            np.testing.assert_allclose(first, second)
            self.assertTrue(os.path.isfile(ensemble.get_sample_filename()))
            self.assertTrue(np.all(first[:, ensemble.param_names.index("EIS")] <= 0.833))

    def test_gaussian_generation_allows_mode_at_eis_truncation_boundary(self):
        samples = generate_gaussian_samples(
            32, 1, [0.55], [0.833], means=[0.833], stds=[0.1],
            save_file=False, random_seed=7,
        )
        self.assertEqual(samples.shape, (32, 1))
        self.assertTrue(np.all(samples >= 0.55))
        self.assertTrue(np.all(samples <= 0.833))


if __name__ == "__main__":
    unittest.main()

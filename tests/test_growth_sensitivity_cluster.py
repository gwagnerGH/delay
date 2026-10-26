"""Preflight and dispatch checks without launching cluster solves."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import main_growth_sensitivity_cluster as suite
from src.config.revised_baseline import BASELINE
from src.config.terminal_growth import TerminalGrowthSettings, NonFiniteTerminalError


class GrowthSensitivityClusterTests(unittest.TestCase):
    def test_every_case_has_finite_correctly_compounded_continuation(self):
        rows = suite.validate_grid()
        self.assertEqual(len(rows), 13)
        self.assertEqual(len({r['label'] for r in rows}), 13)
        rho = 1-1/BASELINE.eis
        beta = (1-BASELINE.prtp)**5
        for row in rows:
            annual_log = (row['log_mean'] + (1-BASELINE.ra)*row['log_sigma']**2/2
                          if row['mode'] == 'lognormal' else np.log1p(row['growth']))
            ratio = beta*np.exp(5*rho*annual_log)
            self.assertLess(ratio, 1)
            self.assertTrue(np.isfinite(row['multiplier']))
            self.assertGreater(row['multiplier'], 0)
            independent_k = ((1-beta)/(1-ratio))**(1/rho)
            np.testing.assert_allclose(row['multiplier'], independent_k, rtol=2e-12)

    def test_exact_baseline_and_effective_growth_control(self):
        rows = {r['label']: r for r in suite.validate_grid()}
        baseline = rows['log_mu0p015_sigma0p02']
        control = rows['det_baseline_effective_growth']
        self.assertAlmostEqual(baseline['multiplier'], 530.23532662464, places=7)
        np.testing.assert_allclose(control['multiplier'], baseline['multiplier'], rtol=1e-12)
        self.assertAlmostEqual(control['growth'], np.expm1(.015-(10-1)*.02**2/2))
        self.assertNotAlmostEqual(control['expected_annual_net_growth'], baseline['expected_annual_net_growth'])
        self.assertEqual(control['matched_control_for'], baseline['label'])

    def test_nonfinite_case_is_rejected_before_dispatch(self):
        bad = suite.GrowthCase('infinite', TerminalGrowthSettings(mode='deterministic', growth=.02))
        with patch.object(suite, 'CASES', (bad,)), \
             patch('scripts.run_revised_cluster.run_cluster_pair') as run:
            with self.assertRaisesRegex(NonFiniteTerminalError, 'Growth case infinite'):
                suite.execute_task(1)
            run.assert_not_called()

    def test_task_mapping_and_single_delay_cover_each_pair_once(self):
        pairs = [suite.task_configuration(task) for task in range(1, 40)]
        self.assertEqual(len({(case.label, delay) for case, delay in pairs}), 39)
        self.assertEqual(suite.task_configuration(1)[1], 5)
        self.assertEqual(suite.task_configuration(39)[1], 15)
        for task in range(1, 14):
            self.assertEqual(suite.task_configuration(task, (10,)), (suite.CASES[task-1], 10))
        for task in (0, 40):
            with self.assertRaises(ValueError):
                suite.task_configuration(task)
        for spec in ('', '10,10', '0', '7', '10,'):
            with self.assertRaises(ValueError):
                suite.parse_delay_years(spec)

    def test_dispatch_pins_baseline_and_records_case_despite_stale_environment(self):
        import scripts.run_revised_cluster as cluster
        calls = []
        def run(**kwargs):
            calls.append((kwargs, TerminalGrowthSettings.from_environment(), dict(os.environ)))
            return dict(optimal_certified=True, delayed_certified=True)
        with patch.dict(os.environ, {'REVISED_TERMINAL_GROWTH_MODE': 'legacy',
                 'REVISED_TERMINAL_LOG_SIGMA': '.9', 'REVISED_BACKSTOP': 'cubic',
                 'LBFGSB_POLICY_UPPER_BOUND': '1', 'SGE_TASK_ID': '999'}), \
             patch.object(cluster, 'run_cluster_pair', side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()):
            suite.execute_task(1, output_folder='test_growth')
            suite.execute_task(32, output_folder='test_growth')  # baseline lognormal, 10-year delay
        self.assertEqual(len(calls), 2)
        for kwargs, settings, env in calls:
            np.testing.assert_array_equal(kwargs['param_vals'][0, :7], suite.RUN0_PARAMETER_VALUES)
            self.assertEqual(kwargs['tree_spec'], 'full')
            self.assertEqual(env['REVISED_BACKSTOP'], 's_curve')
            self.assertEqual(env['LBFGSB_POLICY_UPPER_BOUND'], '4.0')
            self.assertEqual(kwargs['delay_window_years'], kwargs['delay_year'])
            self.assertEqual(len(kwargs['output_metadata']['growth_grid_preflight']), 13)
            self.assertEqual(int(env['SGE_TASK_ID']), kwargs['output_metadata']['growth_grid_task_id'])
        self.assertEqual(calls[0][1].mode, 'deterministic')
        self.assertAlmostEqual(calls[0][1].growth, .005)
        self.assertEqual(calls[1][1].mode, 'lognormal')
        self.assertAlmostEqual(calls[1][1].log_mean, .015)
        self.assertAlmostEqual(calls[1][1].log_sigma, .02)

    def test_dry_run_does_not_solve_and_json_has_unambiguous_growth_units(self):
        stream = io.StringIO()
        with patch('scripts.run_revised_cluster.run_cluster_pair') as run, \
             contextlib.redirect_stdout(stream):
            suite.main(['--list-cases', '--json', '--delay-years', '10'])
            run.assert_not_called()
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload['array_tasks'], 13)
        self.assertEqual(payload['preterminal_growth'], .015)
        self.assertEqual(payload['delays'], [10])
        self.assertEqual(len(payload['cases']), 13)

    def test_shell_handles_empty_args_and_forwards_environment_and_options(self):
        root = Path(suite.__file__).resolve().parent
        runner = root/'run_growth_sensitivity_array_job.sh'
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp)/'python'
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "${DELAY_YEARS:-unset}" "$@"\n')
            stub.chmod(0o755)
            env = dict(os.environ, EZDELAY_SKIP_CONDA='1', PATH=tmp+':/usr/bin:/bin')
            env.pop('DELAY_YEARS', None)
            for arguments, years, forwarded in (
                ([], 'unset', []),
                (['DELAY_YEARS=10'], '10', []),
                (['DELAY_YEARS=10', '--list-cases', '--json'], '10', ['--list-cases', '--json'])):
                result = subprocess.run(['bash', str(runner), *arguments],
                    cwd=tmp, env=env, text=True, capture_output=True, check=True)
                self.assertEqual(result.stdout.splitlines(),
                    [str(root.parent), years, '-u', '-m', 'scripts.main_growth_sensitivity_cluster', *forwarded])

    def test_uncertified_outputs_cause_task_failure(self):
        with patch.dict(os.environ, {}), \
             patch('scripts.run_revised_cluster.run_cluster_pair', return_value=dict(
                 optimal_certified=True, delayed_certified=False)), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'failed local stationarity'):
                suite.execute_task(1)


if __name__ == '__main__':
    unittest.main()

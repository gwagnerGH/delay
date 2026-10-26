"""Independent checks of timing, stochastic EZ tails, and cluster forwarding."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.polynomial.hermite import hermgauss

from src.config.revised_baseline import BASELINE
from src.config.terminal_growth import (TerminalGrowthSettings,
    NonFiniteTerminalError, add_terminal_arguments, terminal_settings_from_args)
from src.utility import EZUtility
from src.tree import TreeModel


class TerminalGrowthTests(unittest.TestCase):
    def test_baseline_against_independent_lognormal_moment_and_fixed_point(self):
        settings = TerminalGrowthSettings()
        info = settings.describe(ra=10, eis=1.5, prtp=.005)
        self.assertAlmostEqual(info['effective_annual_net_growth'], .0132875045963293)
        self.assertAlmostEqual(info['multiplier'], 530.23532662464, places=7)
        x, weights = hermgauss(60)
        alpha, rho, beta = 1-10, 1-1/1.5, (1-.005)**5
        # Direct quadrature of the FIVE-YEAR compounded annual shock.
        gross = np.exp(5*.015 + np.sqrt(5)*.02*np.sqrt(2)*x)
        moment = np.sum(weights/np.sqrt(np.pi) * gross**alpha)
        ce = moment**(1/alpha)
        self.assertAlmostEqual(ce, info['growth_factor_per_utility_step'], places=13)
        k = info['multiplier']
        self.assertAlmostEqual(k, ((1-beta)+beta*(k*ce)**rho)**(1/rho), places=10)
        # Finite-horizon recursion converges to the same K, without using the
        # analytical multiplier inside the iteration.
        v = 1.0
        for _ in range(16000):
            v = (1-beta) + beta*ce**rho*v
        np.testing.assert_allclose(v**(1/rho), k, rtol=2e-12)

    def test_zero_volatility_matches_deterministic_compounded_growth(self):
        stochastic = TerminalGrowthSettings(log_sigma=0)
        deterministic = TerminalGrowthSettings(mode='deterministic', growth=np.expm1(.015))
        for step in (1, 5, 10):
            a = stochastic.describe(ra=10, eis=1.5, prtp=.005, period_len=step)
            b = deterministic.describe(ra=10, eis=1.5, prtp=.005, period_len=step)
            np.testing.assert_allclose(a['multiplier'], b['multiplier'], rtol=1e-12)
        self.assertAlmostEqual(deterministic.log_growth_per_step(5, 10), .075)

    def test_unit_eis_and_unit_ra_limits(self):
        settings = TerminalGrowthSettings(log_mean=.005)
        self.assertAlmostEqual(float(settings.log_growth_per_step(5, 1)), .025)
        center = settings.describe(ra=1, eis=1, prtp=.005)['multiplier']
        for eis in (.999999, 1.000001):
            np.testing.assert_allclose(settings.describe(ra=1, eis=eis, prtp=.005)['multiplier'],
                                       center, rtol=3e-6)

    def test_infinite_tail_is_not_silently_clipped(self):
        settings = TerminalGrowthSettings()
        self.assertFalse(settings.admissible_mask(10, 2.2, .005))
        with self.assertRaises(NonFiniteTerminalError):
            settings.describe(ra=10, eis=2.2, prtp=.005)
        self.assertTrue(settings.admissible_mask(10, 2.2, .024))

    def test_deterministic_and_legacy_modes_are_separate(self):
        deterministic = TerminalGrowthSettings(mode='deterministic', growth=.01)
        self.assertAlmostEqual(deterministic.describe(ra=10, eis=1.5, prtp=.005)['multiplier'],
                               25.19439222893, places=8)
        legacy = TerminalGrowthSettings(mode='legacy')
        self.assertAlmostEqual(float(legacy.log_growth_per_step(5, 10, .015)), np.log1p(.015))
        self.assertAlmostEqual(float(deterministic.log_growth_per_step(5, 10)), 5*np.log1p(.01))

    def test_constructor_keeps_preterminal_growth_separate(self):
        u = EZUtility(TreeModel([0, 5, 10]), None, None, 5,
                      terminal_growth_mode='deterministic', terminal_growth=.01)
        np.testing.assert_allclose(u.potential_cons, 1.015**np.array([0, 5, 10]))
        self.assertAlmostEqual(u.terminal_growth_term, 1.01**5)
        self.assertAlmostEqual(u._terminal_multiplier(), 25.19439222893, places=8)

    def test_cli_and_cluster_environment_use_the_same_settings(self):
        env = {'REVISED_TERMINAL_GROWTH_MODE': 'deterministic',
               'REVISED_TERMINAL_GROWTH': '.01', 'REVISED_TERMINAL_LOG_MEAN': '.012',
               'REVISED_TERMINAL_LOG_SIGMA': '.03'}
        with patch.dict('os.environ', env):
            parser = argparse.ArgumentParser()
            add_terminal_arguments(parser)
            args = parser.parse_args([])
            self.assertEqual(terminal_settings_from_args(args), TerminalGrowthSettings.from_environment())
            explicit = parser.parse_args(['--terminal-growth-mode', 'lognormal', '--terminal-log-sigma', '.02'])
            self.assertEqual(terminal_settings_from_args(explicit).mode, 'lognormal')
        self.assertNotEqual(TerminalGrowthSettings().cache_tag(),
                           TerminalGrowthSettings(log_sigma=.03).cache_tag())

    def test_cluster_marks_inadmissible_grid_case_without_solving(self):
        import scripts.run_revised_cluster as cluster
        values = np.array([[10, 2.2, 1.5, 0, .005, 10000, .015]])
        with tempfile.TemporaryDirectory() as tmp, patch.object(cluster, 'ROOT', Path(tmp)), \
             patch.object(cluster, 'solve') as solver, \
             patch.dict('os.environ', {'REVISED_TERMINAL_GROWTH_MODE': 'lognormal',
                 'REVISED_TERMINAL_LOG_MEAN': '.015', 'REVISED_TERMINAL_LOG_SIGMA': '.02'}):
            result = cluster.run_cluster_pair(sample_index=0, delay_year=10,
                param_vals=values, out_folder='test', baseline=2, run_type='preference_grid')
            self.assertEqual(result['status'], 'terminal_inadmissible')
            solver.assert_not_called()
            records = list(Path(tmp).rglob('terminal_inadmissible.json'))
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0].read_text())['eis'], 2.2)
            self.assertFalse(list(Path(tmp).rglob('result.json')))

    def test_stochastic_tail_adjoint_with_and_without_learning(self):
        from test_revised_learning_baseline import objective
        policy = np.array([.6, 1.3, .75])
        for learning in (0, 1.5):
            model = objective(ra=10, eis=1.5, tech_scale=learning)
            self.assertEqual(model.zero.terminal_settings.mode, 'lognormal')
            self.assertAlmostEqual(model.zero._terminal_multiplier(), 530.23532662464, places=7)
            value, gradient = model.value_and_gradient(policy)
            self.assertGreater(model.diagnostics()['minimum_consumption'], 0)
            for node in range(3):
                step = np.zeros(3)
                step[node] = 2e-5
                finite = (model.utility(policy+step)-model.utility(policy-step))/(4e-5)
                np.testing.assert_allclose(gradient[node], finite, rtol=3e-5, atol=2e-7)
            self.assertAlmostEqual(value, model.utility(policy), places=10)

    def test_cluster_pair_forwards_tail_to_both_solves_and_compensation(self):
        import scripts.run_revised_cluster as cluster
        calls = []
        settings = TerminalGrowthSettings()

        def solve(args):
            calls.append(args)
            folder = Path(args[args.index('--output')+1])
            folder.mkdir(parents=True)
            value = 1. if folder.name == 'optimal' else .9
            metadata = settings.describe(ra=10, eis=1.5, prtp=.005, modeled_growth=.015)
            (folder/'manifest.json').write_text(json.dumps(dict(
                final_utility=value, local_stationarity_certified=True,
                terminal_continuation=metadata)))
            np.savez(folder/'policy.npz', mitigation=np.array([.6, .7, .8]))
            rows = {
                'interval_nodes.csv': dict(node=0, year=2025, probability=1,
                    mitigation=.6, interval_mac=188.6),
                'decision_state_nodes.csv': dict(node=0, year=2025, ghg_ppm=425)}
            for name, row in rows.items():
                with (folder/name).open('w') as handle:
                    writer = csv.DictWriter(handle, fieldnames=row)
                    writer.writeheader()
                    writer.writerow(row)

        values = np.array([[10, 1.5, 1.5, 0, .005, 10000, .015]])
        with tempfile.TemporaryDirectory() as tmp, patch.object(cluster, 'ROOT', Path(tmp)), \
             patch.object(cluster, 'TREES', {'full': [0, 5, 10]}), \
             patch.object(cluster, 'solve', side_effect=solve), \
             patch.object(cluster, 'build_saved_revised_model') as builder, \
             patch.dict('os.environ', {'REVISED_TERMINAL_GROWTH_MODE': 'lognormal',
                 'REVISED_TERMINAL_GROWTH': '.015',
                 'REVISED_TERMINAL_LOG_MEAN': '.015', 'REVISED_TERMINAL_LOG_SIGMA': '.02'}):
            builder.return_value.delay_window_compensated_utility.side_effect = lambda m, years, f: .9*f
            result = cluster.run_cluster_pair(sample_index=0, delay_year=5,
                param_vals=values, out_folder='test', baseline=2, run_type='preference_grid')
            self.assertAlmostEqual(result['delay_window_ce_fraction'], 1/9)
            self.assertAlmostEqual(result['delay_window_ce_pct_per_year'], 100/9)
            self.assertEqual(len(calls), 2)
            for args in calls:
                self.assertEqual(args[args.index('--terminal-growth-mode')+1], 'lognormal')
                self.assertEqual(float(args[args.index('--terminal-log-sigma')+1]), .02)
            self.assertEqual(builder.call_args.kwargs['terminal_growth_mode'], 'lognormal')
            self.assertEqual(builder.call_args.kwargs['terminal_log_sigma'], .02)
            self.assertAlmostEqual(result['terminal_continuation']['multiplier'], 530.23532662464, places=7)

    def test_saved_grid_builder_inherits_baseline_and_accepts_deterministic_tail(self):
        from src.revised_experiment import build_saved_revised_model
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'damages.npz'
            knots = np.arange(4., -.001, -.2)
            simulated = np.broadcast_to(
                (np.arange(len(knots))**1.4*.0004)[:, None, None], (len(knots), 2, 2))
            np.savez(source, knots=knots, damages=simulated, decision_times=[0, 5, 10])
            model = build_saved_revised_model(source, decision_times=[0, 5, 10])
            self.assertAlmostEqual(model.zero._terminal_multiplier(), 530.23532662464, places=7)
            deterministic = build_saved_revised_model(source, decision_times=[0, 5, 10],
                terminal_growth_mode='deterministic', terminal_growth=.01)
            self.assertAlmostEqual(deterministic.zero._terminal_multiplier(), 25.19439222893, places=8)

    def test_robustness_grid_preparation_is_cached_and_ignores_terminal_settings(self):
        import scripts.revised_robustness_inputs as inputs
        with tempfile.TemporaryDirectory() as tmp, patch.object(inputs, 'ROOT', Path(tmp)), \
             patch.object(inputs, 'DamageSimulation') as simulation, \
             patch.dict('os.environ', {'REQUIRE_DAMAGE_IMPORT': '0'}):
            source = Path(tmp)/'data/revised_default/full_damages.npz'
            source.parent.mkdir(parents=True)
            np.savez(source, knots=[1, .5, 0], draws=10, seed=10)
            simulation.return_value._run_path.return_value = np.zeros((2, 2))
            options = ('damage', 'test', [0, 5, 10], 1.,
                       dict(dam_func=1, tip_on=0, d_unc=1, t_unc=1))
            first = inputs.ensure_grid(*options)
            with patch.dict('os.environ', {'REVISED_TERMINAL_LOG_SIGMA': '.04'}):
                second = inputs.ensure_grid(*options)
            self.assertEqual(first, second)
            self.assertEqual(simulation.call_count, 1)
            with np.load(first[0]) as saved:
                self.assertEqual(saved['damages'].shape, (3, 2, 2))
                np.testing.assert_array_equal(saved['decision_times'], [0, 5, 10])


if __name__ == '__main__':
    unittest.main()

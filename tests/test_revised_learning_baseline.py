"""Finite-difference checks for the revised learning and backstop baseline."""
import unittest
import tempfile
import csv
import os
from unittest import mock
from pathlib import Path

import numpy as np

from src.climate import BPWClimate
from src.revised_experiment import (CubicBackstopCost, ExponentialCubicBackstopCost,
    FiveYearFlowObjective,
    FlatAfter2100Emissions, LearningFiveYearFlowObjective, PchipDamage,
    SmoothBackstopCost)
from src.tree import TreeModel


def objective(ra=15.0, eis=1.5, tech_scale=1.5, backstop='s_curve', gamma=0.0,
              decision_times=None, learning_reference_year=None, cost_floor=0.0):
    tree = TreeModel([0, 5, 10] if decision_times is None else decision_times)
    emissions = FlatAfter2100Emissions(tree, 2, emissions_time_step=5)
    emissions.baseline_emission_setup()
    climate = BPWClimate(tree, emissions, draws=10)
    knots = np.round(np.arange(4.0, -0.0001, -0.2), 8)
    damage = PchipDamage(tree, emissions, climate, knots, draws=10)
    simulated = np.arange(len(knots), dtype=float)**1.4 * 0.0004
    damage.d_rcomb = np.broadcast_to(simulated[:, None, None],
        (len(knots), tree.num_final_states, tree.num_periods)).copy()
    emit0 = float(np.interp(2030, emissions.times, emissions.baseline_gtco2))
    if backstop == 's_curve':
        cost = SmoothBackstopCost(tree, emit0, 2, 1.5, tech_scale, 86252.0, 0.0, True)
    elif backstop == 'cubic':
        cost = CubicBackstopCost(tree, emit0, 2, 1.5, tech_scale, 86252.0, 0.0,
                                 True, gamma=gamma)
    elif backstop == 'exp_cubic':
        cost = ExponentialCubicBackstopCost(
            tree, emit0, 2, 1.5, tech_scale, 86252.0, 0.0,
            True, gamma=gamma)
    else:
        raise ValueError(backstop)
    cls = LearningFiveYearFlowObjective if tech_scale else FiveYearFlowObjective
    options = ({'learning_reference_year': learning_reference_year}
               if tech_scale else {})
    return cls(tree, damage, cost, emissions, eis=eis, ra=ra,
               cost_floor=cost_floor, **options)


class LearningBaselineTests(unittest.TestCase):
    def test_cost_floor_preserves_2030_and_adjoint_derivatives(self):
        for learning in (0.0, 1.5):
            with self.subTest(learning=learning):
                dates = [0, 10, 35]
                reference = objective(tech_scale=learning, decision_times=dates)
                floored = objective(tech_scale=learning, cost_floor=0.4,
                                    decision_times=dates)
                mitigation = np.array([0.6, 1.2, 0.8])
                year_technology = (floored.technology_by_year(mitigation)
                    if learning else None)
                for elapsed in (0, 5):
                    year = 2025+elapsed
                    if learning:
                        self.assertAlmostEqual(
                            year_technology[elapsed][0],
                            reference.technology_by_year(mitigation)[elapsed][0])
                    else:
                        self.assertAlmostEqual(
                            floored.technology_multiplier(year),
                            reference.technology_multiplier(year))
                self.assertGreater(reference.utility(mitigation),
                                   floored.utility(mitigation))
                _, gradient = floored.value_and_gradient(mitigation)
                for node in (0, 1, 2):
                    h = 1e-5
                    up, down = mitigation.copy(), mitigation.copy()
                    up[node] += h
                    down[node] -= h
                    finite = (floored.utility(up)-floored.utility(down))/(2*h)
                    self.assertAlmostEqual(gradient[node], finite, delta=3e-7)

    def test_requested_shared_baseline_and_bound_inequality(self):
        from src.config.revised_baseline import BASELINE
        from scripts.run_revised_adjoint import feasible_foc_gap
        self.assertEqual((BASELINE.ra, BASELINE.eis, BASELINE.prtp),
                         (10.0, 1.5, 0.005))
        self.assertEqual((BASELINE.growth, BASELINE.tech_const,
                          BASELINE.tech_scale), (0.015, 1.5, 0.0))
        self.assertEqual(BASELINE.backstop, 's_curve')
        from src.config.revised_baseline import PREFERENCE_RA_LEVELS, PREFERENCE_EIS_LEVELS
        self.assertLess(min(PREFERENCE_RA_LEVELS), BASELINE.ra)
        self.assertGreater(max(PREFERENCE_RA_LEVELS), BASELINE.ra)
        self.assertLess(min(PREFERENCE_EIS_LEVELS), BASELINE.eis)
        self.assertGreater(max(PREFERENCE_EIS_LEVELS), BASELINE.eis)
        from scripts import main_preference_grid_cluster as preference_grid
        _, labels, _ = preference_grid.preference_grid_values()
        self.assertIn('RA45_EIS3_PRTP0.005', labels)
        self.assertAlmostEqual(feasible_foc_gap([-1.0, 2.0, 9.0],
            [0.0, 1.0, 0.0], [4.0, 1.0, 0.0], fixed=[2]), 0.0)
        self.assertAlmostEqual(feasible_foc_gap([1.0, -2.0, 9.0],
            [0.0, 1.0, 0.0], [4.0, 1.0, 0.0], fixed=[2]), 2.0)

    def test_preference_grid_and_frontier_cleanup(self):
        from scripts import main_preference_grid_cluster as preference
        from scripts import main_delay_frontier_cluster as frontier
        from scripts.reproduce_main_spec import CLUSTER_RUNS
        from src.config.parameter_priors import PARAMETER_PRIOR_INDEX as index, RUN0_PARAMETER_VALUES
        from src.config.terminal_growth import TerminalGrowthSettings
        env_keys = ('PREFERENCE_RA_LEVELS', 'PREFERENCE_EIS_LEVELS', 'PREFERENCE_PRTP_LEVELS',
                    'FRONTIER_PARAMETER_SPECS', 'FRONTIER_RA', 'FRONTIER_EIS', 'FRONTIER_PRTP')
        env = {key: value for key, value in os.environ.items() if key not in env_keys}
        with mock.patch.dict(os.environ, env, clear=True):
            values, labels, levels = preference.preference_grid_values()
            self.assertEqual(levels['EIS'], [0.9, 1.5, 2.2, 3.0])
            self.assertEqual(len(values), 48)
            self.assertFalse(np.any(values[:, index['EIS']] == 0.1))
            rows, names = frontier.parameter_spec_values()
        expected = {'high_ra': ('RA', 15.0), 'high_eis': ('EIS', 2.2),
                    'endogenous_learning': ('tech_scale', 1.5),
                    'low_ra': ('RA', 5.0), 'low_eis': ('EIS', 0.9)}
        self.assertEqual(names, list(expected))
        for row, name in zip(rows, names):
            key, target = expected[name]
            self.assertEqual(row[index[key]], target)
            changed = np.flatnonzero(row != RUN0_PARAMETER_VALUES)
            np.testing.assert_array_equal(changed, [index[key]])
        high = rows[names.index('high_eis')]
        self.assertFalse(TerminalGrowthSettings().admissible_mask(
            high[index['RA']], high[index['EIS']], high[index['PRTP']]))
        pref_run = next(run for run in CLUSTER_RUNS if run['name'].startswith('preference grid'))
        frontier_run = next(run for run in CLUSTER_RUNS if run['name'] == 'delay frontier preference and learning robustness')
        self.assertEqual(pref_run['array'], '1-144')
        self.assertEqual(frontier_run['array'], '1-20')
        self.assertEqual(frontier_run['env']['FRONTIER_PARAMETER_SPECS'].split(','), names)

    def test_learning_adjoint_and_interval_decomposition(self):
        for eis in (0.9, 1.0, 1.5):
            for backstop, gamma in (('s_curve', 0.0), ('cubic', 100000.0)):
                with self.subTest(eis=eis, backstop=backstop):
                    o = objective(eis=eis, backstop=backstop, gamma=gamma)
                    m = np.array([0.6, 1.2, 0.8])
                    value, grad = o.value_and_gradient(m)
                    self.assertAlmostEqual(value, o.utility(m), places=10)
                    d = o.diagnostics()
                    denominator = np.asarray(d['dollar_denominator'])
                    direct = np.asarray(d['direct_cost_gradient'])
                    learning = np.asarray(d['learning_gradient'])
                    climate = np.asarray(d['climate_gradient'])
                    self.assertTrue(np.all(denominator > 0))
                    np.testing.assert_allclose(grad, direct+learning+climate, atol=1e-10)
                    np.testing.assert_allclose(
                        climate,
                        np.asarray(d['warming_gradient'])+np.asarray(d['low_ghg_gradient']),
                        atol=1e-10)
                    np.testing.assert_allclose(np.asarray(d['interval_foc_gap_dollars_per_ton']),
                                               (direct+learning+climate)/denominator, atol=1e-9)
                    for i in range(len(m)):
                        h = 2e-5
                        plus, minus = m.copy(), m.copy()
                        plus[i] += h
                        minus[i] -= h
                        fd = (o.utility(plus)-o.utility(minus))/(2*h)
                        self.assertLess(abs(fd-grad[i]), 2e-8)

    def test_fixed_quantity_learning_matches_share_at_reference_and_gradient(self):
        dates = [0, 10, 35]
        old = objective(decision_times=dates)
        quantity = objective(decision_times=dates, learning_reference_year=2040)
        m = np.array([0.6, 1.2, 0.8])
        # Through 2040 the two numerators coincide; only the denominator differs.
        np.testing.assert_allclose(old.technology_by_year(m)[15],
                                   quantity.technology_by_year(m)[15], atol=1e-12)
        self.assertFalse(np.allclose(old.technology_by_year(m)[10],
                                        quantity.technology_by_year(m)[10]))
        value, gradient = quantity.value_and_gradient(m)
        self.assertTrue(np.isfinite(value))
        for i in range(len(m)):
            h = 2e-5
            plus, minus = m.copy(), m.copy()
            plus[i] += h
            minus[i] -= h
            fd = (quantity.utility(plus)-quantity.utility(minus))/(2*h)
            self.assertLess(abs(fd-gradient[i]), 2e-8)

    def test_cubic_cost_integrates_mac(self):
        c = objective(backstop='cubic', gamma=200000).cost
        for m in (0.2, 0.9, 1.1, 1.5):
            h = 1e-6
            fd = (c._raw_integrated_cost(m+h)-c._raw_integrated_cost(m-h))/(2*h)
            self.assertAlmostEqual(float(fd), float(c._raw_marginal_cost(m)), places=4)
        self.assertAlmostEqual(float(c._raw_marginal_cost(0.5)),
                               float(objective().cost._raw_marginal_cost(0.5)))

    def test_zero_learning_recovers_existing_flow_objective(self):
        o = objective(tech_scale=0.0)
        self.assertIsInstance(o, FiveYearFlowObjective)
        self.assertNotIsInstance(o, LearningFiveYearFlowObjective)

    def test_exogenous_technology_advances_between_decisions(self):
        o = objective(tech_scale=0.0, decision_times=[0, 10, 35])
        m = np.array([0.6, 0.8, 1.2])
        _, _, _, _, shares, _, _ = o._forward(m)
        self.assertEqual(o._flow_factors[0][0], o._flow_factors[5][0])
        for elapsed in (0, 5, 10, 15, 20):
            period, factor = o._flow_factors[elapsed]
            year = o.tree.base_year+elapsed
            emissions = float(np.interp(year, o.emit.times, o.emit.baseline_gtco2))
            expected = (emissions/86252.0
                *(1-0.015)**(year-o.cost.anchor_year)/(1.015)**elapsed)
            self.assertAlmostEqual(factor, expected, places=14)
        self.assertNotEqual(float(shares[0][0]), float(shares[5][0]))
        value, gradient = o.value_and_gradient(m)
        self.assertTrue(np.isfinite(value))
        diagnostics = o.diagnostics()
        interval_mac = (-diagnostics['direct_cost_gradient'][0]
                        / diagnostics['dollar_denominator'][0])
        decision_date_mac = ((1-0.015)**(2025-o.cost.anchor_year)
            *o.cost._raw_marginal_cost(m[0]))
        self.assertLess(interval_mac, decision_date_mac)
        for node in range(len(m)):
            plus, minus = m.copy(), m.copy()
            plus[node] += 2e-5
            minus[node] -= 2e-5
            finite = (o.utility(plus)-o.utility(minus))/4e-5
            self.assertLess(abs(gradient[node]-finite), 2e-8)

    def test_learning_accumulates_during_long_decision_interval(self):
        o = objective(decision_times=[0, 10, 35])
        m = np.array([0.6, 1.2, 0.8])
        matrices = o._learning_weight_matrices
        self.assertAlmostEqual(float((matrices[5] @ m)[0]), m[0])
        np.testing.assert_allclose(matrices[10] @ m, [m[0], m[0]])
        averages_2040 = matrices[15] @ m
        self.assertGreater(averages_2040[0], m[0])
        self.assertLess(averages_2040[0], m[1])
        self.assertGreater(averages_2040[1], m[0])
        self.assertLess(averages_2040[1], m[2])
        self.assertNotAlmostEqual(*o.technology_by_year(m)[15])
        decision_average = o.damage.average_mitigation(m, 1)
        np.testing.assert_allclose(matrices[10] @ m, decision_average,
                                   rtol=1e-10, atol=1e-10)
        for eis in (0.9, 1.0, 1.5):
            with self.subTest(eis=eis):
                current = objective(eis=eis, decision_times=[0, 10, 35])
                value, gradient = current.value_and_gradient(m)
                self.assertTrue(np.isfinite(value))
                diag = current.diagnostics()
                np.testing.assert_allclose(gradient,
                    np.asarray(diag['direct_cost_gradient'])
                    +np.asarray(diag['learning_gradient'])
                    +np.asarray(diag['climate_gradient']), atol=1e-10)
                for node in range(len(m)):
                    plus, minus = m.copy(), m.copy()
                    plus[node] += 2e-5
                    minus[node] -= 2e-5
                    finite = (current.utility(plus)-current.utility(minus))/4e-5
                    self.assertLess(abs(gradient[node]-finite), 2e-8)

    def test_extreme_backstop_integral_and_adjoint(self):
        o = objective(backstop='exp_cubic', gamma=200000.0)
        for m in (0.8, 1.0, 1.003, 1.01, 1.05, 1.1):
            h = 1e-6
            fd = (o.cost._raw_integrated_cost(m+h)
                  -o.cost._raw_integrated_cost(m-h))/(2*h)
            np.testing.assert_allclose(fd, o.cost._raw_marginal_cost(m),
                                       rtol=2e-5, atol=1e-3)
        m = np.array([0.6, 1.003, 0.8])
        value, gradient = o.value_and_gradient(m)
        self.assertTrue(np.isfinite(value))
        h = 2e-5
        plus, minus = m.copy(), m.copy()
        plus[1] += h
        minus[1] -= h
        fd = (o.utility(plus)-o.utility(minus))/(2*h)
        self.assertLess(abs(fd-gradient[1]), 2e-7)

    def test_backstop_cluster_task_mapping(self):
        from scripts.main_backstop_delay_cluster import CASES, task_configuration
        self.assertEqual(len(CASES), 8)
        self.assertEqual(task_configuration(1)[1], 5)
        self.assertEqual(task_configuration(15)[1], 15)
        self.assertEqual(task_configuration(15)[0]['economic_cap'], 1.0)
        self.assertEqual(task_configuration(10)[0]['backstop'], 'exp_cubic')
        self.assertEqual(task_configuration(1)[0]['label'], 'baseline_s_curve')
        self.assertEqual(task_configuration(16)[0]['curvature'], 0.5)
        self.assertEqual(task_configuration(19)[0]['curvature'], 1.5)
        self.assertEqual(task_configuration(22)[0]['curvature'], 2.0)
        self.assertEqual(task_configuration(24)[1], 15)
        with self.assertRaises(ValueError):
            task_configuration(25)

    def test_backstop_array_passes_curvature_to_revised_runner(self):
        from scripts import main_backstop_delay_cluster as backstop_array
        for task_id, expected in ((1, 1.0), (16, 0.5), (19, 1.5), (22, 2.0)):
            with self.subTest(task_id=task_id), \
                 mock.patch.dict(os.environ, {"SGE_TASK_ID": str(task_id)}, clear=False), \
                 mock.patch.object(backstop_array, "run_cluster_pair",
                     return_value={"optimal_certified": True,
                                   "delayed_certified": True}) as runner:
                backstop_array.main()
                params = runner.call_args.kwargs["param_vals"]
                self.assertEqual(params.shape, (1, 8))
                self.assertEqual(float(params[0, -1]), expected)
                self.assertEqual(runner.call_args.kwargs["output_metadata"]
                                 ["backstop_curvature"], expected)

    def test_state_export_retains_node_and_five_year_consumption(self):
        from scripts.export_revised_state import export_state_trees
        o = objective()
        m = np.array([0.6, 1.003, 0.8])
        value = o.utility(m)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            decision_count, flow_count, terminal_count = export_state_trees(o, m, folder, value)
            with (folder/'decision_state_nodes.csv').open() as handle:
                decision = list(csv.DictReader(handle))
            with (folder/'five_year_state_nodes.csv').open() as handle:
                flow = list(csv.DictReader(handle))
            with (folder/'terminal_state_nodes.csv').open() as handle:
                terminal = list(csv.DictReader(handle))
        self.assertEqual(decision_count, 3)
        self.assertEqual(flow_count, len(flow))
        self.assertEqual(terminal_count, len(terminal))
        self.assertEqual(decision[0]['year'], '2025')
        self.assertGreater(float(decision[0]['ghg_ppm']), 0)
        self.assertGreater(float(decision[0]['consumption_model_units']), 0)
        self.assertEqual(flow[-1]['is_terminal_continuation'], 'True')

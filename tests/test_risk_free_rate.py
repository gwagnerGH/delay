import unittest

import numpy as np

from src.utility import EZUtility


class _LinearMarginalUtility:
    """Minimal utility double with known date-zero and maturity MU."""

    period_len = 5.0

    class tree:
        decision_times = np.array([0.0, 5.0, 10.0])

    zero_coupon_bond_price = EZUtility.zero_coupon_bond_price
    ezclimate_term_structure_price = EZUtility.ezclimate_term_structure_price

    def adjusted_utility(self, m, first_period_consadj=0.0,
                         period_consadj=None, period_cons_eps=None, **unused):
        maturity_adjustment = 0.0
        if period_consadj is not None:
            maturity_adjustment = period_consadj[1]
        if period_cons_eps is not None:
            maturity_adjustment = period_cons_eps[-2]
        # MU at date zero is 2; MU of a date-five state-invariant payoff is .25.
        return 2.0 * first_period_consadj + 0.25 * maturity_adjustment


class TestRiskFreeRate(unittest.TestCase):
    def setUp(self):
        self.utility = _LinearMarginalUtility()

    def test_zero_coupon_price_is_marginal_utility_ratio(self):
        price = EZUtility.zero_coupon_bond_price(
            self.utility, np.array([0.0]), maturity_years=5.0
        )
        self.assertAlmostEqual(price, 0.125, places=10)

    def test_rate_is_annualized_from_bond_price(self):
        rate = EZUtility.risk_free_rate(
            self.utility, np.array([0.0]), maturity_years=5.0
        )
        self.assertAlmostEqual(rate, 0.125 ** (-1.0 / 5.0) - 1.0, places=10)

    def test_maturity_must_match_utility_grid(self):
        with self.assertRaises(ValueError):
            EZUtility.zero_coupon_bond_price(
                self.utility, np.array([0.0]), maturity_years=3.0
            )

    def test_ezclimate_term_structure_price_matches_its_utility_equivalence(self):
        price = EZUtility.ezclimate_term_structure_price(
            self.utility, np.array([0.0]), payment=0.01
        )
        self.assertAlmostEqual(price, 0.125, places=10)

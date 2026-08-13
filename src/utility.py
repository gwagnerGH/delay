"""Utility class.

Adam M. Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
3.21.2022

Utility class used for calculating the economic utility and related
quantities in CAP6. We implement Epstein-Zin preferences for
our utility calculations below.
"""

import numpy as np
from scipy.optimize import brentq

from src.storage_tree import BigStorageTree, SmallStorageTree

np.seterr(all='ignore')

class EZUtility(object):
    """Calculation of Epstein-Zin utility for the CAP6 model.

    The Epstein-Zin utility allows for different rates of substitution across
    time and states. For specification see DLW-paper (2017) and BPW-paper (2023).

    Parameters
    ----------
    tree : `TreeModel` object
        tree structure used
    damage : `Damage` object
        class that provides damage methods
    cost : `Cost` object
        class that provides cost methods
    period_len : float
        subinterval length
    eis : float, optional
        elasticity of intertemporal substitution
    ra : float, optional
        risk-aversion
    time_pref : float, optional
        pure rate of time preference

    Attributes
    ----------
    tree : `TreeModel` object
        tree structure used
    damage : `Damage` object
        class that provides damage methods
    cost : `Cost` object
        class that provides cost methods
    period_len : float
        subinterval length
    cons_growth : float
        consumption growth
    growth_term : float
        1 + cons_growth
    r : float
        the parameter rho from the DLW-paper
    a : float
        the parameter alpha from the DLW-paper
    b : float
        the parameter beta from the DLW-paper
    potential_cons: float
        not sure, but seems like potential_cons = (1 + g)^t, where g is
        constant growth rate.
    """

    _LOG_EIS_TOL = 1e-12

    def __init__(self, tree, damage, cost, period_len, eis=0.9, ra=7.0,
                 time_pref=0.005, cons_growth=0.015):
        self.tree = tree
        self.damage = damage
        self.cost = cost
        self.period_len = period_len
        self.cons_growth = cons_growth
        self.growth_term = 1.0 + self.cons_growth
        self.r = 1.0 - 1.0/eis
        self.is_log_eis = abs(self.r) <= self._LOG_EIS_TOL
        self.a = 1.0 - ra
        self.b = (1.0-time_pref)**period_len
        self.potential_cons = (np.ones(self.tree.decision_times.shape) \
                               + self.cons_growth)**self.tree.decision_times

    def _intertemporal_aggregate(self, consumption, cert_equiv):
        """Stable CES aggregate, including the EIS=1 logarithmic limit."""

        consumption = np.asarray(consumption)
        cert_equiv = np.asarray(cert_equiv)
        log_consumption = np.log(consumption)
        log_ce_ratio = np.log(cert_equiv) - log_consumption

        if self.is_log_eis:
            log_utility = log_consumption + self.b * log_ce_ratio
        else:
            log_utility = log_consumption + (
                np.log1p(self.b * np.expm1(self.r * log_ce_ratio)) / self.r
            )
        return np.exp(log_utility)

    def _terminal_utility(self, consumption):
        """Stable terminal continuation value, including the EIS=1 limit."""

        log_growth = np.log(self.growth_term)
        if self.is_log_eis:
            continuation_log = self.b * log_growth / (1.0 - self.b)
        else:
            continuation_log = -np.log1p(
                -(self.b / (1.0 - self.b))
                * np.expm1(self.r * log_growth)
            ) / self.r
        return np.asarray(consumption) * np.exp(continuation_log)

    def utility(self, m, return_trees=False):
        """Calculating utility for the specific mitigation decisions `m`.

        Parameters
        ----------
        m : ndarray or list
            array of mitigations
        return_trees : bool
            True if method should return trees calculated in producing the utility

        Returns
        -------
        ndarray or tuple
            tuple of `BaseStorageTree` if return_trees else ndarray with utility at period 0

        Examples
        ---------
        Assuming we have declared a EZUtility object as 'ezu' and have a mitigation array 'm'

        >>> ezu.utility(m)
        array([ 9.83391921])
        >>> tree_dict = ezu.utility(m, return_trees=True)
        """

        utility_tree = BigStorageTree(subinterval_len=self.period_len,
                                      decision_times=self.tree.decision_times)
        cons_tree = BigStorageTree(subinterval_len=self.period_len,
                                   decision_times=self.tree.decision_times)
        ce_tree = BigStorageTree(subinterval_len=self.period_len,
                                 decision_times=self.tree.decision_times)
        cost_tree = SmallStorageTree(decision_times=self.tree.decision_times)

        self._end_period_utility(m, utility_tree, cons_tree, cost_tree)

        # makes generator object and iterates over it to fill the utility tree
        # with values in each period 
        it = self._utility_generator(m, utility_tree, cons_tree, cost_tree, ce_tree)
        for u, period in it:
            utility_tree.set_value(period, u)

        if return_trees:
            return {'Utility':utility_tree, 'Consumption':cons_tree,
                    'Cost':cost_tree, 'CertainEquivalence':ce_tree}
        # returns first value
        return utility_tree[0]

    def _end_period_utility(self, m, utility_tree, cons_tree, cost_tree,
                            period_consmult=None):
        """Calculate the terminal utility.

        Calculates the utility in the final period and stores the values in the
        utility_tree object.

        Parameters
        ----------
        m: nd array
            Array of mitigation valus
        utility_tree: `BigStorageTree` object
            storage tree of utility values
        cons_tree: `BigStorageTree` object
            storage tree of consumption values
        cost_tree: `SmallStorageTree` object
            storage tree of cost values
        """

        # calc average mitigation and damages in the period 
        period_ave_mitigation = self.damage.average_mitigation(m,
                                                               self.tree.num_periods,
                                                               is_last=True)
        period_damage = self.damage.damage_function(m,
                                                    self.tree.num_periods,
                                                    is_last=True)

        # get a tuple of nodes in period
        damage_nodes = self.tree.get_nodes_in_period(self.tree.num_periods)

        # mitigation in the period
        period_mitigation = m[damage_nodes[0]:damage_nodes[1]+1]

        # calc cost in period, store value, and calculate the remaining values
        period_cost = self.cost.cost(self.tree.num_periods, period_mitigation,
                                     period_ave_mitigation)
        cost_tree.set_value(cost_tree.last_period, period_cost)
        period_consumption = self.potential_cons[-1] * (1.0 - period_damage)
        period_consumption[period_consumption<=0.0] = 1e-18
        if period_consmult is not None:
            terminal_index = int(round(
                float(self.tree.decision_times[-1]) / float(self.period_len)
            ))
            if terminal_index < len(period_consmult):
                multiplier = float(period_consmult[terminal_index])
                if not np.isfinite(multiplier) or multiplier <= 0.0:
                    raise ValueError("period_consmult values must be finite and positive")
                period_consumption *= multiplier
        cons_tree.set_value(cons_tree.last_period, period_consumption)
        utility_tree.set_value(
            utility_tree.last_period,
            self._terminal_utility(cons_tree.last),
        )

    def _utility_generator(self, m, utility_tree, cons_tree, cost_tree,
                           ce_tree, cons_adj=0.0, period_consadj=None,
                           period_consmult=None):
        """Generator fora calculating utility for each utility period besides
        the terminal utility.

        Parameters
        ----------
        m: nd array
            Array of mitigation valus
        utility_tree: `BigStorageTree` object
            storage tree of utility values
        cons_tree: `BigStorageTree` object
            storage tree of consumption values
        cost_tree: `SmallStorageTree` object
            storage tree of cost values
        ce_tree: `BigStorageTree` object
            storage tree of certain equivalence values
        cons_adj: float
            constant adjustment for first period utility
        period_consadj: ndarray, optional
            exact consumption-flow adjustment by utility subperiod index.
        period_consmult: ndarray, optional
            exact multiplicative consumption adjustment by utility subperiod
            index. Values must be finite and strictly positive.
        """

        periods = utility_tree.periods[::-1]

        for period in periods[1:]:
            damage_period = utility_tree.between_decision_times(period)
            cert_equiv = self._certain_equivalence(period, damage_period, utility_tree)

            if utility_tree.is_decision_period(period+self.period_len):
                damage_nodes = self.tree.get_nodes_in_period(damage_period)
                period_mitigation = m[damage_nodes[0]:damage_nodes[1]+1]
                period_ave_mitigation = self.damage.average_mitigation(m, damage_period)
                period_cost = self.cost.cost(damage_period, period_mitigation,
                                             period_ave_mitigation)
                period_damage = self.damage.damage_function(m, damage_period)
                cost_tree.set_value(cost_tree.index_below(period+self.period_len),
                                    period_cost)

            period_consumption = self.potential_cons[damage_period] \
                                    * (1.0 - period_damage) * (1.0 - period_cost)
            period_consumption[period_consumption <= 0.0] = 1e-18

            if not utility_tree.is_decision_period(period):
                next_consumption = cons_tree.get_next_period_array(period)
                segment = period - utility_tree.decision_times[damage_period]
                interval = segment + utility_tree.subinterval_len

                if utility_tree.is_decision_period(period+self.period_len):
                    if period < utility_tree.decision_times[-2]:
                        next_cost = cost_tree[period+self.period_len]
                        next_consumption *= (1.0 - np.repeat(period_cost,2)) / (1.0 - next_cost)
                        next_consumption[next_consumption<=0.0] = 1e-18

                if period < utility_tree.decision_times[-2]:
                    temp_consumption = next_consumption/np.repeat(period_consumption,2)
                    period_consumption = np.sign(temp_consumption)*(np.abs(temp_consumption)**(segment/float(interval))) \
                                         * np.repeat(period_consumption,2)
                else:
                    temp_consumption = next_consumption/period_consumption
                    period_consumption = np.sign(temp_consumption)*(np.abs(temp_consumption)**(segment/float(interval))) \
                                         * period_consumption
            if period == 0:
                period_consumption += cons_adj
            if period_consadj is not None:
                period_index = int(round(float(period) / float(self.period_len)))
                if 0 <= period_index < len(period_consadj):
                    period_consumption += period_consadj[period_index]
            if period_consmult is not None:
                period_index = int(round(float(period) / float(self.period_len)))
                if 0 <= period_index < len(period_consmult):
                    multiplier = float(period_consmult[period_index])
                    if not np.isfinite(multiplier) or multiplier <= 0.0:
                        raise ValueError("period_consmult values must be finite and positive")
                    period_consumption *= multiplier

            ce_term = self.b * cert_equiv**self.r
            ce_tree.set_value(period, ce_term)
            cons_tree.set_value(period, period_consumption)
            u = self._intertemporal_aggregate(period_consumption, cert_equiv)
            yield u, period

    def _certain_equivalence(self, period, damage_period, utility_tree):
        """Calculate ceartainty equivalence utility.

        If we are between decision nodes, i.e. no branching, then certainty
        equivalent utility at time period depends only on the utility next
        period given information known today. Otherwise the certainty
        equivalent utility is the ability weighted sum of next period utility
        over the partition reachable from the state.

        Parameters
        ----------
        period: int
            The period we are at
        damage_period: nd array
            array of damages for each node in the period, sorted from worst to best
        utility_tree: `BigStorageTree` object
            tree which stores all utility values
        """

        if utility_tree.is_information_period(period):
            damage_nodes = self.tree.get_nodes_in_period(damage_period+1)
            probs = self.tree.node_prob[damage_nodes[0]:damage_nodes[1]+1]
            even_probs = probs[::2]
            odd_probs = probs[1::2]
            even_util = ((utility_tree.get_next_period_array(period)[::2])**self.a) * even_probs
            odd_util = ((utility_tree.get_next_period_array(period)[1::2])**self.a) * odd_probs
            ave_util = (even_util + odd_util) / (even_probs + odd_probs)
            cert_equiv = ave_util**(1.0/self.a)
        else:
            # no branching implies certainty equivalent utility at time period depends only on
            # the utility next period given information known today
            cert_equiv = utility_tree.get_next_period_array(period)

        return cert_equiv

    def adjusted_utility(self, m, period_cons_eps=None, node_cons_eps=None,
                         final_cons_eps=0.0, first_period_consadj=0.0,
                         period_consadj=None, period_consmult=None,
                         return_trees=False):
        """Calculating aadjusted utility for sensitivity analysis.

        Used e.g. to find zero-coupon bond price.
        Values in parameters are used to adjust utility in different ways.

        Parameters
        ----------
        m : ndarray
            array of mitigations
        period_cons_eps : ndarray, optional
            array of increases in consumption per period
        node_cons_eps : `SmallStorageTree`, optional
            increases in consumption per node
        final_cons_eps : float, optional
            value to increase the final utilities by
        first_period_consadj : float, optional
            value to increase consumption at period 0 by
        period_consadj : ndarray, optional
            exact consumption-flow adjustment by utility subperiod index. This
            differs from period_cons_eps, which is a marginal utility
            perturbation.
        period_consmult : ndarray, optional
            exact multiplicative consumption adjustment by utility subperiod
            index. A value of 1 leaves consumption unchanged.
        return_trees : bool, optional
            True if method should return trees calculated in producing the
            utility

        Returns
        -------
        ndarray or tuple
            tuple of `BaseStorageTree` if return_trees else ndarray with utility at period 0

        Examples
        ---------
        Assuming we have declared a EZUtility object as 'ezu' and have a mitigation array 'm'

        >>> ezu.adjusted_utility(m, final_cons_eps=0.1)
        array([ 9.83424045])
        >>> tree_dict = ezu.adjusted_utility(m, final_cons_eps=0.1, return_trees=True)

        >>> arr = np.zeros(int(ezu.decision_times[-1]/ezu.period_len) + 1)
        >>> arr[-1] = 0.1
        >>> ezu.adjusted_utility(m, period_cons_eps=arr)
        array([ 9.83424045])

        >>> bst = BigStorageTree(5.0, [0, 15, 45, 85, 185, 285, 385])
        >>> bst.set_value(bst.last_period, np.repeat(0.01, len(bst.last)))
        >>> ezu.adjusted_utility(m, node_cons_eps=bst)
        array([ 9.83391921])

        The last example differs from the rest in that the last values of the `node_cons_eps` will never be
        used. Hence if you want to update the last period consumption, use one of these two methods.

        >>> ezu.adjusted_utility(m, first_period_consadj=0.01)
        array([ 9.84518772])
        """

        utility_tree = BigStorageTree(subinterval_len=self.period_len,
                                      decision_times=self.tree.decision_times)
        cons_tree = BigStorageTree(subinterval_len=self.period_len,
                                   decision_times=self.tree.decision_times)
        ce_tree = BigStorageTree(subinterval_len=self.period_len,
                                 decision_times=self.tree.decision_times)
        cost_tree = SmallStorageTree(decision_times=self.tree.decision_times)

        periods = utility_tree.periods[::-1]
        if period_cons_eps is None:
            period_cons_eps = np.zeros(len(periods))
        if node_cons_eps is None:
            node_cons_eps = BigStorageTree(subinterval_len=self.period_len,
                                           decision_times=self.tree.decision_times)
        self._end_period_utility(
            m, utility_tree, cons_tree, cost_tree,
            period_consmult=period_consmult,
        )

        it = self._utility_generator(m, utility_tree, cons_tree, cost_tree,
                                     ce_tree, first_period_consadj,
                                     period_consadj=period_consadj,
                                     period_consmult=period_consmult)
        i = len(utility_tree)-2
        for u, period in it:
            if period == periods[1]:
                final_adjustment = final_cons_eps + period_cons_eps[-1] + node_cons_eps.last
                current_adjustment = period_cons_eps[i] + node_cons_eps.tree[period]
                if np.any(final_adjustment != 0.0) or np.any(current_adjustment != 0.0):
                    self._require_nonlog_marginal_adjustments()
                    mu_0 = (1.0-self.b) * (u/cons_tree[period])**(1.0-self.r)
                    next_term = self.b * (1.0-self.b) / (1.0-self.b*self.growth_term**self.r)
                    mu_1 = (u**(1.0-self.r)) * next_term * (cons_tree.last**(self.r-1.0))
                    u += final_adjustment * mu_1
                    u += current_adjustment * mu_0
                utility_tree.set_value(period, u)
            else:
                current_adjustment = period_cons_eps[i] + node_cons_eps.tree[period]
                if np.any(current_adjustment != 0.0):
                    self._require_nonlog_marginal_adjustments()
                    mu_0, m_1, m_2 = self._period_marginal_utility(
                        period, utility_tree, cons_tree, ce_tree
                    )
                    u += current_adjustment * mu_0
                utility_tree.set_value(period, u)
            i -= 1

        if return_trees:
            return utility_tree, cons_tree, cost_tree, ce_tree

        return utility_tree.tree[0]

    def zero_coupon_bond_price(self, m, maturity_years=None, payoff=1.0,
                               epsilon=1e-5):
        """Price a real zero-coupon bond from the model's pricing kernel.

        The bond pays ``payoff`` units of consumption in *every* state at
        ``maturity_years``.  Its date-zero price is the ratio of the marginal
        utility of that payoff to the marginal utility of date-zero
        consumption.  Thus, unlike a deterministic ``PRTP + growth / EIS``
        approximation, the result includes the Epstein--Zin certainty
        equivalents and all uncertainty in the continuation tree.

        Parameters
        ----------
        m : ndarray
            Mitigation policy at which to value the bond.
        maturity_years : float, optional
            Bond maturity measured from date zero. Defaults to one utility
            subperiod (normally five years).
        payoff : float, optional
            State-contingent consumption payoff at maturity. It must be
            positive; the returned price scales linearly for marginal payoffs.
        epsilon : float, optional
            Central-difference step used to obtain marginal utilities.

        Returns
        -------
        float
            Date-zero consumption price of the bond.
        """
        if maturity_years is None:
            maturity_years = self.period_len
        maturity_years = float(maturity_years)
        payoff = float(payoff)
        epsilon = float(epsilon)
        if maturity_years <= 0.0:
            raise ValueError("maturity_years must be positive")
        if payoff <= 0.0:
            raise ValueError("payoff must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        period_index = int(round(maturity_years / self.period_len))
        if not np.isclose(period_index * self.period_len, maturity_years):
            raise ValueError("maturity_years must be a multiple of period_len")
        max_index = int(round(self.tree.decision_times[-1] / self.period_len))
        if period_index > max_index:
            raise ValueError("maturity_years is beyond the model horizon")

        # ``period_consadj`` is an exact (not linearized) adjustment to
        # consumption at the requested utility date in every reachable state.
        adjustments = np.zeros(max_index + 1)
        adjustments[period_index] = epsilon * payoff
        target_up = self.adjusted_utility(m, period_consadj=adjustments)
        adjustments[period_index] = -epsilon * payoff
        target_down = self.adjusted_utility(m, period_consadj=adjustments)
        marginal_payoff_utility = float(
            np.asarray(target_up - target_down).reshape(-1)[0]
        ) / (2.0 * epsilon)

        initial_up = self.adjusted_utility(
            m, first_period_consadj=epsilon
        )
        initial_down = self.adjusted_utility(
            m, first_period_consadj=-epsilon
        )
        marginal_initial_utility = float(
            np.asarray(initial_up - initial_down).reshape(-1)[0]
        ) / (2.0 * epsilon)
        if not np.isfinite(marginal_payoff_utility) or not np.isfinite(marginal_initial_utility):
            raise ValueError("non-finite marginal utility while pricing bond")
        if marginal_initial_utility <= 0.0:
            raise ValueError("date-zero marginal utility must be positive")
        return marginal_payoff_utility / marginal_initial_utility

    def risk_free_rate(self, m, maturity_years=None, payoff=1.0,
                       epsilon=1e-5):
        """Return the annualized effective real risk-free rate.

        This reports ``bond_price**(-1 / maturity_years) - 1``, where
        ``bond_price`` is calculated by :meth:`zero_coupon_bond_price` from
        the full Epstein--Zin utility tree.
        """
        if maturity_years is None:
            maturity_years = self.period_len
        maturity_years = float(maturity_years)
        price = self.zero_coupon_bond_price(
            m, maturity_years=maturity_years, payoff=payoff, epsilon=epsilon
        )
        if not np.isfinite(price) or price <= 0.0:
            raise ValueError("bond price must be finite and positive")
        return price**(-1.0 / maturity_years) - 1.0

    def ezclimate_term_structure_price(self, m, payment=0.01,
                                       lower=0.0, upper=1.5):
        """Reproduce EZClimate's deferred-perpetuity bond-price convention.

        This is intentionally distinct from :meth:`zero_coupon_bond_price`.
        The original EZClimate ``find_term_structure`` routine applies a
        marginal consumption payment in the penultimate utility subperiod and
        finds the date-zero payment with equal utility.  Its output is then
        converted to a perpetuity yield beginning in that penultimate period.
        """
        payment = float(payment)
        if payment <= 0.0:
            raise ValueError("payment must be positive")
        max_index = int(round(self.tree.decision_times[-1] / self.period_len))
        if max_index < 1:
            raise ValueError("model horizon must contain at least two subperiods")

        period_cons_eps = np.zeros(max_index + 1)
        period_cons_eps[-2] = payment
        target_utility = float(np.asarray(
            self.adjusted_utility(m, period_cons_eps=period_cons_eps)
        ).reshape(-1)[0])

        def objective(price):
            initial_utility = float(np.asarray(self.adjusted_utility(
                m, first_period_consadj=payment * price
            )).reshape(-1)[0])
            return target_utility - initial_utility

        return brentq(objective, float(lower), float(upper))

    @staticmethod
    def ezclimate_perpetuity_yield(price, start_year,
                                   lower=0.1, upper=100000.0):
        """Convert an EZClimate deferred-perpetuity price to percent yield.

        This is algebraically identical to EZClimate's ``perpetuity_yield``;
        its return value is in percentage points (e.g. ``3.11``, not ``.0311``).
        """
        price = float(price)
        start_year = float(start_year)
        if price <= 0.0 or start_year <= 0.0:
            raise ValueError("price and start_year must be positive")

        def objective(yield_percent):
            return price - (100.0 / (yield_percent + 100.0))**start_year * (
                yield_percent + 100.0
            ) / yield_percent

        return brentq(objective, float(lower), float(upper))

    def ezclimate_deferred_perpetuity_yield(self, m, payment=0.01):
        """Return the original EZClimate-style deferred-perpetuity yield.

        The perpetuity starts one utility subperiod before the terminal model
        date. The result is in percentage points, matching EZClimate output.
        """
        price = self.ezclimate_term_structure_price(m, payment=payment)
        start_year = float(self.tree.decision_times[-1] - self.period_len)
        return self.ezclimate_perpetuity_yield(price, start_year)

    def _require_nonlog_marginal_adjustments(self):
        if self.is_log_eis:
            raise NotImplementedError(
                "Linearized marginal-utility adjustments are not implemented "
                "for EIS=1. Use first_period_consadj or period_consadj so utility "
                "is recomputed from exact consumption adjustments."
            )

    def _period_marginal_utility(self, period, utility_tree, cons_tree, ce_tree):
        """Marginal utility for each node in a period.

        Parameters
        ----------
        period: int
            the current period
        utility_tree: `BigStorageTree` object
            storage tree containing utility values
        cons_tree: `SmallStorageTree` object
            storage tree containing consumption values
        ce_tree: `BigStorageTree` object
            storage tree containing certain equivalence values

        Returns
        -------
        m_0: float
            marginal utility with respect to consumption function
        m_1: float
            marginal utility with respect to consumption next period
        m_2: float
            marginal utility with respect to last period consumption
        """

        damage_period = utility_tree.between_decision_times(period)
        mu_0 = self._mu_0(cons_tree[period], ce_tree[period])

        prev_ce = ce_tree.get_next_period_array(period)
        prev_cons = cons_tree.get_next_period_array(period)
        if utility_tree.is_information_period(period):
            probs = self.tree.get_probs_in_period(damage_period+1)
            up_prob = np.array([probs[i]/(probs[i]+probs[i+1]) for i in range(0, len(probs), 2)])
            down_prob = 1.0 - up_prob

            up_cons = prev_cons[::2]
            down_cons = prev_cons[1::2]
            up_ce = prev_ce[::2]
            down_ce = prev_ce[1::2]

            mu_1 = self._mu_1(cons_tree[period], up_prob, up_cons, down_cons, up_ce, down_ce)
            mu_2 = self._mu_1(cons_tree[period], down_prob, down_cons, up_cons, down_ce, up_ce)
            return mu_0, mu_1, mu_2
        else:
            mu_1 = self._mu_2(cons_tree[period], prev_cons, prev_ce)
            return mu_0, mu_1, None

    def _mu_0(self, cons, ce_term):
        """Marginal utility with respect to consumption function.

        Parameters
        ----------
        cons: float
            consumption value
        ce_term: float
            certain equivalence value

        Returns
        -------
        t1 * t2: float
            the marginal utility w.r.t consumption function.
        """

        t1 = (1.0 - self.b)*cons**(self.r-1.0)
        t2 = (ce_term - (self.b-1.0)*cons**self.r)**((1.0/self.r)-1.0)
        return t1 * t2

    def _mu_1(self, cons, prob, cons_1, cons_2, ce_1, ce_2):
        """ marginal utility with respect to consumption next period.
        Parameters
        ----------
        cons: float
            consumption value
        prob: float
            probability of making move to the next node we're considering
        cons_1, ce_1: float
            consumption/certain equivalence of up-move node
        cons_2, ce_2: float
            consumption/certain equivalence of down-move node

        Returns
        -------
        t1 * t2 * t3 * t5: float
            the marginal utility w.r.t the next period.
        """

        t1 = (1.0-self.b) * self.b * prob * cons_1**(self.r-1.0)
        t2 = (ce_1 - (self.b-1.0) * cons_1**self.r )**((self.a/self.r)-1)
        t3 = (prob * (ce_1 - (self.b*(cons_1**self.r)) + cons_1**self.r)**(self.a/self.r) \
             + (1.0-prob) * (ce_2 - (self.b-1.0) * cons_2**self.r)**(self.a/self.r))**((self.r/self.a)-1.0)
        t4 = prob * (ce_1-self.b * (cons_1**self.r) + cons_1**self.r)**(self.a/self.r) \
             + (1.0-prob) * (ce_2 - self.b * (cons_2**self.r) + cons_2**self.r)**(self.a/self.r)
        t5 = (self.b * t4**(self.r/self.a) - (self.b-1.0) * cons**self.r )**((1.0/self.r)-1.0)

        return t1 * t2 * t3 * t5

    def _mu_2(self, cons, prev_cons, ce_term):
        """Marginal utility with respect to last period consumption.

        Parameters
        ----------
        cons: float
            consumption value at the node
        prev_cons: float
            consumption at the previous node
        ce_term: float
            certain equivalence at the current node

        Returns
        -------
        t1 * t2: float
            marginal utility with respect to the last period of consumption.
        """

        t1 = (1.0-self.b) * self.b * prev_cons**(self.r-1.0)
        t2 = ((1.0 - self.b) * cons**self.r - (self.b - 1.0) * self.b \
             * prev_cons**self.r + self.b * ce_term)**((1.0/self.r)-1.0)
        return t1 * t2

    def partial_grad(self, m, i, delta=1e-8):
        """Calculate the ith element of the gradient vector.

        Parameters
        ----------
        m : ndarray
            array of mitigations
        i : int
            node to calculate partial grad for

        Returns
        -------
        float
            gradient element
        """

        m_copy = m.copy()
        m_copy[i] -= delta
        minus_utility = self.utility(m_copy)
        m_copy[i] += 2*delta
        plus_utility = self.utility(m_copy)
        grad = (plus_utility-minus_utility) / (2*delta)
        return grad

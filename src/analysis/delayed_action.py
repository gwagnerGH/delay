"""Delayed action analysis for CAP6.

This module implements delayed action functionality similar to EZClimate's
constraint_first_period function. It allows forcing mitigation to be zero
for the first N periods to analyze the economic cost of delaying climate action.

Addition by Theo 
"""

import numpy as np
from scipy.optimize import brentq
from src.optimization import GeneticAlgorithm, GradientSearch


def get_delay_nodes(tree, delay_periods):
    """Get the node indices that should be constrained for a given delay.
    
    Helper function to determine which decision nodes fall within the first
    `delay_periods` periods of the model.
    
    Parameters
    ----------
    tree : `TreeModel` object
        tree model containing the decision structure
    delay_periods : int
        number of initial periods to constrain
    
    Returns
    -------
    ndarray
        array of node indices to be constrained
    
    Examples
    --------
    >>> t = TreeModel(decision_times=[0, 5, 35, 75, 125, 175, 225])
    >>> # Get nodes in first 2 periods (years 0-40)
    >>> delay_nodes = get_delay_nodes(t, delay_periods=2)
    """
    fixed_indices = []
    for period in range(delay_periods):
        start_node, end_node = tree.get_nodes_in_period(period)
        for node in range(start_node, end_node + 1):
            fixed_indices.append(node)
    
    return np.array(fixed_indices)



def find_consumption_equivalence(m_delayed, m_optimal, u_delay, u_optimal=None, method='first_period',
                                   a=-150, b=150, tol=1e-8, compensation_years=5.0):
    """Find the consumption compensation needed to equalize utilities between delayed and optimal paths.
    
    This function uses the same methodology as EZClimate's find_bec() function.
    Two methods are supported:
    - 'first_period': Adjust only period-0 consumption (matches EZClimate's find_bec)
    - 'first_years': Uniform flow adjustment to the first N calendar years'
      consumption, with N controlled by compensation_years
    - 'permanent': Uniform percentage adjustment to ALL periods' consumption
    
    Parameters
    ----------
    m_delayed : ndarray
        mitigation array from delayed action scenario (corresponds to 'm' in find_bec)
    m_optimal : ndarray
        mitigation array from optimal (unconstrained) scenario
    utility : `EZUtility` object
        utility object for calculations
    method : str, optional
        'first_period', 'first_years', or 'permanent' (default: 'first_period')
    a : float, optional
        lower bound for root finding (default: -150)
    b : float, optional
        upper bound for root finding (default: 150)
    tol : float, optional
        tolerance for root finding (default: 1e-8)
    
    Returns
    -------
    float
        consumption compensation value (delta_c or g, depending on method)
        - For 'first_period': absolute consumption increase in first period
        - For 'first_years': absolute annual-flow increase in the first N years
        - For 'permanent': fractional increase (e.g., 0.05 = 5% increase across all periods)
    
    Examples
    --------
    >>> # First-period compensation (EZClimate method)
    >>> delta_c = find_consumption_equivalence(m_del, m_opt, u, method='first_period')
    >>> print(f"First-period compensation: {delta_c:.6f}")
    
    >>> # Permanent uniform compensation
    >>> g = find_consumption_equivalence(m_del, m_opt, u, method='permanent')
    >>> print(f"Permanent consumption increase needed: {g*100:.2f}%")
    
    Note
    ----
    This implements the same algorithm as EZClimate's find_bec():
        - Solves: U(m_delayed + δ) - U(m_delayed) - constraint_cost = 0
        - Where constraint_cost = U(m_optimal) - U(m_delayed)
        - Which simplifies to: U(m_delayed + δ) = U(m_optimal)
    """
    
    # Calculate utilities and constraint cost (matching EZClimate's ConstraintAnalysis._constraint_cost)
    opt_u = u_optimal.utility(m_optimal)
    cfp_u = u_delay.utility(m_delayed)

    if isinstance(opt_u, np.ndarray):
        opt_u = float(opt_u[0])
    if isinstance(cfp_u, np.ndarray):
        cfp_u = float(cfp_u[0])
    
    constraint_cost = opt_u - cfp_u
    
    print(f"Consumption Equivalence Calculation:")
    print(f"Optimal utility: {opt_u:.10f}")
    print(f"Delayed utility: {cfp_u:.10f}")
    print(f"Constraint cost: {constraint_cost:.10f} ({constraint_cost/opt_u*100:.4f}%)")
    
    if method == 'first_period':
        def min_func(delta_con):
            base_utility = u_delay.utility(m_delayed)
            new_utility = u_delay.adjusted_utility(m_delayed, first_period_consadj=delta_con)
            
            if isinstance(base_utility, np.ndarray):
                base_utility = float(base_utility[0])
            if isinstance(new_utility, np.ndarray):
                new_utility = float(new_utility[0])
            
            return new_utility - base_utility - constraint_cost
        
        try:
            delta_c = brentq(min_func, a, b, xtol=tol)
            print(f"\nFirst-period compensation: {delta_c:.6f}")
            print(f"({'increase' if delta_c > 0 else 'decrease'} of {abs(delta_c):.6f} in period-0 consumption)")
            
            u_check = u_delay.adjusted_utility(m_delayed, first_period_consadj=delta_c)
            if isinstance(u_check, np.ndarray):
                u_check = float(u_check[0])
            
            print(f"\nVerification:")
            print(f" Adjusted utility:  {u_check:.10f}")
            print(f" Target utility:    {opt_u:.10f}")
            print(f" Difference:        {abs(u_check - opt_u):.2e}")
            
            return delta_c
        except ValueError as e:
            print(f"\nError: Root not found in interval [{a}, {b}]")
            print(f"Try expanding the search interval.")
            print(f"Error: {e}")
            print(f"{'='*80}\n")
            return None

    elif method == 'first_years':
        if compensation_years <= 0:
            raise ValueError("compensation_years must be positive.")
        if not np.isclose(compensation_years / u_delay.period_len,
                          round(compensation_years / u_delay.period_len)):
            raise ValueError(
                "compensation_years must be an integer multiple of the utility "
                f"period length. Got compensation_years={compensation_years} "
                f"and period_len={u_delay.period_len}."
            )

        n_comp_periods = int(round(compensation_years / u_delay.period_len))
        n_periods = int(round(u_delay.tree.decision_times[-1] / u_delay.period_len)) + 1

        def compensation_schedule(delta_con):
            schedule = np.zeros(n_periods)
            schedule[:n_comp_periods] = delta_con
            return schedule

        def min_func(delta_con):
            base_utility = u_delay.utility(m_delayed)
            new_utility = u_delay.adjusted_utility(
                m_delayed,
                period_consadj=compensation_schedule(delta_con)
            )

            if isinstance(base_utility, np.ndarray):
                base_utility = float(base_utility[0])
            if isinstance(new_utility, np.ndarray):
                new_utility = float(new_utility[0])

            return new_utility - base_utility - constraint_cost

        try:
            delta_c = brentq(min_func, a, b, xtol=tol)
            print(f"\nFirst-{compensation_years:g}-year compensation: {delta_c:.6f}")
            print(
                f"({'increase' if delta_c > 0 else 'decrease'} of "
                f"{abs(delta_c):.6f} in each compensated annual flow)"
            )

            u_check = u_delay.adjusted_utility(
                m_delayed,
                period_consadj=compensation_schedule(delta_c)
            )
            if isinstance(u_check, np.ndarray):
                u_check = float(u_check[0])

            print(f"\nVerification:")
            print(f" Adjusted utility:  {u_check:.10f}")
            print(f" Target utility:    {opt_u:.10f}")
            print(f" Difference:        {abs(u_check - opt_u):.2e}")

            return delta_c
        except ValueError as e:
            print(f"\nError: Root not found in interval [{a}, {b}]")
            print(f"Try expanding the search interval.")
            print(f"Error: {e}")
            print(f"{'='*80}\n")
            return None
            
    elif method == 'permanent':
        print("\nPermanent method is currently disabled due to utility object issues.")
        return None  # Permanent method disabled for now due to utility object issues
    
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'first_period', 'first_years', "
            "or 'permanent'."
        )


class ConstraintAnalysis(object):
    """Calculate comprehensive deadweight loss analysis for delayed climate action.
    
    This class implements the same methodology as EZClimate's ConstraintAnalysis,
    adapted for CAP6's delayed action scenarios with different tree structures.
    It calculates:
    - Consumption compensation needed to equalize utilities
    - Deadweight cost per ton of emissions
    - Marginal benefits and costs of emission reductions
    
    Parameters
    ----------
    u_delay : `EZUtility` object
        utility object for delayed scenario (uses delayed tree structure)
    u_optimal : `EZUtility` object
        utility object for optimal scenario (uses baseline tree structure)
    m_delayed : ndarray
        mitigation array from delayed action scenario (constrained first period)
    m_optimal : ndarray
        mitigation array from optimal (unconstrained) scenario
    
    Attributes
    ----------
    u_delay : `EZUtility` object
        utility object for delayed scenario
    u_optimal : `EZUtility` object
        utility object for optimal scenario
    cfp_m : ndarray
        constrained first period mitigation (delayed scenario)
    opt_m : ndarray
        optimal mitigation (baseline scenario)
    con_cost : float
        utility cost of constraint (opt_u - cfp_u)
    delta_c : float
        first-period consumption compensation (absolute value)
    delta_c_pct : float
        first-period compensation as % of year 0 delayed consumption
    delta_c_billions : float
        delta_c in billions of dollars
    delta_c_permanent : float
        permanent (lifetime) consumption compensation (fractional increase)
    delta_c_permanent_pct : float
        permanent compensation as percentage (e.g., 2.5 means 2.5% increase)
    year0_cons_delayed : float
        consumption at year 0 in the delayed scenario
    delta_emission_gton : float
        emission reduction foregone in gigatons CO2
    deadweight : float
        deadweight cost per ton of emission reduction foregone
    delta_u : float
        marginal utility impact of $0.01 consumption increase
    delta_u2 : float
        marginal utility impact of 0.01 mitigation increase
    
    Examples
    --------
    >>> ca = ConstraintAnalysis(u_delay, u_optimal, m_delayed, m_optimal)
    >>> print(f"Deadweight: ${ca.deadweight:.2f}/ton")
    >>> print(f"Consumption compensation: ${ca.delta_c_billions:.2f}B")
    >>> ca.save_output(prefix="test_run")
    """
    
    def __init__(self, u_delay, u_optimal, m_delayed, m_optimal,
                 compensation_years=5.0):
        self.u_delay = u_delay
        self.u_optimal = u_optimal
        self.cfp_m = m_delayed  # constrained first period mitigation
        self.opt_m = m_optimal  # optimal mitigation
        self.compensation_years = float(compensation_years)
        
        # Calculate constraint cost
        self.con_cost = self._constraint_cost()
        
        # Calculate marginal utility impacts
        self.delta_u = self._first_period_delta_udiff()
        self.delta_u2 = self._first_period_delta_udiff2()
        
        # Calculate consumption compensation (matches EZClimate's _delta_consumption)
        self.delta_c = self._delta_consumption()
        self.delta_c_5yr = self._delta_consumption_first_years(self.compensation_years)
        
        # Calculate year 0 consumption for the delayed path (for percentage calculations)
        cons_tree_delayed = self.u_delay.utility(self.cfp_m, return_trees=True)['Consumption']
        self.year0_cons_delayed = cons_tree_delayed.tree[0][0]  # Period 0, node 0
        
        # Calculate percentage increases relative to year 0 delayed consumption
        if self.delta_c is not None:
            self.delta_c_pct = (self.delta_c / self.year0_cons_delayed) * 100
        else:
            self.delta_c_pct = None
        if self.delta_c_5yr is not None:
            self.delta_c_5yr_pct = (self.delta_c_5yr / self.year0_cons_delayed) * 100
        else:
            self.delta_c_5yr_pct = None
        
        # Permanent method disabled (has issues with utility object attributes)
        self.delta_c_permanent = None
        self.delta_c_permanent_pct = None
        
        # Convert delta_c to billions of dollars:
        # delta_c is in units of "consumption" (dimensionless fraction)
        # cons_per_ton converts consumption units to $/ton CO2
        # baseline_gtco2_periods[0] is GtCO2/year at time 0
        # Result: delta_c * ($/ton) * (Gton/year) = billion $ / year
        if self.delta_c is not None:
            self.delta_c_billions = self.delta_c * self.u_delay.cost.cons_per_ton \
                                    * self.u_delay.damage.emit_baseline.baseline_gtco2_periods[0]
        else:
            self.delta_c_billions = np.nan
        if self.delta_c_5yr is not None:
            self.delta_c_5yr_billions = (
                self.delta_c_5yr * self.u_delay.cost.cons_per_ton
                * self.u_delay.damage.emit_baseline.baseline_gtco2_periods[0]
            )
            self.delta_c_5yr_total_billions = (
                self.delta_c_5yr_billions * self.compensation_years
            )
        else:
            self.delta_c_5yr_billions = np.nan
            self.delta_c_5yr_total_billions = np.nan
        self.delta_emission_gton = self.opt_m[0] * self.u_delay.damage.emit_baseline.baseline_gtco2_periods[0]
        
        # Deadweight cost per ton (matches EZClimate)
        if self.delta_c is not None and self.opt_m[0] > 0:
            self.deadweight = self.delta_c * self.u_delay.cost.cons_per_ton / self.opt_m[0]
        else:
            self.deadweight = None
        
        # Marginal benefit and cost - calculate at BOTH scenarios for complete analysis
        # At delayed (zero mitigation) scenario:
        #self.marginal_benefit_delayed = (self.delta_u2 / self.delta_u) * self.utility.cost.cons_per_ton
        #self.marginal_cost_delayed = self.utility.cost.price(0, self.cfp_m[0], 0)
        
        # At optimal scenario - need to calculate marginal utility at optimal point:
        #m_opt_plus = self.opt_m.copy()
        #m_opt_plus[0] += 0.01
        #u_opt_plus = self.utility.utility(m_opt_plus)
        #u_opt = self.utility.utility(self.opt_m)
        #if isinstance(u_opt_plus, np.ndarray):
        #    u_opt_plus = float(u_opt_plus[0])
        #if isinstance(u_opt, np.ndarray):
        #    u_opt = float(u_opt[0])
        #delta_u2_opt = u_opt_plus - u_opt
        
       # self.marginal_benefit_optimal = (delta_u2_opt / self.delta_u) * self.utility.cost.cons_per_ton
       # self.marginal_cost_optimal = self.utility.cost.price(0, self.opt_m[0], 0)
        
        # Print summary
        self._print_summary()
    
    def _constraint_cost(self):
        """Calculate utility cost of constraining first period (matches EZClimate)."""
        opt_u = self.u_optimal.utility(self.opt_m)
        cfp_u = self.u_delay.utility(self.cfp_m)
        
        if isinstance(opt_u, np.ndarray):
            opt_u = float(opt_u[0])
        if isinstance(cfp_u, np.ndarray):
            cfp_u = float(cfp_u[0])
        
        return opt_u - cfp_u
    
    def _delta_consumption(self):
        """Calculate consumption compensation using find_bec methodology (matches EZClimate)."""
        return find_consumption_equivalence(self.cfp_m, self.opt_m, self.u_delay, self.u_optimal,
                                           method='first_period', a=-150, b=150)

    def _delta_consumption_first_years(self, compensation_years):
        """Calculate compensation paid over the first N calendar years.

        This is the paper-facing DWL object for annual-grid runs because it is
        comparable to the five-year first period used elsewhere in the model.
        """
        return find_consumption_equivalence(
            self.cfp_m, self.opt_m, self.u_delay, self.u_optimal,
            method='first_years', a=-150, b=150,
            compensation_years=compensation_years
        )
    
    def _delta_consumption_permanent(self):
        """Calculate permanent (lifetime) consumption compensation."""
        return find_consumption_equivalence(self.cfp_m, self.opt_m, self.u_delay, self.u_optimal,
                                           method='permanent', a=-0.5, b=0.5)
    
    def _first_period_delta_udiff(self):
        """Marginal utility impact of $0.01 consumption increase in first period (matches EZClimate)."""
        u_given_delta_con = self.u_delay.adjusted_utility(self.cfp_m, first_period_consadj=0.01)
        cfp_u = self.u_delay.utility(self.cfp_m)
        
        if isinstance(u_given_delta_con, np.ndarray):
            u_given_delta_con = float(u_given_delta_con[0])
        if isinstance(cfp_u, np.ndarray):
            cfp_u = float(cfp_u[0])
        
        return u_given_delta_con - cfp_u
    
    def _first_period_delta_udiff2(self):
        """Marginal utility impact of 0.01 mitigation increase in first period (matches EZClimate)."""
        m = self.cfp_m.copy()
        m[0] += 0.01
        u = self.u_delay.utility(m)
        cfp_u = self.u_delay.utility(self.cfp_m)
        
        if isinstance(u, np.ndarray):
            u = float(u[0])
        if isinstance(cfp_u, np.ndarray):
            cfp_u = float(cfp_u[0])
        
        return u - cfp_u
    
    def _print_summary(self):
        print(f"Constraint Analysis Summary:")
        print(f"\nOptimization Results:")
        print(f"  Optimal first-period mitigation:   {self.opt_m[0]:.6f}")
        print(f"  Delayed first-period mitigation:   {self.cfp_m[0]:.6f}")
        print(f"  Mitigation foregone:               {self.opt_m[0] - self.cfp_m[0]:.6f}")
        
        print(f"\nUtility Metrics:")
        print(f"  Constraint cost (utility loss):    {self.con_cost:.10f}")
        print(f"  Marginal utility (consumption):    {self.delta_u:.10e}")
        print(f"  Marginal utility (mitigation):     {self.delta_u2:.10e}")
        
        print(f"\nConsumption Compensation (First-Period Method):")
        print(f"  Year 0 consumption (delayed):      {self.year0_cons_delayed:.6f}")
        if self.delta_c is not None:
            print(f"  Delta consumption (absolute):      {self.delta_c:.6f}")
            print(f"  As % of year 0 consumption:        {self.delta_c_pct:.4f}%")
            print(f"  In billions $:                     ${self.delta_c_billions:.2f}B")
        else:
            print(f"  FAILED TO CONVERGE")

        print(f"\nConsumption Compensation (First {self.compensation_years:g} Years):")
        if self.delta_c_5yr is not None:
            print(f"  Delta consumption (annual flow):   {self.delta_c_5yr:.6f}")
            print(f"  As % of year 0 consumption:        {self.delta_c_5yr_pct:.4f}%")
            print(f"  Annual-flow equivalent:            ${self.delta_c_5yr_billions:.2f}B")
            print(f"  Undiscounted window total:         ${self.delta_c_5yr_total_billions:.2f}B")
        else:
            print(f"  FAILED TO CONVERGE")
        
        print(f"\nEmissions Metrics:")
        print(f"  Delta emission (Gton CO2):         {self.delta_emission_gton:.4f} Gt")
        
        if self.deadweight is not None:
            print(f"\nDeadweight Analysis:")
            print(f"  Deadweight cost:                   ${self.deadweight:.2f} per ton CO2")

    
    def save_output(self, prefix=None):
        """Save constraint analysis results to CSV file.
        
        Parameters
        ----------
        prefix : str, optional
            prefix to add to output filename
        """
        from ..tools import write_columns_csv
        
        if prefix is not None:
            prefix += "_"
        else:
            prefix = ""
        
        # Compile results (extended from EZClimate's save_output)
        data = [
            [self.con_cost],
            [self.delta_c] if self.delta_c is not None else [np.nan],
            [self.delta_c_pct] if self.delta_c_pct is not None else [np.nan],
            [self.delta_c_billions],
            [self.delta_c_5yr] if self.delta_c_5yr is not None else [np.nan],
            [self.delta_c_5yr_pct] if self.delta_c_5yr_pct is not None else [np.nan],
            [self.delta_c_5yr_billions],
            [self.delta_c_5yr_total_billions],
            [self.delta_emission_gton],
            [self.deadweight] if self.deadweight is not None else [np.nan],
            [self.delta_u],
            #[self.marginal_benefit_delayed],
            #[self.marginal_cost_delayed],
            #[self.marginal_benefit_optimal],
            #[self.marginal_cost_optimal],
            [self.year0_cons_delayed]
        ]
        
        headers = [
            "Constraint Cost",
            "Delta Consumption (First Period)",
            "Delta Consumption (First Period) %",
            "Delta Consumption $b",
            "Delta Consumption (First 5 Years)",
            "Delta Consumption (First 5 Years) %",
            "Delta Consumption First 5 Years Annual $b",
            "Delta Consumption First 5 Years Total $b",
            "Delta Emission Gton",
            "Deadweight Cost",
            "Marginal Impact Utility",
            #"Marginal Benefit Emissions Reduction (Delayed)",
            #"Marginal Cost Emission Reduction (Delayed)",
            #"Marginal Benefit Emissions Reduction (Optimal)",
            #"Marginal Cost Emission Reduction (Optimal)",
            "Year 0 Consumption (Delayed)"
        ]
        
        filename = prefix + "constraint_output"
        write_columns_csv(data, filename, headers)
        
        print(f"Constraint analysis saved to: data/{filename}.csv")

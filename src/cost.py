"""Cost classes.

Theo Moers
tlm2160@columbia.edu
Columbia University

based originally on Adam Bauer

This code contains two classes. The first is an abstract Cost class to be used
as a template for other cost classes. The second is our cost class -- BPWCost
-- which computes the cost of carbon.
"""

import numpy as np
from abc import ABCMeta, abstractmethod

from .cal.cost_cal import cost_cal_params

class Cost(object, metaclass=ABCMeta):
    """Abstract Cost class for the EZ-Climate model.
    """

    @abstractmethod
    def cost(self):
        pass

    @abstractmethod
    def price(self):
        pass

class BPWCost(Cost):
    """Class to evaluate the cost curve for the EZ-Climate model.

    Parameters
    ----------
    tree : `TreeModel` object
        tree structure used
    emit_at_0 : float
        initial GHG emission level
    baseline_num: int
        tells what emissions baseline we're using
    tech_const: float
        rate of enxogeneous technological development
    tech_scale: float
        rate of exdogeneous technological development
    cons_at_0 : float
        initial consumption. Default $61880 bn based on US 2020 values.
    backstop_premium: float
        premium tax on co2 removal
    no_free_lunch: bool
        "No free lunch" calibration on?
    backstop_smoothing_width: float
        Width in mitigation units of the differentiable removal-premium transition.
        Zero retains the sharp paper formulation.
    backstop_smoothing_mode: str
        ``"one_sided_huber"`` preserves the sharp cost through full mitigation
        and smooths only above one. ``"symmetric_huber"`` smooths in a narrow
        band on either side of one, matching the sharp cost outside that band.

    Attributes
    ----------
    tree: `TreeModel` object
        tree structure used
    taus: list
        list of tau_0 values that were fit to IPCC AR6 WGIII data, see the
        paper for details
    powers: list
        list of power values (greek letter xi in paper) values that were fit to
        IPCC AR6 WGIII data, see the paper for details
    tau_0: float
        price of emitting all emissions from baseline
    power: float
        power law scaling of cost curve
    tech_const: float
        rate of exogeneous technological development
    tech_scale: float
        rate of endogeneous technological development
    cons_per_ton : float
        consumption per tonne of CO2 emitted
    backstop_premium: float
        premium tax on co2 removal
    no_free_lunch: bool
        "No free lunch" calibration on?
    backstop_smoothing_width: float
        Width in mitigation units of the differentiable removal-premium transition.
    backstop_smoothing_mode: str
        Shape of the differentiable removal-premium transition.
    """

    def __init__(self, tree, emit_at_0, baseline_num, tech_const, tech_scale,
                 cons_at_0, backstop_premium, no_free_lunch,
                 backstop_smoothing_width=0.0,
                 backstop_smoothing_mode="one_sided_huber",
                 mac_horizontal_shift=0.0,
                 mac_vertical_shift=0.0):

        self.tree = tree
        if no_free_lunch:
            self.taus, self.powers = cost_cal_params['no-free-lunches']
        else:
            self.taus, self.powers = cost_cal_params['main-specification']

        self.tau_0 = self.taus[baseline_num - 1]
        self.power = self.powers[baseline_num - 1]
        self.tech_const = tech_const
        self.tech_scale = tech_scale
        self.cons_per_ton = cons_at_0 / emit_at_0
        self.backstop_premium = backstop_premium
        self.backstop_smoothing_width = float(backstop_smoothing_width)
        if (
            not np.isfinite(self.backstop_smoothing_width)
            or self.backstop_smoothing_width < 0.0
        ):
            raise ValueError(
                "backstop_smoothing_width must be finite and nonnegative"
            )
        self.backstop_smoothing_mode = str(backstop_smoothing_mode).strip().lower()
        if self.backstop_smoothing_mode not in (
            "one_sided_huber", "symmetric_huber",
        ):
            raise ValueError(
                "backstop_smoothing_mode must be 'one_sided_huber' or "
                "'symmetric_huber'"
            )
        self.mac_horizontal_shift = float(mac_horizontal_shift)
        self.mac_vertical_shift = float(mac_vertical_shift)
        if not np.isfinite(self.mac_horizontal_shift) or self.mac_horizontal_shift < 0.0:
            raise ValueError("mac_horizontal_shift must be finite and nonnegative")
        if not np.isfinite(self.mac_vertical_shift) or self.mac_vertical_shift < 0.0:
            raise ValueError("mac_vertical_shift must be finite and nonnegative")
        self.anchor_year = 2030


    @property
    def cost_formulation(self):
        """Stable identifier for the removal-cost formulation in use."""

        if self.backstop_smoothing_width <= 0.0:
            return "additive_removal_premium_v1"
        if self.backstop_smoothing_mode == "symmetric_huber":
            return "additive_removal_premium_symmetric_huber_v1"
        if self.backstop_smoothing_mode == "one_sided_huber":
            return "additive_removal_premium_huber_v1"
        raise RuntimeError("unrecognized removal-premium smoothing mode")

    def _removal_premium_integral(self, mitigation):
        """Integrated removal premium, optionally smoothed around ``m == 1``.

        With ``one_sided_huber`` and a positive width ``delta``, this is the
        one-sided Huberized positive part: it is exactly zero through full
        mitigation, ramps quadratically from one to ``1 + delta``, and then
        has the intended full marginal premium. The approximation is convex
        and continuously differentiable. Its maximum level difference from
        the sharp premium is ``backstop_premium * delta / 2``.

        With ``symmetric_huber``, ``delta`` is the half-width of a narrow
        transition band ``[1 - delta, 1 + delta]``. It matches the original
        sharp premium exactly outside that band, including for all sufficiently
        high removal. Its largest level difference is
        ``backstop_premium * delta / 4`` at full mitigation. This option makes
        a deliberately tiny adjustment just below full mitigation in exchange
        for a lower-curvature transition and no permanent post-transition
        subsidy.
        """

        x = np.maximum(np.asarray(mitigation, dtype=float), 0.0) - 1.0
        delta = self.backstop_smoothing_width
        if delta == 0.0:
            return self.backstop_premium * np.maximum(x, 0.0)
        if self.backstop_smoothing_mode == "symmetric_huber":
            transition = (x + delta) ** 2 / (4.0 * delta)
            return self.backstop_premium * np.where(
                x <= -delta,
                0.0,
                np.where(x < delta, transition, x),
            )
        transition = x * x / (2.0 * delta)
        tail = x - 0.5 * delta
        return self.backstop_premium * np.where(
            x <= 0.0,
            0.0,
            np.where(x < delta, transition, tail),
        )

    def _removal_premium_marginal(self, mitigation):
        """Marginal removal premium corresponding to the chosen formulation."""

        mitigation = np.asarray(mitigation, dtype=float)
        x = np.maximum(mitigation, 0.0) - 1.0
        delta = self.backstop_smoothing_width
        if delta == 0.0:
            return self.backstop_premium * (x > 0.0)
        if self.backstop_smoothing_mode == "symmetric_huber":
            return self.backstop_premium * np.where(
                x <= -delta,
                0.0,
                np.where(x < delta, (x + delta) / (2.0 * delta), 1.0),
            )
        return self.backstop_premium * np.where(
            x <= 0.0,
            0.0,
            np.where(x < delta, x / delta, 1.0),
        )

    def _raw_integrated_cost(self, mitigation):
        """Return the pre-technology integrated mitigation cost.

        Mitigation below zero is clipped because the model does not credit
        negative mitigation. Carbon removal (mitigation above one) pays an additive per-tonne
        premium only on the amount removed. By default this retains the sharp paper kink at
        one; a positive ``backstop_smoothing_width`` replaces it with a narrow
        differentiable transition selected by ``backstop_smoothing_mode``.
        """

        mitigation = np.asarray(mitigation, dtype=float)
        m0 = np.maximum(mitigation, 0.0)
        # The shifted calibrated component is the integral from zero to m of
        # MAC_base(s + h).  Subtracting its value at h keeps total cost zero
        # at zero mitigation while allowing a positive right derivative there.
        shifted_m = m0 + self.mac_horizontal_shift
        calibrated_cost = self.tau_0 * (
            (np.expm1(self.power * shifted_m) -
             np.expm1(self.power * self.mac_horizontal_shift)) / self.power
            - m0
        )
        calibrated_cost = calibrated_cost + self.mac_vertical_shift * m0
        removal_cost = self._removal_premium_integral(m0)
        return calibrated_cost + removal_cost

    def _raw_marginal_cost(self, mitigation):
        """Return the pre-technology marginal mitigation cost.

        At the clipping boundary the reported derivative is zero. For the
        default sharp premium the derivative is one-sided at mitigation one;
        for positive smoothing widths it is the ordinary derivative there.
        """

        mitigation = np.asarray(mitigation, dtype=float)
        m0 = np.maximum(mitigation, 0.0)
        shifted_m = m0 + self.mac_horizontal_shift
        marginal_cost = (
            self.tau_0 * np.expm1(self.power * shifted_m)
            + self.mac_vertical_shift
        )
        marginal_cost = marginal_cost + self._removal_premium_marginal(mitigation)
        # At m=0 this reports the economically relevant right derivative.
        # With zero shifts this remains exactly zero, preserving the legacy
        # curve and all existing production specifications.
        return np.where(mitigation >= 0.0, marginal_cost, 0.0)

    def cost(self, period, mitigation, ave_mitigation):
        """Calculates the mitigation cost for the period. For details about the
        cost function see our paper.

        Parameters
        ----------
        period : int
            period in tree for which mitigation cost is calculated
        mitigation : ndarray
            current mitigation values for period
        ave_mitigation : ndarray
            average mitigation up to this period for all nodes in the period

        Returns
        -------
        ndarray :
            cost
        """

        years_since_base_year = self.tree.decision_times[period]
        abs_year = self.tree.base_year + years_since_base_year
        discount_exponent = abs_year - self.anchor_year
        tech_term = (1.0 - ((self.tech_const + self.tech_scale*ave_mitigation)
                            / 100.0))**(discount_exponent)
        mitigation_cost = self._raw_integrated_cost(mitigation)
        c = (mitigation_cost * tech_term) / self.cons_per_ton
        return c

    def price(self, years, mitigation, ave_mitigation):
        """Inverse of the cost function. Gives emissions price for any given
        degree of mitigation, average_mitigation, and horizon.

        Parameters
        ----------
        years : int y
            years of technological change so far
        mitigation : float
            mitigation value in node
        ave_mitigation : float
            average mitigation up to this period

        Returns
        -------
        float
            the price.
        """

        abs_year = self.tree.base_year + years
        discount_exponent = abs_year - self.anchor_year
        tech_term = (1.0 - ((self.tech_const + self.tech_scale*ave_mitigation)
                            / 100.0))**(discount_exponent)
        return tech_term * self._raw_marginal_cost(mitigation)

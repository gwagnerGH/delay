"""Emission baseline class.

Theo Moers
tlm2160@columbia.edu
Columbia University

based originally on Adam Bauer

Contains the abstract class BusinessAsUsual as well as its subclass,
BPWBusinessAsUsual, which is implemented in CAP6.
"""

import numpy as np
from abc import ABC, abstractmethod

from .config import DEFAULT_BASE_YEAR, get_base_year_reference
from .tools import get_integral_var_ub, import_csv

class EmissionBaseline(ABC):
    """Abstract emissions baseline class for CAP6.

    Attributes
    ----------
    times: nd array
        times (in years) that the emissions are taking place

    baseline_gtco2: nd array
        baseline emissions in GtCO2/year

    baseline_ppm: nd array
        baseline emissions in ppm CO2/year

    baseline_cumemit: nd array
        cumulative emissions of baseline in 1000 GtCO2

    baseline_gtco2_periods: nd array
        baseline emissions in GtCO2/year evaluated at tree.decision_times

    baseline_ppm_periods: nd array
        baseline emissions in ppm CO2/year evaluated at tree.decision_times

    baseline_cumemit_periods: ndarray
        cumulative emissions in 1000 GtCO2 evaluated at tree.decision_times

    dec_times_ind: ndarray
        indexes at which decision times are within self.times

    DELTA_T: int
        time difference between emission data points

    CUMEMIT_BASE_YEAR: float
        cumulative emissions as of the base year reference (in 1000 GtCO2)

    GTCO2_TO_PPM: float
        conversion factor which takes GtCO2 and results in ppm CO2

    Methods
    -------
    baseline_emission_setup:
        sets up various class attributes

    get_mitigated_baseline:
        for a vector or value of mitigation, return mitigated baseline
    """

    def __init__(self):
        self.times = None
        self.baseline_gtco2 = None
        self.baseline_ppm = None
        self.baseline_cumemit = None
        self.baseline_gtco2_periods = None
        self.baseline_ppm_periods = None
        self.baseline_cumemit_periods = None
        self.dec_times_ind = None
        self.DELTA_T = None
        self.base_year = DEFAULT_BASE_YEAR
        self.base_year_reference = get_base_year_reference(self.base_year)
        self.CUMEMIT_BASE_YEAR = self.base_year_reference.cumemit_value
        self.GTCO2_TO_PPM = 7.8**(-1) # takes GtCO2 -> ppm CO2

    @abstractmethod
    def baseline_emission_setup(self):
        pass

    @abstractmethod
    def get_mitigated_baseline(self):
        pass

class BPWEmissionBaseline(EmissionBaseline):
    """Baseline CO2 emission pathways.

    This class contains information on the various emissions pathways
    used by CAP6. It features baseline flexibility, allowing the code
    to change which SSP baseline we use with a parameter.

    Parameters
    ----------
    tree: `TreeModel` object
        tree structure of the model

    baseline_num: int
        baseline number, such that:
            1: SSP1
            2: SSP2
            3: SSP3
            4: SSP4
            5: SSP5
        an error will be raised if any number other that
        1-5 are given.

    Attributes
    ----------
    times: nd array
        times (in years) that the emissions are taking place

    baseline_gtco2: nd array
        baseline emissions in GtCO2/year

    baseline_ppm: nd array
        baseline emissions in ppm CO2/year

    baseline_cumemit: nd array
        cumulative emissions of baseline in 1000 GtCO2

    baseline_gtco2_periods: nd array
        baseline emissions in GtCO2/year evaluated at tree.decision_times

    baseline_ppm_periods: nd array
        baseline emissions in ppm CO2/year evaluated at tree.decision_times

    baseline_cumemit_periods: ndarray
        cumulative emissions in 1000 GtCO2 evaluated at tree.decision_times

    dec_times_ind: nd array
        indexes at which decision times are within self.times

    DELTA_T: int
        time difference between emission data points

    CUMEMIT_BASE_YEAR: float
        cumulative emissions as of the base year reference (in 1000 GtCO2)

    GTCO2_TO_PPM: float
        conversion factor which takes GtCO2 and results in ppm CO2

    Methods
    -------
    baseline_emission_setup:
        sets up various class attributes

    _make_extension:
        per Meinshausen et al., 2020 (https://doi.org/10.5194/gmd-13-3571-2020)
        the extensions of the various SSPs are just (basically, going off of
        Figure 2) linearly connect the final emissione value in 2100 to zero in
        2250. this method carries out this prescription and makes the final
        emissions time series.

    get_mitigated_baseline:
        for a given node and mitigation vector/value, make the mitigated version
        of a given baseline.
    """

    def __init__(self, tree, baseline_num, emissions_time_step=None,
                 baseline_source='ssp'):
        self.tree = tree
        self.baseline_num = baseline_num
        self.emissions_time_step = emissions_time_step
        self.baseline_source = baseline_source
        self.times = None
        self.baseline_gtco2 = None
        self.baseline_ppm = None
        self.baseline_cumemit = None
        self.baseline_gtco2_periods = None
        self.baseline_ppm_periods = None
        self.baseline_cumemit_periods = None
        self.dec_times_ind = None
        self.DELTA_T = None
        self.base_year = self.tree.base_year
        self.base_year_reference = get_base_year_reference(self.base_year)
        if self.base_year_reference.cumemit_value is None:
            raise ValueError(f"Missing cumulative emissions reference for base year {self.base_year}.")
        self.CUMEMIT_BASE_YEAR = self.base_year_reference.cumemit_value
        self.GTCO2_TO_PPM = 7.8**(-1) # takes GtCO2 -> ppm CO2

    def baseline_emission_setup(self):
        """Make baselines & evaluate them at decision node times.

        CAP6 relies on a designated emissions pathway, from which
        we mitigate to limit global warming. In this function, we import every
        SSP baseline from "SSP_baselines.csv".
        (The values in this file were taken from
        https://tntcat.iiasa.ac.at/SspDb/dsd?Action=htmlpage&page=about)

        We then extend them to the year 2400 using the prescription of
        Meinshausen et al., 2020 by basically drawing a straight line from the
        emissions values in 2100 to zero by 2250. (Seriously, look at their
        Figure 2.)

        We then evaluate each of these baselines at the decision period times.
        These will be used throughout CAP6.

        *Raises ValueError if baseline_num is outside acceptable range.
        """

        time, baseline_gtco2_2100 = self._baseline_input_series()

        # make extended baseline in gtco2
        self._make_extension(time, baseline_gtco2_2100,
                             self.tree.calendar_years[-1])

        # make baseline in ppm CO2
        self.baseline_ppm = self.baseline_gtco2 * self.GTCO2_TO_PPM

        # calculate the cumulative emissions in 1000 GtCO2 of the baseline.
        # Note that the get_integral_var_ub call is multiplied by 10**(-3), as
        # the baseline that is being integrated is in GtCO2, so we multiply the
        # reuslt by 10**(-3) to transform the units.
        self.baseline_cumemit = self.CUMEMIT_BASE_YEAR + \
                                get_integral_var_ub(self.baseline_gtco2,
                                                    self.times, self.times) * 10**(-3)

        # Evaluate the baselines at decision years robustly for any uniform CSV grid
        # Build absolute calendar years for decision times.
        self.decision_years = np.asarray(self.tree.calendar_years, dtype=int)

        missing_decision_years = [
            year for year in self.decision_years if year not in set(self.times)
        ]
        if missing_decision_years:
            raise ValueError(
                "Decision years must exist exactly on the emissions time grid. "
                f"Missing years: {missing_decision_years}. "
                "Use emissions_time_step=1 for annual-grid delay runs."
            )

        # Map decision years to exact indices on self.times.
        self.dec_times_ind = np.searchsorted(self.times, self.decision_years, side='left')

        # Sanity checks: bounds and increasing indices
        if not (self.times[0] <= self.decision_years[0] and self.decision_years[-1] <= self.times[-1]):
            raise ValueError("Decision years fall outside baseline time grid. Check extension_year or CSV years.")
        assert np.all(np.diff(self.dec_times_ind) > 0), \
            "Decision times indices must be strictly increasing"

        # Evaluate baselines at exact decision years via interpolation
        self.baseline_gtco2_periods = np.interp(self.decision_years, self.times, self.baseline_gtco2)
        self.baseline_ppm_periods = np.interp(self.decision_years, self.times, self.baseline_ppm)
        self.baseline_cumemit_periods = np.interp(self.decision_years, self.times, self.baseline_cumemit)
        
    def _decision_interval_samples(self, period, baseline):
        """Return a discontinuity-aware emissions path through ``period``.

        A decision made at the start of interval ``i`` governs the complete
        interval from decision time ``i`` through decision time ``i + 1``.
        Adjacent intervals therefore contain two samples at their shared
        boundary: the left-limit value under the old control and the
        right-limit value under the new control.  The duplicate has zero width
        in trapezoidal integration, so it represents the policy jump without
        spuriously blending controls across an entire emissions-grid step.

        Returns
        -------
        trunc_times : ndarray
            Calendar-year samples, including duplicated internal boundaries.
        raw_baseline : ndarray
            Unmitigated emissions evaluated at ``trunc_times``.
        interval_indices : ndarray
            Decision-interval index controlling every returned sample.
        """

        period = int(period)
        if period < 0 or period > self.tree.num_periods:
            raise ValueError("period is outside the decision tree")
        if baseline not in ("ppm", "gtco2"):
            raise ValueError("baseline must be 'ppm' or 'gtco2'")

        raw_source = (
            self.baseline_ppm if baseline == "ppm" else self.baseline_gtco2
        )
        if period == 0:
            return (
                np.asarray(self.times[:1]),
                np.zeros(1, dtype=float),
                np.full(1, -1, dtype=int),
            )

        time_segments = []
        baseline_segments = []
        interval_segments = []
        for interval in range(period):
            low = int(self.dec_times_ind[interval])
            high = int(self.dec_times_ind[interval + 1]) + 1
            time_segments.append(np.asarray(self.times[low:high]))
            baseline_segments.append(
                np.asarray(raw_source[low:high], dtype=float)
            )
            interval_segments.append(
                np.full(high - low, interval, dtype=int)
            )
        return (
            np.concatenate(time_segments),
            np.concatenate(baseline_segments),
            np.concatenate(interval_segments),
        )

    @staticmethod
    def _cumulative_trapezoid_preserve_jumps(integrand, times):
        """Cumulative trapezoid that retains duplicated jump boundaries."""

        integrand = np.asarray(integrand, dtype=float)
        times = np.asarray(times, dtype=float)
        cumulative = np.zeros_like(integrand, dtype=float)
        if len(integrand) > 1:
            increments = 0.5 * np.diff(times) * (
                integrand[:-1] + integrand[1:]
            )
            cumulative[1:] = np.cumsum(increments)
        return cumulative

    def _ssp_baseline_input_series(self):
        """Return the configured SSP baseline through 2100."""

        time, _, run_data = import_csv("SSP_baselines", delimiter=',',
                                       header=True, indices=1)

        time = np.array(time, dtype=int)

        if self.baseline_num > 5 or self.baseline_num <= 0:
            raise ValueError("Invalid baseline_num parameter; must be a value "
                             "between 1 and 5.")
        baseline_gtco2_2100 = run_data[self.baseline_num - 1]
        return time, baseline_gtco2_2100

    def _historical_splice_input_series(self):
        """Return observed 1975-2025 emissions spliced to the SSP baseline.

        The historical segment is total anthropogenic CO2 emissions, defined as
        fossil/industrial plus land-use-change emissions from Global Carbon
        Budget 2025. The 2025 point is the GCB 2025 projection. From 2030 on,
        the configured SSP baseline is used unchanged.
        """

        hist_header, hist_data = import_csv(
            "gcb2025_historical_total_co2", delimiter=',', header=True
        )
        if hist_header != ['year', 'total_gtco2']:
            raise ValueError(
                "gcb2025_historical_total_co2.csv must have columns "
                "year,total_gtco2."
            )

        hist_years = hist_data[:, 0].astype(int)
        hist_gtco2 = hist_data[:, 1].astype(float)
        if hist_years[0] != 1975 or hist_years[-1] != 2025:
            raise ValueError(
                "Historical CO2 baseline must span 1975 through 2025."
            )
        if not np.all(np.diff(hist_years) == 1):
            raise ValueError("Historical CO2 baseline must be annual.")

        ssp_years, ssp_gtco2 = self._ssp_baseline_input_series()
        post_2025 = ssp_years > hist_years[-1]
        if not post_2025.any():
            raise ValueError("SSP baseline must include years after 2025.")

        return (
            np.hstack((hist_years, ssp_years[post_2025])),
            np.hstack((hist_gtco2, ssp_gtco2[post_2025])),
        )

    def _baseline_input_series(self):
        source = self.baseline_source.lower()
        if source in ('ssp', 'default'):
            return self._ssp_baseline_input_series()
        if source in ('historical_splice', 'historical_splice_ssp'):
            return self._historical_splice_input_series()
        raise ValueError(
            "baseline_source must be 'ssp' or 'historical_splice' "
            f"(got {self.baseline_source!r})."
        )

    def _make_extension(self, time, baseline_gtco2_2100, extension_year):
        """Make baseline extensions from 2100 -> extension_year.

        CAP6 requires emissions values through the last decision time.
        The SSP database only provides values through 2100, but in Meinshausen
        et al., 2020 extensions are provided. We make their approximate
        versions here by linearly interpolating the final emission value to
        zero in 2250.

        Parameters
        ----------
        time: nd array
            array of time values for which emissions time series are evaluated

        baseline_gtco2_2100: nd array
            array of emissions values in GtCO2 at times in `time` argument

        extension_year: int
            year for the emissions baseline to be extended to
        """

        native_delta_t = time[1] - time[0]
        self.DELTA_T = (
            native_delta_t if self.emissions_time_step is None
            else self.emissions_time_step
        )
        if self.DELTA_T <= 0:
            raise ValueError("emissions_time_step must be positive.")

        # append desired value to time and baselines
        time_appended = np.hstack((time, np.array([2250])))
        baseline_gtco2_appended = np.hstack((baseline_gtco2_2100, np.array([0])))

        # now time is [2010, ..., 2100, 2250] and the emission time series are
        # [val_2010, ..., val_2100, val_2250]. we can now interpolate by making
        # a new set of times to interpolate to.
        time_2100_ext = np.arange(2100 + self.DELTA_T, int(extension_year) +
                                   self.DELTA_T, self.DELTA_T)

        # make new full times that new emissions pathways will be evaluated at
        if self.emissions_time_step is None:
            full_times = np.hstack((time, time_2100_ext))
        else:
            full_times = np.arange(
                int(time[0]), int(extension_year) + self.DELTA_T, self.DELTA_T
            )
        self.times = full_times[full_times >= self.base_year]

        # now interpolate and return extended time emissions time series
        self.baseline_gtco2 = np.interp(self.times, time_appended,
                                          baseline_gtco2_appended)

    def get_mitigated_baseline(self, m, node=None, baseline="ppm",
                               is_last=False):
        """Calculate the mitigated version of a given baseline.

        In CAP6, we often need to apply a mitigation to the baseline
        emission pathway. This function creates this "mitigated" baseline by
        multiplying every emissions value between two decision nodes by a
        mitigation value supplied by m. If m is a constant, then we make an
        array of constant mitigations.

        Parameters
        ----------
        m: nd array
            mitigation values; can either be a constant value (with node=None)
            or an array with shape equal to (tree.num_decision_nodes,)

        node: int
            node number. method makes an emission time series *up until and
            including* the time of the node. (None by default.)

        baseline: string
            tells code which baseline to create a mitigated version of, such
            that:
                "ppm": baseline in ppm (default)
                "gtco2": baseline in gtco2
                "cumemit": cumulative emissions baseline (in 1000 gtco2)

        is_last: bool
            is this the final period?

        Returns
        -------
        mitigated_baseline: nd array
            mitigated emissions baseline

        trunc_times: nd array
            times at which the mitigated baseline is evaluated
            * only if node is not None
        """


        # make dictionary of baselines; needed for later
        baseline_dict = {'ppm': self.baseline_ppm, 'gtco2':
                         self.baseline_gtco2, 'cumemit': self.baseline_cumemit}

        # if not None is given as node, then create a truncated, mitigated
        # emissions baseline. 

        # NOTE: the baseline returned here does not include the action at the
        # node in its baseline. this is because the emissions prior to a node
        # should not depend on the action at that node.

        if node is not None:
            # we cannot just take m(t) * C(t) to make a mitigated version of
            # the cumulative emissions baseline because the integral is time
            # dependent. Therefore, we make a mitigated baseline in terms of
            # gtco2 and calculate the integral after we've applied the
            # mitigation
            if baseline == "cumemit":
                tmp_baseline = 'gtco2'
            else:
                tmp_baseline = baseline

            # get period we're in (i.e., how many decisions have we made?)
            period = self.tree.get_period(node)

            if is_last:
                period += 1

            path = np.asarray(self.tree.get_path(node), dtype=int)
            trunc_times, raw_baseline, interval_indices = \
                self._decision_interval_samples(period, tmp_baseline)
            if period == 0:
                mitigated_baseline = np.zeros_like(raw_baseline)
            else:
                controlling_nodes = path[interval_indices]
                mitigated_baseline = raw_baseline * (
                    1.0 - np.asarray(m, dtype=float)[controlling_nodes]
                )

            if baseline == 'cumemit':
                mitigated_baseline = self.CUMEMIT_BASE_YEAR + 10**(-3) * \
                    self._cumulative_trapezoid_preserve_jumps(
                        mitigated_baseline, trunc_times
                    )

            return mitigated_baseline, trunc_times

        # otherwise we're after just a scaled up/down version of the baseline,
        # so just make that without the above headache
        else:
            try:
                if baseline == "cumemit":
                    flow = self.baseline_gtco2 * (1 - m)
                    mitigated_baseline = self.CUMEMIT_BASE_YEAR + 1e-3 * get_integral_var_ub(
                        flow, self.times, self.times
                    )
                else:
                    mitigated_baseline = baseline_dict[baseline] * (1 - m)
                return mitigated_baseline

            except KeyError:
                print("Invalid baseline. Only 'ppm', 'gtco2, and \
                      'cumemit' are implemented.")
                raise

"""Centralized configuration for CAP6 base-year dependent parameters.

This module consolidates the default decision schedule for the CAP6 tree and
stores reference values that depend on the model base year (for example,
starting cumulative emissions and atmospheric CO₂ concentration).  The goal is
for all modules to consult this single source so that changing the base year or
adding new reference data does not require hunting for hard-coded constants
scattered across the code base.

Values that require external research are set to ``None``.  When running with a
base year whose reference values have not yet been filled in, downstream code
will raise a helpful error explaining what needs to be provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


DEFAULT_BASE_YEAR: int = 2025
DEFAULT_DECISION_TIMES = [0, 5, 35, 75, 125, 175, 225]
DEFAULT_CALENDAR_YEARS = [DEFAULT_BASE_YEAR + offset for offset in DEFAULT_DECISION_TIMES]


@dataclass(frozen=True)
class BaseYearReference:

    cumemit_reference_year: int
    cumemit_value: Optional[float]
    co2_reference_year: int
    co2_concentration: Optional[float]


BASE_YEAR_REFERENCES: Dict[int, BaseYearReference] = {
    1975: BaseYearReference(
        cumemit_reference_year=1974,
        # 1000 GtCO2, cumulative fossil/industrial plus land-use-change CO2
        # through 1974 from Global Carbon Budget 2025 historical budget.
        cumemit_value=1.096959078076,
        co2_reference_year=1975,
        # ppm, NOAA/Scripps Mauna Loa annual mean used because the NOAA global
        # annual series begins in 1979.
        co2_concentration=331.13,
    ),
    2020: BaseYearReference(
        cumemit_reference_year=2019,
        cumemit_value=2.39,  # 1000 GtCO2
        co2_reference_year=2020,
        co2_concentration=420.87,  # ppm (co2.earth daily average)
    ),
    2025: BaseYearReference(
        cumemit_reference_year=2024,
        cumemit_value=2.730,  # 1000 GtCO2 cumulative anthropogenic CO2 through 2024 (Global Carbon Budget 2025 / Global Carbon Project)
        co2_reference_year=2025,
        co2_concentration=425.6,  # ppm global atmospheric CO2 concentration projected for 2025 (Global Carbon Budget 2025 / Global Carbon Project)
    ),
}


def get_base_year_reference(base_year: int) -> BaseYearReference:
    return BASE_YEAR_REFERENCES[base_year]

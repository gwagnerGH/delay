"""Shared economic baseline for revised EZDelay and cluster experiments.

Technology coefficients are percentages in the inherited abatement-learning rule;
growth and pure time preference are annual fractions. This is abatement learning,
not endogenous aggregate productivity growth.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RevisedBaseline:
    ra: float = 10.0
    eis: float = 1.5
    prtp: float = 0.005
    growth: float = 0.015
    terminal_growth_mode: str = "lognormal"
    terminal_growth: float = 0.015  # deterministic alternative, annual net growth
    terminal_log_mean: float = 0.015  # annual E[log(C_next/C)]
    terminal_log_sigma: float = 0.02
    tech_const: float = 1.5
    tech_scale: float = 0.0
    baseline_num: int = 2
    backstop: str = "s_curve"
    backstop_curvature: float = 1.0
    gamma: float = 0.0
    no_free_lunch: bool = True
    flat_emissions_after_2100: bool = True
    five_year_cost_accounting: bool = True
    pchip_damages: bool = True
    solver: str = "adjoint_lbfgsb"


BASELINE = RevisedBaseline()
PREFERENCE_RA_LEVELS = (7.0, 10.0, 15.0, 45.0)
PREFERENCE_EIS_LEVELS = (0.9, 1.5, 2.2, 3.0)
PREFERENCE_PRTP_LEVELS = (0.002, 0.005, 0.024)
ENDOGENOUS_LEARNING_ROBUSTNESS = 1.5

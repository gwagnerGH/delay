"""Parameter prior configuration for ensemble and robustness runs."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .revised_baseline import BASELINE


@dataclass(frozen=True)
class ParameterPrior:
    """One truncated Gaussian prior with a configured mode/location."""

    name: str
    lower: float
    upper: float
    mean_value: float
    source_note: str = ""
    std_value: float | None = None

    @property
    def mean(self):
        return self.mean_value

    @property
    def std(self):
        return ((self.upper - self.lower) / 4.0
                if self.std_value is None else self.std_value)


GAUSSIAN_PRIOR_SET_NAME = "revised_RA10_EIS15_logterminal_curvature_RAmax15_v5"

RESEARCH_RUN_PARAMETER_COLUMNS = {
    "RA": "RA",
    "EIS": "EIS",
    "tech_chg": "tech_change",
    "tech_scale": "tech_scale",
    "PRTP": "preference",
    "bs_premium": "bs_premium",
    "growth": "growth",
}


def _research_runs_path():
    return Path(__file__).resolve().parents[2] / "data" / "research_runs.csv"


def _load_research_run0():
    with _research_runs_path().open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {_research_runs_path()}")
    return rows[0]


RUN0_RESEARCH_RUN = _load_research_run0()


GAUSSIAN_PARAMETER_PRIORS = (
    ParameterPrior(
        "RA",
        1.0,
        15.0,
        BASELINE.ra,
        "RA 10 revised baseline; RA 7, RA 15, and RA 45 are deterministic sensitivities.",
        std_value=3.75,
    ),
    ParameterPrior(
        "EIS",
        0.01,
        3.0,
        BASELINE.eis,
        "EIS 1.5 revised baseline; EIS 0.9 and 0.1 are deterministic low-EIS sensitivities.",
        std_value=0.5625,
    ),
    ParameterPrior(
        "tech_chg",
        0.0,
        3.0,
        BASELINE.tech_const,
        "Revised baseline exogenous abatement technology rate.",
    ),
    ParameterPrior(
        "tech_scale",
        0.0,
        3.0,
        BASELINE.tech_scale,
        "Zero-learning revised baseline; positive values represent learning robustness draws.",
    ),
    ParameterPrior(
        "PRTP",
        0.004,
        0.020,
        BASELINE.prtp,
        "Revised baseline annual pure rate of time preference; joint preference draws are conditioned on finite terminal continuation.",
        std_value=0.00225,
    ),
    ParameterPrior(
        "bs_premium",
        5000.0,
        20000.0,
        float(RUN0_RESEARCH_RUN["bs_premium"]),
        "Legacy compatibility slot; ignored by revised S-curve model. Use cubic gamma explicitly.",
    ),
    ParameterPrior(
        "growth",
        0.005,
        0.025,
        BASELINE.growth,
        "Revised baseline annual exogenous consumption growth.",
        std_value=0.00375,
    ),
)

# Revised ensemble only: keep the seven legacy-compatible columns intact.
REVISED_ENSEMBLE_CURVATURE_PRIOR = ParameterPrior(
    "backstop_curvature", 0.5, 2.0, BASELINE.backstop_curvature,
    "Smooth removal-curve curvature; 1 reproduces the default S-curve.",
    std_value=0.375,
)

PARAMETER_PRIOR_NAMES = [prior.name for prior in GAUSSIAN_PARAMETER_PRIORS]
PARAMETER_PRIOR_INDEX = {
    name: index for index, name in enumerate(PARAMETER_PRIOR_NAMES)
}
PARAMETER_PRIOR_LOWER_BOUNDS = np.asarray(
    [prior.lower for prior in GAUSSIAN_PARAMETER_PRIORS], dtype=float
)
PARAMETER_PRIOR_UPPER_BOUNDS = np.asarray(
    [prior.upper for prior in GAUSSIAN_PARAMETER_PRIORS], dtype=float
)
PARAMETER_PRIOR_MEANS = np.asarray(
    [prior.mean for prior in GAUSSIAN_PARAMETER_PRIORS], dtype=float
)
PARAMETER_PRIOR_STDS = np.asarray(
    [prior.std for prior in GAUSSIAN_PARAMETER_PRIORS], dtype=float
)
PARAMETER_PRIOR_DIMS = len(GAUSSIAN_PARAMETER_PRIORS)
RUN0_PARAMETER_VALUES = PARAMETER_PRIOR_MEANS.copy()
RUN0_FIXED_PARAMETERS = {
    "dam_func": int(RUN0_RESEARCH_RUN["dam_func"]),
    "baseline_num": int(RUN0_RESEARCH_RUN["baseline_num"]),
    "tip_on": int(RUN0_RESEARCH_RUN["tip_on"]),
    "d_unc": int(RUN0_RESEARCH_RUN["d_unc"]),
    "t_unc": int(RUN0_RESEARCH_RUN["t_unc"]),
    "no_free_lunch": BASELINE.no_free_lunch,
}


def parameter_prior_table():
    """Return prior metadata as plain dictionaries for notebooks and tables."""

    return [
        {
            "name": prior.name,
            "lower": prior.lower,
            "upper": prior.upper,
            "mean": prior.mean,
            "std": prior.std,
            "source_note": prior.source_note,
        }
        for prior in GAUSSIAN_PARAMETER_PRIORS + (REVISED_ENSEMBLE_CURVATURE_PRIOR,)
    ]

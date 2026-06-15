"""Parameter prior configuration for ensemble and robustness runs."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ParameterPrior:
    """One truncated Gaussian prior with a configured mode/location."""

    name: str
    lower: float
    upper: float
    mean_value: float
    source_note: str = ""

    @property
    def mean(self):
        return self.mean_value

    @property
    def std(self):
        return (self.upper - self.lower) / 4.0


GAUSSIAN_PRIOR_SET_NAME = "BRprefs_run0mode"

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
        3.0,
        15.0,
        float(RUN0_RESEARCH_RUN["RA"]),
        "Risk aversion support used in ensemble runs; mode is data/research_runs.csv run 0.",
    ),
    ParameterPrior(
        "EIS",
        0.55,
        1.86,
        float(RUN0_RESEARCH_RUN["EIS"]),
        "Bauer & Rudebusch term-structure interpolation support; mode is data/research_runs.csv run 0.",
    ),
    ParameterPrior(
        "tech_chg",
        0.0,
        3.0,
        float(RUN0_RESEARCH_RUN["tech_change"]),
        "Exogenous technological progress support; mode is data/research_runs.csv run 0.",
    ),
    ParameterPrior(
        "tech_scale",
        0.0,
        3.0,
        float(RUN0_RESEARCH_RUN["tech_scale"]),
        "Endogenous learning support; mode is data/research_runs.csv run 0.",
    ),
    ParameterPrior(
        "PRTP",
        0.001,
        0.024,
        float(RUN0_RESEARCH_RUN["preference"]),
        "Bauer & Rudebusch term-structure interpolation support; mode is data/research_runs.csv run 0.",
    ),
    ParameterPrior(
        "bs_premium",
        5000.0,
        20000.0,
        float(RUN0_RESEARCH_RUN["bs_premium"]),
        "Backstop premium support; mode is data/research_runs.csv run 0.",
    ),
    ParameterPrior(
        "growth",
        0.010,
        0.025,
        float(RUN0_RESEARCH_RUN["growth"]),
        "Consumption growth support; mode is data/research_runs.csv run 0.",
    ),
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
    "no_free_lunch": bool(int(RUN0_RESEARCH_RUN["no_free_lunch"])),
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
        for prior in GAUSSIAN_PARAMETER_PRIORS
    ]

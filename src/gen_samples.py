"""Generate samples for CAP6 ensemble model runs.

Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
8.19.2022

This code contains functions which make text files that contain sampled model
parameter values.
"""

import numpy as np

from scipy.stats import truncnorm


def truncnorm_locs_at_modes(lbs, ubs, means):
    """Return raw normal locs whose truncated Gaussian modes equal means."""
    lbs = np.asarray(lbs, dtype=float)
    ubs = np.asarray(ubs, dtype=float)
    means = np.asarray(means, dtype=float)

    if len(lbs) != len(ubs) or len(lbs) != len(means):
        raise ValueError("lbs, ubs, and means must have the same length")
    if np.any(ubs <= lbs):
        raise ValueError("Each upper bound must be greater than its lower bound")
    if np.any(means <= lbs) or np.any(means >= ubs):
        raise ValueError("Each Gaussian mode must lie strictly inside its bounds")

    return means.copy()


def generate_gaussian_samples(N_RUNS, DIMS, lbs, ubs, means=None, stds=None,
                              save_file=True,
                              filename="/data/BPW_gaussian_samples.csv",
                              random_seed=None):
    """Generate bounded Gaussian samples of model parameters.

    The configured lower and upper bounds define the support of a truncated
    normal distribution. The ``means`` argument is interpreted as the raw
    normal ``loc`` and therefore the peak/mode of the truncated density.
    By default, each parameter mean is the midpoint of its bounds and the
    standard deviation is one quarter of the bound width, so the interval
    corresponds to roughly +/- two standard deviations before truncation.

    Parameters
    ----------
    N_RUNS: int
        number of total samples desired
    DIMS: int
        dimensionality of sample space
    lbs: list
        lower bounds of each parameter range
    ubs: list
        upper bounds of each parameter range
    means: list or ndarray, optional
        Gaussian modes/locations. Defaults to bound midpoints.
    stds: list or ndarray, optional
        Gaussian standard deviations. Defaults to one quarter of each bound
        width.
    save_file: bool (default = True)
        save output to .csv file?
    filename: string
        output sample file path
    random_seed: int, optional
        seed for reproducible draws

    Returns
    -------
    if save_file is False; returns:
        sample: (N_RUNS, DIMS) numpy array of parameter values
    """

    lbs = np.asarray(lbs, dtype=float)
    ubs = np.asarray(ubs, dtype=float)

    if len(lbs) != DIMS or len(ubs) != DIMS:
        raise ValueError("lbs and ubs must each have length DIMS")

    if np.any(ubs <= lbs):
        raise ValueError("Each upper bound must be greater than its lower bound")

    if means is None:
        means = (lbs + ubs) / 2.0
    else:
        means = np.asarray(means, dtype=float)

    if stds is None:
        stds = (ubs - lbs) / 4.0
    else:
        stds = np.asarray(stds, dtype=float)

    if len(means) != DIMS or len(stds) != DIMS:
        raise ValueError("means and stds must each have length DIMS")

    if np.any(stds <= 0):
        raise ValueError("Each Gaussian standard deviation must be positive")

    locs = truncnorm_locs_at_modes(lbs, ubs, means)
    a = (lbs - locs) / stds
    b = (ubs - locs) / stds
    rng = np.random.default_rng(random_seed)
    sample = truncnorm.rvs(a, b, loc=locs, scale=stds,
                           size=(N_RUNS, DIMS), random_state=rng)

    if save_file:
        np.savetxt(filename, sample, delimiter=',')
        print("Gaussian samples drawn and saved!")

    else:
        print("Gaussian samples drawn and returned!")
        return sample

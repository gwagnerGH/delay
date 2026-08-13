"""Explicit, all-or-nothing preference overrides for deterministic runners."""

import os

from .parameter_priors import PARAMETER_PRIOR_INDEX, RUN0_PARAMETER_VALUES


def preference_override_values(ra, eis, prtp):
    """Apply RA_OVERRIDE/EIS_OVERRIDE/PRTP_OVERRIDE if all are supplied."""
    names = ("RA_OVERRIDE", "EIS_OVERRIDE", "PRTP_OVERRIDE")
    supplied = [name for name in names if os.environ.get(name, "") != ""]
    if supplied and len(supplied) != len(names):
        missing = [name for name in names if name not in supplied]
        raise ValueError(
            "Specify RA_OVERRIDE, EIS_OVERRIDE, and PRTP_OVERRIDE together; "
            "missing {}".format(", ".join(missing))
        )
    if supplied:
        return (float(os.environ["RA_OVERRIDE"]),
                float(os.environ["EIS_OVERRIDE"]),
                float(os.environ["PRTP_OVERRIDE"]))
    return float(ra), float(eis), float(prtp)


def preference_override_row(row=None):
    """Return a row-0-like vector with any complete preference override."""
    result = RUN0_PARAMETER_VALUES.copy() if row is None else row.copy()
    ra, eis, prtp = preference_override_values(
        result[PARAMETER_PRIOR_INDEX["RA"]],
        result[PARAMETER_PRIOR_INDEX["EIS"]],
        result[PARAMETER_PRIOR_INDEX["PRTP"]],
    )
    result[PARAMETER_PRIOR_INDEX["RA"]] = ra
    result[PARAMETER_PRIOR_INDEX["EIS"]] = eis
    result[PARAMETER_PRIOR_INDEX["PRTP"]] = prtp
    return result

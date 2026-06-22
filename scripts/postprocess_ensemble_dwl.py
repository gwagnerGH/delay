#!/usr/bin/env python
"""Recover consumption-equivalent DWL from saved ensemble utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONSUMPTION_BILLIONS = 86252.0
DEFAULT_COMPENSATION_YEARS = 5.0


def recover_consumption_compensation(
    u_optimal: float,
    u_delayed: float,
    year0_consumption: float,
    eis: float,
    time_pref: float,
    period_len: float,
) -> float:
    """Invert the Epstein-Zin first-period aggregator for compensation."""

    values = np.array(
        [u_optimal, u_delayed, year0_consumption, eis, time_pref, period_len],
        dtype=float,
    )
    if not np.isfinite(values).all():
        return np.nan
    if u_optimal <= 0.0 or u_delayed <= 0.0 or year0_consumption <= 0.0:
        return np.nan
    if eis <= 0.0 or not 0.0 <= time_pref < 1.0 or period_len <= 0.0:
        return np.nan

    beta = (1.0 - time_pref) ** period_len
    rho = 1.0 - 1.0 / eis
    log_utility_ratio = np.log(u_optimal) - np.log(u_delayed)

    if abs(rho) <= 1e-8:
        return year0_consumption * np.expm1(
            log_utility_ratio / (1.0 - beta)
        )

    relative_power = 1.0 + (
        np.exp(rho * (np.log(u_delayed) - np.log(year0_consumption)))
        * np.expm1(rho * log_utility_ratio)
        / (1.0 - beta)
    )
    if not np.isfinite(relative_power) or relative_power <= 0.0:
        return np.nan

    return year0_consumption * np.expm1(np.log(relative_power) / rho)


def postprocess_results(
    results: pd.DataFrame,
    consumption_billions: float = DEFAULT_CONSUMPTION_BILLIONS,
    compensation_years: float = DEFAULT_COMPENSATION_YEARS,
) -> pd.DataFrame:
    """Return results with closed-form DWL and preserved raw columns."""

    out = results.copy()
    dwl_columns = [
        "delta_c",
        "delta_c_pct",
        "delta_c_billions",
        "delta_c_5yr",
        "delta_c_5yr_pct",
        "delta_c_5yr_billions",
        "delta_c_5yr_total_billions",
    ]
    for column in dwl_columns:
        if column in out:
            out[f"{column}_raw"] = out[column]

    recovered = [
        recover_consumption_compensation(
            row.u_optimal,
            row.u_delayed,
            row.year0_cons_delayed,
            row.eis,
            row.pref,
            row.period_len,
        )
        for row in out.itertuples(index=False)
    ]
    out["delta_c_5yr"] = recovered
    out["delta_c"] = out["delta_c_5yr"]
    out["delta_c_5yr_pct"] = (
        100.0 * out["delta_c_5yr"] / out["year0_cons_delayed"]
    )
    out["delta_c_pct"] = out["delta_c_5yr_pct"]
    out["delta_c_5yr_billions"] = (
        out["delta_c_5yr_pct"] / 100.0 * consumption_billions
    )
    out["delta_c_billions"] = out["delta_c_5yr_billions"]
    out["delta_c_5yr_total_billions"] = (
        out["delta_c_5yr_billions"] * compensation_years
    )

    finite = np.isfinite(out["delta_c_5yr_pct"])
    raw = pd.to_numeric(out.get("delta_c_5yr_pct_raw"), errors="coerce")
    raw_positive = raw > 0.0
    agrees = finite & raw_positive & np.isclose(
        out["delta_c_5yr_pct"], raw, rtol=1e-7, atol=1e-6
    )
    recovered_zero = finite & raw.eq(0.0) & out["delta_c_5yr_pct"].gt(0.0)

    out["dwl_postprocess_status"] = "invalid"
    out.loc[agrees, "dwl_postprocess_status"] = "validated_existing"
    out.loc[recovered_zero, "dwl_postprocess_status"] = "recovered_false_zero"
    out.loc[
        finite & ~(agrees | recovered_zero), "dwl_postprocess_status"
    ] = "recomputed"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--consumption-billions",
        type=float,
        default=DEFAULT_CONSUMPTION_BILLIONS,
    )
    parser.add_argument(
        "--compensation-years",
        type=float,
        default=DEFAULT_COMPENSATION_YEARS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.input_csv)
    corrected = postprocess_results(
        results,
        consumption_billions=args.consumption_billions,
        compensation_years=args.compensation_years,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(args.output_csv, index=False)

    print(corrected["dwl_postprocess_status"].value_counts(dropna=False))
    print(f"Wrote {len(corrected):,} rows to {args.output_csv}")


if __name__ == "__main__":
    main()

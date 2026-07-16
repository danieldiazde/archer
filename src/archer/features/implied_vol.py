from __future__ import annotations

import numpy as np
import pandas as pd


def vix_to_daily_variance(
    vix: pd.Series,
    *,
    calendar_horizon_days: int = 30,
    trading_horizon_days: int = 21,
    annual_calendar_days: int = 365,
) -> pd.Series:
    """
    Convert an annualized VIX level into a mean daily variance forecast.

    VIX is quoted as annualized percentage volatility over approximately
    30 calendar days.

    The conversion is:

        annual_variance = (VIX / 100)^2

        horizon_variance =
            annual_variance
            * calendar_horizon_days
            / annual_calendar_days

        mean_daily_variance =
            horizon_variance
            / trading_horizon_days

    The output is expressed in decimal daily variance units, matching the
    target produced by ``build_vol_dataset``.
    """
    if calendar_horizon_days <= 0:
        raise ValueError(
            "calendar_horizon_days must be strictly positive."
        )

    if trading_horizon_days <= 0:
        raise ValueError(
            "trading_horizon_days must be strictly positive."
        )

    if annual_calendar_days <= 0:
        raise ValueError(
            "annual_calendar_days must be strictly positive."
        )

    if not isinstance(vix, pd.Series):
        raise TypeError("vix must be a pandas Series.")

    if vix.empty:
        raise ValueError("vix series cannot be empty.")

    if vix.index.has_duplicates:
        raise ValueError("vix index contains duplicate dates.")

    if not pd.api.types.is_numeric_dtype(vix):
        raise ValueError("vix values must be numeric.")

    values = vix.astype(float)

    if (
        values.isna().any()
        or not np.isfinite(values.to_numpy()).all()
        or (values <= 0.0).any()
    ):
        raise ValueError(
            "vix values must be strictly positive and finite."
        )

    annual_variance = (values / 100.0).pow(2)

    horizon_variance = (
        annual_variance
        * calendar_horizon_days
        / annual_calendar_days
    )

    daily_variance = (
        horizon_variance
        / trading_horizon_days
    )

    return daily_variance.rename("vix_implied")
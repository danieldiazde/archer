from __future__ import annotations

from typing import Literal
import numpy as np
import pandas as pd

Estimator = Literal["cc", "parkinson", "gk", "rs", "yz"]

_REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "adj_close"}


def _validate_ohlc(df: pd.DataFrame) -> None:
    """
    Validate that an OHLC frame is safe for realized-volatility estimation.

    We need:
    - required OHLC columns
    - parseable dates
    - no duplicate dates
    - strictly positive prices
    """
    if df.empty:
        raise ValueError("OHLC frame is empty.")

    missing_cols = _REQUIRED_COLUMNS - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing_cols)}")

    dates = pd.to_datetime(df["date"], errors="coerce")

    if dates.isna().any():
        raise ValueError("OHLC frame contains unparseable dates.")

    if dates.duplicated().any():
        raise ValueError("OHLC frame contains duplicate dates.")

    price_cols = ["open", "high", "low", "close", "adj_close"]

    for col in price_cols:
        values = pd.to_numeric(df[col], errors="coerce")

        if values.isna().any():
            raise ValueError(f"Column {col!r} contains missing or non-numeric values.")

        if (values <= 0).any():
            raise ValueError(f"Column {col!r} must be strictly positive.")


def _adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale O/H/L/C by adj_close / close so all signal estimators use adjusted bars.

    Example:
        raw close:      100 -> 50
        adj close:       50 -> 50

    The raw series looks like a fake -50% crash.
    The adjusted series correctly shows no economic move.
    """
    _validate_ohlc(df)

    work = df.copy()

    work["date"] = pd.to_datetime(work["date"], errors="raise")
    work = work.sort_values("date").reset_index(drop=True)

    factor = work["adj_close"] / work["close"]

    adjusted = work.copy()

    for col in ["open", "high", "low", "close"]:
        adjusted[col] = work[col] * factor

    adjusted["adj_close"] = work["adj_close"]

    return adjusted

def daily_variance(df: pd.DataFrame, method: Estimator) -> pd.Series:
    """
    Compute a per-day variance proxy.

        r_t = log(C_t / C_{t-1})
        variance_t = r_t²

    Returns daily variance, not annualized volatility.
    """
    if method != "cc":
        raise NotImplementedError(f"Estimator {method!r} is not implemented yet.")

    adjusted = _adjust_ohlc(df)

    close = adjusted["close"].astype(float)

    log_return = np.log(close / close.shift(1))
    variance = log_return**2

    variance.index = pd.DatetimeIndex(adjusted["date"])
    variance.name = "cc"

    return variance

def realized_vol(
    df: pd.DataFrame,
    *,
    method: Estimator = 'cc',
    window : int = 21,
    trading_days : int = 252
) -> pd.Series:
    '''
    Compute annualized realized volatility in decimal units.

    Example:
        0.20 means 20% annualized volatility.
    '''
    if window < 2:
        raise ValueError('Window must include at least 2.')
    
    if trading_days <= 0:
        raise ValueError('Trading days must be positive.')

    variance = daily_variance(df, method = 'cc')

    rolling_variance = variance.rolling(
        window = window,
        min_periods = window,
    ).mean()

    annualized_variance = rolling_variance * float(trading_days)

    vol = np.sqrt(annualized_variance)
    vol.name = f'rv_{method}_{window}'


    return vol
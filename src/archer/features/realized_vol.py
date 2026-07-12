from __future__ import annotations

from typing import Literal, TypeAlias, get_args
import numpy as np
import pandas as pd

Estimator : TypeAlias = Literal["cc", "parkinson", "gk", "rs", "yz"]

VALID_ESTIMATORS = get_args(Estimator)

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
    Compute a per-day variance proxy based on different estimators.
    Returns daily variance, not annualized volatility
    """
    if method == "yz":
        raise ValueError("Yang-Zhang is window-level only. Use realized_vol(..., method='yz').")

    if method not in {"cc", "parkinson", "gk", "rs"}:
        raise NotImplementedError(f"Estimator {method!r} is not implemented yet.")

    adjusted = _adjust_ohlc(df)

    open_ = adjusted['open'].astype(float)
    high = adjusted['high'].astype(float)
    low = adjusted['low'].astype(float)
    close = adjusted["close"].astype(float)
    
    if method == 'cc':
        log_return = np.log(close / close.shift(1))
        variance = log_return**2

    elif method == 'parkinson':
        log_range = np.log(high / low)
        variance = log_range ** 2 / (4.0 * np.log(2.0))
    
    elif method == 'gk':
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        variance = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    
    else:
        variance = (
            np.log(high / close) * np.log(high / open_)
            + np.log(low / close) * np.log(low / open_)
        )


    variance.index = pd.DatetimeIndex(adjusted["date"])
    variance.name = method

    return variance


def _yang_zhang_variance(df: pd.DataFrame, *, window: int) -> pd.Series:
    """
    Compute rolling Yang-Zhang daily variance.

    This is not annualized and not square-rooted yet.
    """
    adjusted = _adjust_ohlc(df)

    open_ = adjusted["open"].astype(float)
    high = adjusted["high"].astype(float)
    low = adjusted["low"].astype(float)
    close = adjusted["close"].astype(float)

    overnight = np.log(open_ / close.shift(1))
    open_close = np.log(close / open_)

    rs = (
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_)
    )

    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))

    overnight_var = overnight.rolling(
        window=window,
        min_periods=window,
    ).var(ddof=1)

    open_close_var = open_close.rolling(
        window=window,
        min_periods=window,
    ).var(ddof=1)

    rs_mean = rs.rolling(
        window=window,
        min_periods=window,
    ).mean()

    yz_variance = overnight_var + k * open_close_var + (1.0 - k) * rs_mean

    yz_variance.index = pd.DatetimeIndex(adjusted["date"])
    yz_variance.name = f"yz_var_{window}"

    return yz_variance

def realized_vol(
    df: pd.DataFrame,
    *,
    method: Estimator = "yz",
    window: int = 21,
    trading_days: int = 252,
) -> pd.Series:
    """
    Compute annualized realized volatility in decimal units.

    Example:
        0.20 means 20% annualized volatility.
    """
    if window < 2:
        raise ValueError("window must be at least 2.")

    if trading_days <= 0:
        raise ValueError("trading_days must be positive.")

    if method == "yz":
        variance = _yang_zhang_variance(df, window=window)

    elif method in {"cc", "parkinson", "gk", "rs"}:
        daily_var = daily_variance(df, method=method)
        variance = daily_var.rolling(
            window=window,
            min_periods=window,
        ).mean()

    else:
        raise NotImplementedError(f"Estimator {method!r} is not implemented yet.")

    annualized_variance = variance * float(trading_days)

    vol = np.sqrt(annualized_variance)
    vol.name = f"rv_{method}_{window}"

    return vol


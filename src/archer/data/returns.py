import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_price_matrix(
    df: pd.DataFrame,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """
    Create a wide price matrix from clean long-format OHLCV data.

    Input format:
        date | symbol | open | high | low | close | adj_close | volume

    Output format:
        date | SPY | QQQ | TLT | ...
    """

    logger.info("Creating price matrix using price column '%s'", price_col)

    required_columns = {"date", "symbol", price_col}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    duplicate_count = df.duplicated(subset=["date", "symbol"]).sum()
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate date-symbol rows.")

    prices = df.pivot(
        index="date",
        columns="symbol",
        values=price_col,
    )

    prices = prices.sort_index()
    prices.columns.name = None

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price matrix index must be sorted.")

    if (prices <= 0).any().any():
        raise ValueError("Price matrix contains non-positive prices.")

    logger.info(
        "Created price matrix with %d dates and %d symbols",
        prices.shape[0],
        prices.shape[1],
    )

    return prices


def make_simple_returns(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple returns from a price matrix.

    Formula:
        r_t = P_t / P_{t-1} - 1
    """

    logger.info("Creating simple returns")

    _validate_price_matrix(price_matrix)

    simple_returns = price_matrix / price_matrix.shift(1) - 1
    simple_returns = simple_returns.dropna(how="all")

    if np.isinf(simple_returns.to_numpy()).any():
        raise ValueError("Simple returns contain infinite values.")

    logger.info(
        "Created simple returns matrix with %d dates and %d symbols",
        simple_returns.shape[0],
        simple_returns.shape[1],
    )

    return simple_returns


def make_log_returns(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Create log returns from a price matrix.

    Formula:
        log_return_t = log(P_t / P_{t-1})
    """

    logger.info("Creating log returns")

    _validate_price_matrix(price_matrix)

    ratio = price_matrix / price_matrix.shift(1)
    log_returns = ratio.apply(np.log)
    log_returns = log_returns.dropna(how="all")

    if np.isinf(log_returns.to_numpy()).any():
        raise ValueError("Log returns contain infinite values.")

    logger.info(
        "Created log returns matrix with %d dates and %d symbols",
        log_returns.shape[0],
        log_returns.shape[1],
    )

    return log_returns


def _validate_price_matrix(price_matrix: pd.DataFrame) -> None:
    """
    Validate that a price matrix is safe to use for return calculations.
    """

    if price_matrix.empty:
        raise ValueError("Price matrix is empty.")

    if not price_matrix.index.is_monotonic_increasing:
        raise ValueError("Price matrix index must be sorted by date.")

    if price_matrix.columns.duplicated().any():
        raise ValueError("Price matrix contains duplicate symbols.")

    if (price_matrix <= 0).any().any():
        raise ValueError("Price matrix contains non-positive prices.")
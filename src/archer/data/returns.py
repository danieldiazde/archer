from typing import Literal
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

PriceField = Literal['close', 'adj_close']

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

def make_price_matrix(
    df: pd.DataFrame,
    field: PriceField = "adj_close",
) -> pd.DataFrame:
    """
    Convert long silver OHLCV data into a wide price matrix.

    Input grain:
        one row per (symbol, date)

    Output:
        index = date
        columns = symbols
        values = selected price field

    Doctrine:
        adj_close -> signals/research/returns
        close     -> fills/sizing/raw execution logic
    """
    required_cols = {"symbol", "date", field}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    work = df[["symbol", "date", field]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    bad_dates = work["date"].isna()
    if bad_dates.any():
        raise ValueError("Cannot build price matrix: some dates could not be parsed.")

    duplicated = work.duplicated(subset=["symbol", "date"], keep=False)
    if duplicated.any():
        examples = work.loc[duplicated, ["symbol", "date"]].head(10)
        raise ValueError(
            "Cannot build price matrix: duplicate (symbol, date) rows found. "
            f"Examples:\n{examples}"
        )

    prices = (
        work.pivot(index="date", columns="symbol", values=field)
        .sort_index()
    )

    prices.columns.name = None

    _validate_price_matrix(prices)

    return prices
    


def make_log_returns(prices : pd.DataFrame) -> pd.DataFrame:
    _validate_price_matrix(prices)
    log_returns = prices.apply(np.log).diff()
    return log_returns.dropna(how='all')


def make_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    _validate_price_matrix(prices)
    simple_returns = prices / prices.shift(1) - 1
    return simple_returns.dropna(how="all")
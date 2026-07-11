from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.data.returns import make_log_returns, make_price_matrix, make_simple_returns


def test_make_price_matrix_defaults_to_adj_close() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "VXX", "VXX"],
            "date": ["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"],
            "close": [100.0, 200.0, 50.0, 100.0],
            "adj_close": [10.0, 20.0, 5.0, 10.0],
        }
    )

    prices = make_price_matrix(df)

    assert list(prices.columns) == ["SPY", "VXX"]
    assert prices.loc[pd.Timestamp("2020-01-02"), "SPY"] == 10.0
    assert prices.loc[pd.Timestamp("2020-01-03"), "VXX"] == 10.0


def test_make_price_matrix_can_use_raw_close() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "date": ["2020-01-02", "2020-01-03"],
            "close": [100.0, 200.0],
            "adj_close": [10.0, 20.0],
        }
    )

    prices = make_price_matrix(df, field="close")

    assert prices.loc[pd.Timestamp("2020-01-02"), "SPY"] == 100.0
    assert prices.loc[pd.Timestamp("2020-01-03"), "SPY"] == 200.0


def test_make_price_matrix_rejects_duplicate_symbol_date() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "date": ["2020-01-02", "2020-01-02"],
            "adj_close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        make_price_matrix(df)


def test_make_price_matrix_rejects_non_positive_prices() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "date": ["2020-01-02", "2020-01-03"],
            "adj_close": [100.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="non-positive"):
        make_price_matrix(df)


def test_log_returns_are_log_price_differences() -> None:
    prices = pd.DataFrame(
        {"SPY": [100.0, 110.0, 121.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
    )

    returns = make_log_returns(prices)

    expected = np.log(1.10)

    assert np.isclose(returns.iloc[0]["SPY"], expected)
    assert np.isclose(returns.iloc[1]["SPY"], expected)


def test_simple_returns_are_price_ratios_minus_one() -> None:
    prices = pd.DataFrame(
        {"SPY": [100.0, 110.0, 121.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
    )

    returns = make_simple_returns(prices)

    assert np.isclose(returns.iloc[0]["SPY"], 0.10)
    assert np.isclose(returns.iloc[1]["SPY"], 0.10)
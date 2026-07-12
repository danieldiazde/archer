import pandas as pd
import numpy as np
from archer.features.realized_vol import _adjust_ohlc, daily_variance


def test_adjusted_bars_removes_split_jump() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "open": [100.0, 50.0],
            "high": [102.0, 51.0],
            "low": [98.0, 49.0],
            "close": [100.0, 50.0],
            "adj_close": [50.0, 50.0],
            "volume": [1000, 1000],
        }
    )

    adjusted = _adjust_ohlc(df)

    assert adjusted.loc[0, "open"] == 50.0
    assert adjusted.loc[0, "high"] == 51.0
    assert adjusted.loc[0, "low"] == 49.0
    assert adjusted.loc[0, "close"] == 50.0

    assert adjusted.loc[1, "open"] == 50.0
    assert adjusted.loc[1, "high"] == 51.0
    assert adjusted.loc[1, "low"] == 49.0
    assert adjusted.loc[1, "close"] == 50.0

def test_daily_variance_cc_uses_adjusted_close_to_close_returns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "open": [100.0, 110.0, 121.0],
            "high": [101.0, 111.0, 122.0],
            "low": [99.0, 109.0, 120.0],
            "close": [100.0, 110.0, 121.0],
            "adj_close": [100.0, 110.0, 121.0],
            "volume": [1000, 1000, 1000],
        }
    )

    out = daily_variance(df, method="cc")

    expected_return_1 = np.log(110.0 / 100.0)
    expected_return_2 = np.log(121.0 / 110.0)

    assert np.isnan(out.iloc[0])
    assert np.isclose(out.iloc[1], expected_return_1**2)
    assert np.isclose(out.iloc[2], expected_return_2**2)
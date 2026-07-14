import pandas as pd
import numpy as np
from archer.features.realized_vol import _adjust_ohlc, daily_variance, realized_vol, VALID_ESTIMATORS
import pytest

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

def test_realized_vol_cc_averages_variance_then_sqrt() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
            ),
            "open": [100.0, 110.0, 121.0, 133.1],
            "high": [101.0, 111.0, 122.0, 134.0],
            "low": [99.0, 109.0, 120.0, 132.0],
            "close": [100.0, 110.0, 121.0, 133.1],
            "adj_close": [100.0, 110.0, 121.0, 133.1],
            "volume": [1000, 1000, 1000, 1000],
        }
    )

    out = realized_vol(df, method="cc", window=2, trading_days=252)

    daily = daily_variance(df, method="cc")

    expected_first_valid = np.sqrt(daily.iloc[1:3].mean() * 252.0)

    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert np.isclose(out.iloc[2], expected_first_valid)

def test_daily_variance_parkinson_uses_high_low_range() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [99.0, 101.0],
            "close": [102.0, 104.0],
            "adj_close": [102.0, 104.0],
            "volume": [1000, 1000],
        }
    )

    out = daily_variance(df, method = 'parkinson')

    expected_0 = np.log(105.0 / 99.0) ** 2 / (4.0 * np.log(2.0))
    expected_1 = np.log(106.0 / 101.0) ** 2 / (4.0 * np.log(2.0))

    assert np.isclose(out.iloc[0], expected_0)
    assert np.isclose(out.iloc[1], expected_1)

def test_realized_vol_parkinson_averages_variance_then_sqrt() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "open": [100.0, 102.0, 104.0],
            "high": [105.0, 106.0, 108.0],
            "low": [99.0, 101.0, 103.0],
            "close": [102.0, 104.0, 107.0],
            "adj_close": [102.0, 104.0, 107.0],
            "volume": [1000, 1000, 1000],
        }
    )

    out = realized_vol(df, method="parkinson", window=2, trading_days=252)

    daily = daily_variance(df, method="parkinson")
    expected = np.sqrt(daily.iloc[0:2].mean() * 252.0)

    assert np.isnan(out.iloc[0])
    assert np.isclose(out.iloc[1], expected)

def test_daily_variance_garman_klass_uses_high_low_and_open_close() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [99.0, 101.0],
            "close": [102.0, 104.0],
            "adj_close": [102.0, 104.0],
            "volume": [1000, 1000],
        }
    )

    out = daily_variance(df, method="gk")

    log_hl_0 = np.log(105.0 / 99.0)
    log_co_0 = np.log(102.0 / 100.0)
    expected_0 = 0.5 * log_hl_0**2 - (2.0 * np.log(2.0) - 1.0) * log_co_0**2

    log_hl_1 = np.log(106.0 / 101.0)
    log_co_1 = np.log(104.0 / 102.0)
    expected_1 = 0.5 * log_hl_1**2 - (2.0 * np.log(2.0) - 1.0) * log_co_1**2

    assert np.isclose(out.iloc[0], expected_0)
    assert np.isclose(out.iloc[1], expected_1)

def test_realized_vol_gk_averages_variance_then_sqrt() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "open": [100.0, 102.0, 104.0],
            "high": [105.0, 106.0, 108.0],
            "low": [99.0, 101.0, 103.0],
            "close": [102.0, 104.0, 107.0],
            "adj_close": [102.0, 104.0, 107.0],
            "volume": [1000, 1000, 1000],
        }
    )

    out = realized_vol(df, method="gk", window=2, trading_days=252)

    daily = daily_variance(df, method="gk")
    expected = np.sqrt(daily.iloc[0:2].mean() * 252.0)

    assert np.isnan(out.iloc[0])
    assert np.isclose(out.iloc[1], expected)

def test_daily_variance_rogers_satchell_uses_full_ohlc_bar() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [99.0, 101.0],
            "close": [102.0, 104.0],
            "adj_close": [102.0, 104.0],
            "volume": [1000, 1000],
        }
    )

    out = daily_variance(df, method="rs")

    expected_0 = (
        np.log(105.0 / 102.0) * np.log(105.0 / 100.0)
        + np.log(99.0 / 102.0) * np.log(99.0 / 100.0)
    )

    expected_1 = (
        np.log(106.0 / 104.0) * np.log(106.0 / 102.0)
        + np.log(101.0 / 104.0) * np.log(101.0 / 102.0)
    )

    assert np.isclose(out.iloc[0], expected_0)
    assert np.isclose(out.iloc[1], expected_1)

def test_realized_vol_rs_averages_variance_then_sqrt() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "open": [100.0, 102.0, 104.0],
            "high": [105.0, 106.0, 108.0],
            "low": [99.0, 101.0, 103.0],
            "close": [102.0, 104.0, 107.0],
            "adj_close": [102.0, 104.0, 107.0],
            "volume": [1000, 1000, 1000],
        }
    )

    out = realized_vol(df, method="rs", window=2, trading_days=252)

    daily = daily_variance(df, method="rs")
    expected = np.sqrt(daily.iloc[0:2].mean() * 252.0)

    assert np.isnan(out.iloc[0])
    assert np.isclose(out.iloc[1], expected)

def test_daily_variance_rejects_yang_zhang_because_it_is_window_level() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [99.0, 101.0],
            "close": [102.0, 104.0],
            "adj_close": [102.0, 104.0],
            "volume": [1000, 1000],
        }
    )

    with pytest.raises(ValueError, match="window-level"):
        daily_variance(df, method="yz")

def test_realized_vol_yang_zhang_combines_overnight_open_close_and_rs() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
            ),
            "open": [100.0, 104.0, 103.0, 108.0],
            "high": [103.0, 106.0, 107.0, 111.0],
            "low": [99.0, 102.0, 101.0, 105.0],
            "close": [102.0, 103.0, 106.0, 109.0],
            "adj_close": [102.0, 103.0, 106.0, 109.0],
            "volume": [1000, 1000, 1000, 1000],
        }
    )

    out = realized_vol(df, method="yz", window=3, trading_days=252)

    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    overnight = np.log(open_ / close.shift(1))
    open_close = np.log(close / open_)

    rs = (
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_)
    )

    window = 3
    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))

    # First valid YZ window is rows 1, 2, 3.
    # Row 0 has no previous close, so overnight is NaN there.
    expected_variance = (
        overnight[1:4].var(ddof=1)
        + k * open_close[1:4].var(ddof=1)
        + (1.0 - k) * rs[1:4].mean()
    )

    expected_vol = np.sqrt(expected_variance * 252.0)

    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert np.isnan(out.iloc[2])
    assert np.isclose(out.iloc[3], expected_vol)

def simulate_intraday_gbm_ohlc(
    *,
    n_days: int,
    steps_per_day: int,
    sigma: float,
    seed: int,
    annual_log_drift: float = 0.0,
) -> pd.DataFrame:
    """
    Simulate OHLC bars from an intraday log-price process.
    """
    rng = np.random.default_rng(seed)

    dt = 1.0 / (252.0 * steps_per_day)
    log_price = np.log(100.0)

    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2010-01-04", periods=n_days)

    for current_date in dates:
        open_log = log_price

        shocks = rng.normal(
            loc=annual_log_drift * dt,
            scale=sigma * np.sqrt(dt),
            size=steps_per_day,
        )

        intraday_log_path = open_log + np.cumsum(shocks)

        full_log_path = np.r_[open_log, intraday_log_path]
        full_price_path = np.exp(full_log_path)

        open_price = full_price_path[0]
        high_price = full_price_path.max()
        low_price = full_price_path.min()
        close_price = full_price_path[-1]

        rows.append(
            {
                "date": current_date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "adj_close": close_price,
                "volume": 1,
            }
        )

        log_price = full_log_path[-1]

    return pd.DataFrame(rows)

def test_simulate_intraday_gbm_ohlc_produces_valid_bars() -> None:
    df = simulate_intraday_gbm_ohlc(
        n_days=5,
        steps_per_day=10,
        sigma=0.20,
        seed=42,
    )

    assert len(df) == 5

    assert list(df.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    assert df["date"].is_monotonic_increasing

    price_cols = ["open", "high", "low", "close", "adj_close"]

    assert (df[price_cols] > 0).all().all()

    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()

    assert np.allclose(df["adj_close"], df["close"])

def test_estimators_recover_known_volatility_under_driftless_gbm() -> None:
    true_sigma = 0.20

    df = simulate_intraday_gbm_ohlc(
        n_days=3_000,
        steps_per_day=390,
        sigma=true_sigma,
        seed=42,
        annual_log_drift=0.0,
    )

    estimates = {
        method: realized_vol(df, method=method, window=21).dropna()
        for method in VALID_ESTIMATORS
    }

    means = {
        method: series.mean()
        for method, series in estimates.items()
    }

    stds = {
        method: series.std()
        for method, series in estimates.items()
    }

    for method, estimate in means.items():
        assert abs(estimate - true_sigma) < 0.03, (method, estimate)

    assert stds["gk"] < stds["parkinson"] < stds["cc"]
    assert stds["rs"] < stds["parkinson"] < stds["cc"]
    assert stds["yz"] < stds["parkinson"] < stds["cc"]

def test_drift_stress_rs_and_yz_are_more_stable_than_cc() -> None:

    true_sigma = 0.20
    drift_assert_threshold = 0.015

    flat = simulate_intraday_gbm_ohlc(
        n_days=3_000,
        steps_per_day=390,
        sigma=true_sigma,
        seed=42,
        annual_log_drift=0.0,
    )

    trending = simulate_intraday_gbm_ohlc(
        n_days=3_000,
        steps_per_day=390,
        sigma=true_sigma,
        seed=42,
        annual_log_drift=2.0,
    )

    flat_means = {
        method: realized_vol(flat, method=method, window=21).dropna().mean()
        for method in VALID_ESTIMATORS
    }

    trending_means = {
        method: realized_vol(trending, method=method, window=21).dropna().mean()
        for method in VALID_ESTIMATORS
    }

    drift_impact = {
        method: trending_means[method] - flat_means[method]
        for method in VALID_ESTIMATORS
    }

    debug_msg = (
        f"flat_means: {flat_means}\n"
        f"trending_means: {trending_means}\n"
        f"drift_impact: {drift_impact}"
    )

    assert drift_impact["cc"] > drift_impact["rs"], debug_msg
    assert drift_impact["cc"] > drift_impact["yz"], debug_msg

    assert abs(drift_impact["rs"]) < drift_assert_threshold, (
        f"Drift impact for RS is not less than {drift_assert_threshold}.\n"
        f"{debug_msg}"
    )

    assert abs(drift_impact["yz"]) < drift_assert_threshold, (
        f"Drift impact for YZ is not less than {drift_assert_threshold}.\n"
        f"{debug_msg}"
    )

def test_daily_variance_gk_total_adds_overnight_gap_to_intraday_gk() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "open": [100.0, 103.0],
            "high": [102.0, 106.0],
            "low": [99.0, 101.0],
            "close": [100.0, 104.0],
            "adj_close": [100.0, 104.0],
            "volume": [1000, 1000],
        }
    )

    out = daily_variance(df, method="gk_total")

    overnight_1 = np.log(103.0 / 100.0)

    log_hl_1 = np.log(106.0 / 101.0)
    log_co_1 = np.log(104.0 / 103.0)

    intraday_gk_1 = (
        0.5 * log_hl_1**2
        - (2.0 * np.log(2.0) - 1.0) * log_co_1**2
    )

    expected_1 = overnight_1**2 + intraday_gk_1

    assert np.isnan(out.iloc[0])
    assert np.isclose(out.iloc[1], expected_1)

def test_realized_vol_gk_total_averages_variance_then_sqrt() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "open": [100.0, 103.0, 104.0],
            "high": [102.0, 106.0, 108.0],
            "low": [99.0, 101.0, 103.0],
            "close": [100.0, 104.0, 107.0],
            "adj_close": [100.0, 104.0, 107.0],
            "volume": [1000, 1000, 1000],
        }
    )

    out = realized_vol(df, method="gk_total", window=2, trading_days=252)

    daily = daily_variance(df, method="gk_total")
    expected = np.sqrt(daily.iloc[1:3].mean() * 252.0)

    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert np.isclose(out.iloc[2], expected)
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.features.implied_vol import vix_to_daily_variance


def test_vix_to_daily_variance_uses_matched_horizon_conversion() -> None:
    idx = pd.bdate_range("2020-01-01", periods=1)

    vix = pd.Series(
        [20.0],
        index=idx,
        name="vix",
    )

    result = vix_to_daily_variance(
        vix,
        calendar_horizon_days=30,
        trading_horizon_days=21,
        annual_calendar_days=365,
    )

    expected = (0.20**2) * (30.0 / 365.0) / 21.0

    assert np.isclose(result.iloc[0], expected)
    assert result.index.equals(idx)
    assert result.name == "vix_implied"


def test_vix_to_daily_variance_is_strictly_positive() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)

    vix = pd.Series(
        [10.0, 20.0, 40.0],
        index=idx,
    )

    result = vix_to_daily_variance(vix)

    assert (result > 0.0).all()


def test_vix_to_daily_variance_preserves_quadratic_scaling() -> None:
    idx = pd.bdate_range("2020-01-01", periods=2)

    vix = pd.Series(
        [10.0, 20.0],
        index=idx,
    )

    result = vix_to_daily_variance(vix)

    # Doubling volatility quadruples variance.
    assert np.isclose(
        result.iloc[1] / result.iloc[0],
        4.0,
    )


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 20.0],
        [-10.0, 20.0],
        [np.nan, 20.0],
        [np.inf, 20.0],
    ],
)
def test_vix_to_daily_variance_rejects_invalid_values(
    values: list[float],
) -> None:
    idx = pd.bdate_range("2020-01-01", periods=2)

    vix = pd.Series(
        values,
        index=idx,
    )

    with pytest.raises(
        ValueError,
        match="strictly positive and finite",
    ):
        vix_to_daily_variance(vix)


@pytest.mark.parametrize(
    "argument,value",
    [
        ("calendar_horizon_days", 0),
        ("trading_horizon_days", 0),
        ("annual_calendar_days", 0),
    ],
)
def test_vix_to_daily_variance_rejects_invalid_day_counts(
    argument: str,
    value: int,
) -> None:
    idx = pd.bdate_range("2020-01-01", periods=1)
    vix = pd.Series([20.0], index=idx)

    kwargs = {
        "calendar_horizon_days": 30,
        "trading_horizon_days": 21,
        "annual_calendar_days": 365,
    }
    kwargs[argument] = value

    with pytest.raises(
        ValueError,
        match="must be strictly positive",
    ):
        vix_to_daily_variance(
            vix,
            **kwargs,
        )
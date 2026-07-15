from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.models.benchmarks import (
    AlignedSeriesForecaster,
    NaiveMA22Forecaster,
)
from archer.models.dataset import VolDataset
from archer.models.fold import make_forecast_fold


def make_dataset(
    n: int = 80,
    horizon: int = 5,
) -> VolDataset:
    idx = pd.bdate_range("2020-01-01", periods=n)

    X = pd.DataFrame(
        {
            "har_d": np.linspace(0.00001, 0.00020, n),
            "har_w": np.linspace(0.00002, 0.00021, n),
            "har_m": np.linspace(0.00003, 0.00022, n),
        },
        index=idx,
    )

    y = pd.Series(
        np.linspace(0.00004, 0.00023, n),
        index=idx,
        name=f"future_mean_variance_{horizon}",
    )

    y_end = pd.Series(
        idx + pd.offsets.BDay(horizon),
        index=idx,
        name="y_end",
    )

    returns = pd.Series(
        0.0,
        index=idx,
        name="returns",
    )

    return VolDataset(
        X=X,
        y=y,
        y_end=y_end,
        returns=returns,
        horizon=horizon,
    )


def test_naive_ma22_predicts_monthly_har_feature() -> None:
    ds = make_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[49],
    )

    model = NaiveMA22Forecaster()

    model.fit(ds, fold)
    pred = model.predict(ds, fold)

    expected = (
        ds.X.loc[fold.test_idx, "har_m"]
        .rename("naive_ma22")
    )

    assert pred.equals(expected)
    assert pred.index.equals(fold.test_idx)
    assert pred.name == "naive_ma22"


def test_aligned_series_forecaster_returns_fold_rows() -> None:
    ds = make_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[49],
    )

    external_forecast = pd.Series(
        np.linspace(0.00005, 0.00025, len(ds.X)),
        index=ds.X.index,
        name="external",
    )

    model = AlignedSeriesForecaster(
        name="vix_implied",
        series=external_forecast,
    )

    model.fit(ds, fold)
    pred = model.predict(ds, fold)

    expected = external_forecast.loc[fold.test_idx].rename(
        "vix_implied"
    )

    assert pred.equals(expected)
    assert pred.index.equals(fold.test_idx)
    assert pred.name == "vix_implied"


def test_aligned_series_forecaster_rejects_missing_test_dates() -> None:
    ds = make_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[49],
    )

    external_forecast = pd.Series(
        0.00010,
        index=ds.X.index[:-1],
    )

    model = AlignedSeriesForecaster(
        name="vix_implied",
        series=external_forecast,
    )

    with pytest.raises(
        ValueError,
        match="missing required forecast origins",
    ):
        model.predict(ds, fold)


def test_aligned_series_forecaster_rejects_nonpositive_values() -> None:
    ds = make_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[49],
    )

    external_forecast = pd.Series(
        0.00010,
        index=ds.X.index,
    )
    external_forecast.loc[fold.test_idx[0]] = 0.0

    model = AlignedSeriesForecaster(
        name="vix_implied",
        series=external_forecast,
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        model.predict(ds, fold)


def test_aligned_series_forecaster_rejects_duplicate_dates() -> None:
    ds = make_dataset()

    duplicated_index = ds.X.index.insert(
        len(ds.X),
        ds.X.index[-1],
    )

    external_forecast = pd.Series(
        0.00010,
        index=duplicated_index,
    )

    with pytest.raises(
        ValueError,
        match="duplicate dates",
    ):
        AlignedSeriesForecaster(
            name="vix_implied",
            series=external_forecast,
        )
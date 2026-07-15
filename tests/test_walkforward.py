from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from archer.analytics.walkforward import run_walkforward
from archer.models.base import Forecaster
from archer.models.benchmarks import NaiveMA22Forecaster
from archer.models.dataset import VolDataset
from archer.models.fold import ForecastFold, make_expanding_folds
from archer.models.har import HARForecaster


def make_linear_dataset(
    n: int = 170,
    horizon: int = 10,
) -> VolDataset:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=n)

    X = pd.DataFrame(
        {
            "har_d": rng.uniform(0.00005, 0.00050, size=n),
            "har_w": rng.uniform(0.00005, 0.00050, size=n),
            "har_m": rng.uniform(0.00005, 0.00050, size=n),
        },
        index=idx,
    )

    y = (
        0.00001
        + 0.20 * X["har_d"]
        + 0.30 * X["har_w"]
        + 0.40 * X["har_m"]
    )
    y.name = f"future_mean_variance_{horizon}"

    y_end = pd.Series(
        idx + pd.offsets.BDay(horizon),
        index=idx,
        name="y_end",
    )

    returns = pd.Series(
        rng.normal(0.0, 0.01, size=n),
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


def test_run_walkforward_produces_prediction_panel() -> None:
    ds = make_linear_dataset()

    folds = make_expanding_folds(
        ds,
        first_cutoff=ds.X.index[90],
        refit_every=21,
    )

    panel = run_walkforward(
        ds=ds,
        folds=folds,
        model_factories=[
            NaiveMA22Forecaster,
            HARForecaster,
        ],
    )

    assert panel.horizon == ds.horizon
    assert panel.model_names == (
        "naive_ma22",
        "har",
    )

    expected_idx = pd.DatetimeIndex(
        [
            date
            for fold in folds
            for date in fold.test_idx
        ],
        name="forecast_origin",
    )

    assert panel.frame.index.equals(expected_idx)
    assert panel.frame.index.is_unique

    assert "realized" in panel.frame.columns
    assert "target_end" in panel.frame.columns
    assert "fit_cutoff" in panel.frame.columns
    assert "fold_id" in panel.frame.columns
    assert "naive_ma22" in panel.frame.columns
    assert "har" in panel.frame.columns


def test_run_walkforward_har_recovers_exact_linear_relationship() -> None:
    ds = make_linear_dataset()

    folds = make_expanding_folds(
        ds,
        first_cutoff=ds.X.index[90],
        refit_every=21,
    )

    panel = run_walkforward(
        ds=ds,
        folds=folds,
        model_factories=[
            HARForecaster,
        ],
    )

    assert np.allclose(
        panel.frame["har"].to_numpy(),
        panel.frame["realized"].to_numpy(),
    )


def test_run_walkforward_naive_ma22_matches_har_m_feature() -> None:
    ds = make_linear_dataset()

    folds = make_expanding_folds(
        ds,
        first_cutoff=ds.X.index[90],
        refit_every=21,
    )

    panel = run_walkforward(
        ds=ds,
        folds=folds,
        model_factories=[
            NaiveMA22Forecaster,
        ],
    )

    expected = ds.X.loc[
        panel.frame.index,
        "har_m",
    ]

    assert panel.frame["naive_ma22"].equals(
        expected.rename("naive_ma22")
    )


class CountingForecaster:
    created_count = 0

    def __init__(self) -> None:
        type(self).created_count += 1
        self.name = f"counting_{type(self).created_count}"
        self.fitted_ = False

    def fit(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> None:
        _ = ds
        _ = fold
        self.fitted_ = True

    def predict(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("model was not fitted")

        return pd.Series(
            0.00010,
            index=fold.test_idx,
            name=self.name,
        )


def test_run_walkforward_uses_fresh_model_instance_per_fold() -> None:
    ds = make_linear_dataset()

    folds = make_expanding_folds(
        ds,
        first_cutoff=ds.X.index[90],
        refit_every=21,
    )

    CountingForecaster.created_count = 0

    run_walkforward(
        ds=ds,
        folds=folds,
        model_factories=[
            CountingForecaster,
        ],
    )

    assert CountingForecaster.created_count == len(folds)


class BadIndexForecaster:
    name = "bad_index"

    def fit(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> None:
        _ = ds
        _ = fold

    def predict(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> pd.Series:
        _ = ds

        return pd.Series(
            0.00010,
            index=fold.test_idx[:-1],
            name=self.name,
        )


def test_run_walkforward_rejects_bad_prediction_index() -> None:
    ds = make_linear_dataset()

    folds = make_expanding_folds(
        ds,
        first_cutoff=ds.X.index[90],
        refit_every=21,
    )

    with pytest.raises(
        ValueError,
        match="prediction index must equal fold test index",
    ):
        run_walkforward(
            ds=ds,
            folds=folds,
            model_factories=[
                BadIndexForecaster,
            ],
        )
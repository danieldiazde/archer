from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.models.dataset import VolDataset
from archer.models.fold import make_forecast_fold
from archer.models.har import HARForecaster


def make_linear_har_dataset(
    n: int = 120,
    horizon: int = 10,
) -> VolDataset:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=n)

    X = pd.DataFrame(
        {
            "har_d": rng.uniform(0.00005, 0.0005, size=n),
            "har_w": rng.uniform(0.00005, 0.0005, size=n),
            "har_m": rng.uniform(0.00005, 0.0005, size=n),
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


def test_har_forecaster_fits_and_predicts_fold_test_rows() -> None:
    ds = make_linear_har_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[79],
    )

    model = HARForecaster()
    model.fit(ds, fold)

    pred = model.predict(ds, fold)
    expected = ds.y.loc[fold.test_idx]

    assert pred.index.equals(fold.test_idx)
    assert pred.name == "har"
    assert (pred > 0).all()

    # The synthetic target is exactly linear in the HAR features.
    assert np.allclose(
        pred.to_numpy(),
        expected.to_numpy(),
    )


def test_har_forecaster_fits_only_purged_training_rows() -> None:
    ds = make_linear_har_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[79],
    )

    model = HARForecaster()
    model.fit(ds, fold)

    assert model.result_ is not None
    assert int(model.result_.nobs) == len(fold.train_idx)


def test_har_forecaster_uses_hac_covariance() -> None:
    ds = make_linear_har_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[79],
    )

    model = HARForecaster(hac_lags=21)
    model.fit(ds, fold)

    assert model.result_ is not None
    assert model.result_.cov_type == "HAC"


def test_har_forecaster_rejects_predict_before_fit() -> None:
    ds = make_linear_har_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[79],
    )

    model = HARForecaster()

    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(ds, fold)


def test_har_forecaster_floors_negative_predictions() -> None:
    horizon = 5
    idx = pd.bdate_range("2020-01-01", periods=80)

    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        {
            "har_d": np.linspace(0.0, 1.0, len(idx)),
            "har_w": rng.uniform(0.0, 1.0, len(idx)),
            "har_m": rng.uniform(0.0, 1.0, len(idx)),
        },
        index=idx,
    )

    # Exact artificial relationship. Later test rows have sufficiently high
    # har_d values to produce negative raw variance forecasts.
    y = pd.Series(
        0.01 - 0.02 * X["har_d"],
        index=idx,
        name=f"future_mean_variance_{horizon}",
    )

    ds = VolDataset(
        X=X,
        y=y,
        y_end=pd.Series(
            idx + pd.offsets.BDay(horizon),
            index=idx,
            name="y_end",
        ),
        returns=pd.Series(
            0.0,
            index=idx,
            name="returns",
        ),
        horizon=horizon,
    )

    fold = make_forecast_fold(
        ds,
        cutoff=idx[49],
    )

    model = HARForecaster(epsilon=1e-8)
    model.fit(ds, fold)

    pred = model.predict(ds, fold)

    assert (pred >= 1e-8).all()
    assert np.isclose(pred.min(), 1e-8)
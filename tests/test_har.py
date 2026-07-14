from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.models.dataset import VolDataset
from archer.models.har import HARForecaster


def make_linear_har_dataset(n: int = 100) -> VolDataset:
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
    y.name = "future_mean_variance_21"

    returns = pd.Series(0.0, index=idx, name="returns")

    return VolDataset(
        X=X,
        y=y,
        returns=returns,
        horizon=21,
    )


def test_har_forecaster_fits_and_predicts_aligned_variance_series() -> None:
    ds = make_linear_har_dataset()

    model = HARForecaster()
    model.fit(ds)

    pred = model.predict(ds)

    assert pred.index.equals(ds.X.index)
    assert pred.name == "har"
    assert (pred > 0).all()

    # Because the dataset is exactly linear, predictions should be essentially exact.
    assert np.allclose(pred.to_numpy(), ds.y.to_numpy())


def test_har_forecaster_uses_hac_covariance() -> None:
    ds = make_linear_har_dataset()

    model = HARForecaster(hac_lags=21)
    model.fit(ds)

    assert model.result_ is not None
    assert model.result_.cov_type == "HAC"


def test_har_forecaster_rejects_predict_before_fit() -> None:
    ds = make_linear_har_dataset()

    model = HARForecaster()

    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(ds)


def test_har_forecaster_floors_negative_predictions() -> None:
    idx = pd.bdate_range("2020-01-01", periods=30)

    X = pd.DataFrame(
        {
            "har_d": np.linspace(0.0, 1.0, len(idx)),
            "har_w": np.linspace(0.0, 1.0, len(idx)),
            "har_m": np.linspace(0.0, 1.0, len(idx)),
        },
        index=idx,
    )

    # This is intentionally artificial. It forces an OLS line that can predict
    # negative variance for high feature values.
    y = pd.Series(
        0.01 - 0.02 * X["har_d"],
        index=idx,
        name="future_mean_variance_21",
    )

    ds = VolDataset(
        X=X,
        y=y,
        returns=pd.Series(0.0, index=idx, name="returns"),
        horizon=21,
    )

    model = HARForecaster(epsilon=1e-8)
    model.fit(ds)

    pred = model.predict(ds)

    assert (pred >= 1e-8).all()
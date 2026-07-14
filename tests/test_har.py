from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.models.dataset import VolDataset
from archer.models.har import HARForecaster


def make_y_end(
    idx: pd.DatetimeIndex,
    horizon: int,
) -> pd.Series:
    """
    Create synthetic target-window end dates aligned with a test dataset.
    """
    return pd.Series(
        idx + pd.offsets.BDay(horizon),
        index=idx,
        name="y_end",
    )


def make_linear_har_dataset(n: int = 100) -> VolDataset:
    horizon = 21

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

    returns = pd.Series(
        0.0,
        index=idx,
        name="returns",
    )

    return VolDataset(
        X=X,
        y=y,
        y_end=make_y_end(idx, horizon),
        returns=returns,
        horizon=horizon,
    )


def test_har_forecaster_fits_and_predicts_aligned_variance_series() -> None:
    ds = make_linear_har_dataset()

    model = HARForecaster()
    model.fit(ds)

    pred = model.predict(ds)

    assert pred.index.equals(ds.X.index)
    assert pred.name == "har"
    assert (pred > 0).all()

    # The target is exactly linear in the three HAR features, so OLS should
    # reproduce it up to floating-point precision.
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
    horizon = 21
    idx = pd.bdate_range("2020-01-01", periods=30)

    X = pd.DataFrame(
        {
            "har_d": np.linspace(0.0, 1.0, len(idx)),
            "har_w": np.linspace(0.0, 1.0, len(idx)),
            "har_m": np.linspace(0.0, 1.0, len(idx)),
        },
        index=idx,
    )

    # Intentionally artificial: this creates a fitted line that produces
    # negative raw predictions, allowing us to test the positive floor.
    y = pd.Series(
        0.01 - 0.02 * X["har_d"],
        index=idx,
        name="future_mean_variance_21",
    )

    ds = VolDataset(
        X=X,
        y=y,
        y_end=make_y_end(idx, horizon),
        returns=pd.Series(
            0.0,
            index=idx,
            name="returns",
        ),
        horizon=horizon,
    )

    model = HARForecaster(epsilon=1e-8)
    model.fit(ds)

    pred = model.predict(ds)

    assert (pred >= 1e-8).all()
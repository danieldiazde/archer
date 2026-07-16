from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.models.dataset import VolDataset
from archer.models.fold import make_forecast_fold
from archer.models.garch import GarchForecaster


def make_dataset(
    n: int = 260,
    horizon: int = 21,
) -> VolDataset:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=n)

    X = pd.DataFrame(
        {
            "har_d": np.linspace(0.00005, 0.00030, n),
            "har_w": np.linspace(0.00006, 0.00031, n),
            "har_m": np.linspace(0.00007, 0.00032, n),
        },
        index=idx,
    )

    y = pd.Series(
        np.linspace(0.00008, 0.00033, n),
        index=idx,
        name=f"future_mean_variance_{horizon}",
    )

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


def test_garch_forecaster_fits_on_returns_through_cutoff() -> None:
    ds = make_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[179],
    )

    model = GarchForecaster()
    model.fit(ds, fold)

    assert model.result_ is not None
    assert model.fit_end_ == fold.cutoff
    assert model.nobs_ == len(ds.returns.loc[: fold.cutoff].dropna())


def test_garch_forecaster_predicts_positive_fold_test_variances() -> None:
    ds = make_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[179],
    )

    model = GarchForecaster()
    model.fit(ds, fold)

    pred = model.predict(ds, fold)

    assert pred.index.equals(fold.test_idx)
    assert pred.name == "garch"
    assert (pred > 0.0).all()
    assert np.isfinite(pred.to_numpy()).all()


def test_garch_forecaster_outputs_decimal_daily_variance_units() -> None:
    ds = make_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[179],
    )

    model = GarchForecaster()
    model.fit(ds, fold)

    pred = model.predict(ds, fold)

    # Returns are around 1% daily volatility, so daily variance should be
    # around 1e-4, not around 1.0 or 100.0. This catches missing /10000.
    assert pred.median() > 1e-6
    assert pred.median() < 1e-2


def test_garch_forecaster_rejects_predict_before_fit() -> None:
    ds = make_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[179],
    )

    model = GarchForecaster()

    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(ds, fold)


def test_garch_forecaster_rejects_empty_training_returns() -> None:
    ds = make_dataset(n=80)

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[0],
    )

    model = GarchForecaster(min_obs=50)

    with pytest.raises(ValueError, match="at least 50 return observations"):
        model.fit(ds, fold)


def test_garch_forecaster_rejects_missing_cutoff_return() -> None:
    ds = make_dataset()

    fold = make_forecast_fold(
        ds,
        cutoff=ds.X.index[179],
    )

    returns = ds.returns.drop(index=fold.cutoff)

    broken = VolDataset(
        X=ds.X,
        y=ds.y,
        y_end=ds.y_end,
        returns=returns,
        horizon=ds.horizon,
    )

    model = GarchForecaster()

    with pytest.raises(ValueError, match="must include the fold cutoff"):
        model.fit(broken, fold)
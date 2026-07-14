from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.models.dataset import build_vol_dataset
from archer.models.fold import make_forecast_fold


def make_dataset(
    n: int = 150,
    horizon: int = 10,
):
    dates = pd.bdate_range("2020-01-01", periods=n)

    variance = pd.Series(
        np.linspace(0.00001, 0.00050, n),
        index=dates,
        name="variance",
    )

    returns = pd.Series(
        np.linspace(-0.02, 0.02, n),
        index=dates,
        name="returns",
    )

    ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=horizon,
        weekly_window=5,
        monthly_window=22,
    )

    return dates, ds


def test_make_forecast_fold_is_leakage_safe() -> None:
    dates, ds = make_dataset()

    cutoff = dates[90]

    fold = make_forecast_fold(
        ds,
        cutoff=cutoff,
    )

    train_ds = fold.train_dataset(ds)
    test_ds = fold.test_dataset(ds)

    assert fold.cutoff == cutoff

    # Every supervised training label is fully known by the cutoff.
    assert (train_ds.y_end <= cutoff).all()

    # Every test forecast origin occurs strictly after the cutoff.
    assert (test_ds.X.index > cutoff).all()

    # The final HAR training feature occurs before the cutoff because its
    # complete future target must still end by the cutoff.
    assert train_ds.X.index.max() < cutoff


def test_forecast_fold_matches_dataset_split() -> None:
    dates, ds = make_dataset()

    cutoff = dates[90]

    expected_train, expected_test = ds.split(cutoff)

    fold = make_forecast_fold(
        ds,
        cutoff=cutoff,
    )

    train_ds = fold.train_dataset(ds)
    test_ds = fold.test_dataset(ds)

    assert train_ds.X.equals(expected_train.X)
    assert train_ds.y.equals(expected_train.y)
    assert train_ds.y_end.equals(expected_train.y_end)
    assert train_ds.returns.equals(expected_train.returns)

    assert test_ds.X.equals(expected_test.X)
    assert test_ds.y.equals(expected_test.y)
    assert test_ds.y_end.equals(expected_test.y_end)
    assert test_ds.returns.equals(expected_test.returns)


def test_forecast_fold_can_limit_test_period() -> None:
    dates, ds = make_dataset()

    cutoff = dates[90]
    test_end = dates[110]

    fold = make_forecast_fold(
        ds,
        cutoff=cutoff,
        test_end=test_end,
    )

    test_ds = fold.test_dataset(ds)

    assert not test_ds.X.empty
    assert (test_ds.X.index > cutoff).all()
    assert (test_ds.X.index <= test_end).all()
    assert test_ds.X.index.max() <= test_end


def test_full_dataset_keeps_returns_through_cutoff_for_garch() -> None:
    dates, ds = make_dataset(horizon=10)

    cutoff = dates[90]

    fold = make_forecast_fold(
        ds,
        cutoff=cutoff,
    )

    train_ds = fold.train_dataset(ds)
    fit_returns = ds.returns.loc[:fold.cutoff]

    # HAR must stop earlier because its labels extend into the future.
    assert train_ds.X.index.max() < cutoff

    # GARCH may use observed returns through the cutoff.
    assert fit_returns.index.max() == cutoff

    # Therefore GARCH receives more recent information than the final
    # supervised HAR training row, without using any future information.
    assert fit_returns.index.max() > train_ds.X.index.max()


def test_make_forecast_fold_rejects_test_end_at_or_before_cutoff() -> None:
    dates, ds = make_dataset()

    cutoff = dates[90]

    with pytest.raises(ValueError, match="test_end must be after cutoff"):
        make_forecast_fold(
            ds,
            cutoff=cutoff,
            test_end=cutoff,
        )
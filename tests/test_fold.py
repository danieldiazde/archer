from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.models.dataset import build_vol_dataset
from archer.models.fold import (
    make_expanding_folds,
    make_forecast_fold,
    )



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

def test_make_expanding_folds_covers_oos_dates_without_gaps() -> None:
    dates, ds = make_dataset(
        n=170,
        horizon=10,
    )

    first_cutoff = dates[90]

    folds = make_expanding_folds(
        ds,
        first_cutoff=first_cutoff,
        refit_every=21,
    )

    expected_test_idx = ds.X.index[
        ds.X.index > first_cutoff
    ]

    actual_test_idx = pd.DatetimeIndex(
        [
            date
            for fold in folds
            for date in fold.test_idx
        ]
    )

    assert actual_test_idx.equals(expected_test_idx)
    assert actual_test_idx.is_unique


def test_make_expanding_folds_refits_every_21_origins() -> None:
    dates, ds = make_dataset(
        n=170,
        horizon=10,
    )

    folds = make_expanding_folds(
        ds,
        first_cutoff=dates[90],
        refit_every=21,
    )

    assert len(folds) > 1

    # Every complete fold contains exactly 21 forecast origins.
    for fold in folds[:-1]:
        assert len(fold.test_idx) == 21

    # The final fold may be shorter.
    assert 1 <= len(folds[-1].test_idx) <= 21


def test_expanding_fold_cutoffs_chain_without_overlap() -> None:
    dates, ds = make_dataset(
        n=170,
        horizon=10,
    )

    first_cutoff = dates[90]

    folds = make_expanding_folds(
        ds,
        first_cutoff=first_cutoff,
        refit_every=21,
    )

    assert folds[0].cutoff == first_cutoff

    for expected_id, fold in enumerate(folds):
        train_ds = fold.train_dataset(ds)
        test_ds = fold.test_dataset(ds)

        assert fold.fold_id == expected_id
        assert fold.test_end == test_ds.X.index.max()

        # Supervised labels are fully observed by the fit cutoff.
        assert (train_ds.y_end <= fold.cutoff).all()

        # Forecast origins occur strictly after the fit cutoff.
        assert (test_ds.X.index > fold.cutoff).all()

    for previous, current in zip(folds, folds[1:]):
        # The next parameter refit occurs after the previous prediction block.
        assert current.cutoff == previous.test_end

        # Forecast blocks do not overlap.
        assert current.test_idx.min() > previous.test_idx.max()


def test_make_expanding_folds_respects_final_test_end() -> None:
    dates, ds = make_dataset(
        n=170,
        horizon=10,
    )

    first_cutoff = dates[90]
    final_test_end = dates[135]

    folds = make_expanding_folds(
        ds,
        first_cutoff=first_cutoff,
        refit_every=21,
        final_test_end=final_test_end,
    )

    actual_test_idx = pd.DatetimeIndex(
        [
            date
            for fold in folds
            for date in fold.test_idx
        ]
    )

    expected_test_idx = ds.X.index[
        (ds.X.index > first_cutoff)
        & (ds.X.index <= final_test_end)
    ]

    assert actual_test_idx.equals(expected_test_idx)
    assert folds[-1].test_end <= final_test_end


def test_make_expanding_folds_rejects_invalid_refit_frequency() -> None:
    _, ds = make_dataset()

    with pytest.raises(
        ValueError,
        match="refit_every must be at least 1",
    ):
        make_expanding_folds(
            ds,
            first_cutoff="2020-04-01",
            refit_every=0,
        )
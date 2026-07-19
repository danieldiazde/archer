from __future__ import annotations

import numpy as np
import pandas as pd

from archer.models.dataset import build_vol_dataset


def make_series(
    *,
    n: int = 100,
) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series]:
    dates = pd.bdate_range("2020-01-01", periods=n)

    variance = pd.Series(
        np.arange(n, dtype=float),
        index=dates,
        name="variance",
    )

    returns = pd.Series(
        np.arange(n, dtype=float) / 100.0,
        index=dates,
        name="returns",
    )

    return dates, variance, returns


def test_build_vol_dataset_is_point_in_time_safe() -> None:
    dates, variance, returns = make_series(n=100)

    ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=21,
        weekly_window=5,
        monthly_window=22,
    )

    date = dates[30]
    t = 30.0

    assert ds.horizon == 21

    assert ds.X.loc[date, "har_d"] == t
    assert ds.X.loc[date, "har_w"] == t - 2.0
    assert ds.X.loc[date, "har_m"] == t - 10.5

    # mean(t + 1, ..., t + 21) = t + 11
    assert ds.y.loc[date] == t + 11.0

    # The final value in that target is v_{t+21}.
    assert ds.y_end.loc[date] == dates[51]

    assert ds.X.index.equals(ds.y.index)
    assert ds.X.index.equals(ds.y_end.index)
    assert ds.X.index.equals(ds.returns.index)


def test_build_vol_dataset_drops_rows_without_monthly_history() -> None:
    dates, variance, returns = make_series(n=100)

    ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=21,
        weekly_window=5,
        monthly_window=22,
    )

    # A 22-day trailing average first exists at position 21.
    assert ds.X.index.min() == dates[21]


def test_build_vol_dataset_drops_rows_without_complete_future_target() -> None:
    dates, variance, returns = make_series(n=100)

    ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=21,
        weekly_window=5,
        monthly_window=22,
    )

    # The final valid row must still have 21 future observations.
    assert ds.X.index.max() == dates[-22]

    # Its target ends on the final available date.
    assert ds.y_end.iloc[-1] == dates[-1]


def test_vol_dataset_slice_preserves_alignment() -> None:
    _, variance, returns = make_series(n=100)

    ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=21,
        weekly_window=5,
        monthly_window=22,
    )

    idx  = ds.X.index[5:10]

    sliced = ds.slice(idx)

    assert sliced.horizon == ds.horizon

    assert sliced.X.index.equals(idx)
    assert sliced.y.index.equals(idx)
    assert sliced.y_end.index.equals(idx)
    assert sliced.returns.index.equals(idx)

    assert sliced.X.equals(ds.X.loc[idx])
    assert sliced.y.equals(ds.y.loc[idx])
    assert sliced.y_end.equals(ds.y_end.loc[idx])
    assert sliced.returns.equals(ds.returns.loc[idx])


def test_vol_dataset_split_purges_training_targets_after_cutoff() -> None:
    dates, variance, returns = make_series(n=120)

    ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=10,
        weekly_window=5,
        monthly_window=22,
    )

    cutoff = dates[80]

    train, test = ds.split(cutoff)

    # No training label uses information after the cutoff.
    assert (train.y_end <= cutoff).all()

    # Test prediction dates begin strictly after the cutoff.
    assert (test.X.index > cutoff).all()

    # A row whose feature date is before the cutoff can still be excluded
    # when its target extends beyond the cutoff.
    assert train.X.index.max() < cutoff

    assert train.X.index.equals(train.y.index)
    assert train.X.index.equals(train.y_end.index)
    assert train.X.index.equals(train.returns.index)

    assert test.X.index.equals(test.y.index)
    assert test.X.index.equals(test.y_end.index)
    assert test.X.index.equals(test.returns.index)


def test_purged_split_matches_truncate_then_build() -> None:
    dates, variance, returns = make_series(n=120)

    cutoff = dates[80]

    full_ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=10,
        weekly_window=5,
        monthly_window=22,
    )

    split_train, _ = full_ds.split(cutoff)

    truncated_ds = build_vol_dataset(
        variance=variance.loc[:cutoff],
        returns=returns.loc[:cutoff],
        horizon=10,
        weekly_window=5,
        monthly_window=22,
    )

    assert split_train.X.equals(truncated_ds.X)
    assert split_train.y.equals(truncated_ds.y)
    assert split_train.y_end.equals(truncated_ds.y_end)
    assert split_train.returns.equals(truncated_ds.returns)
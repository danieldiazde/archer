from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.analytics.panel import (
    make_prediction_panel,
)
from archer.models.dataset import build_vol_dataset
from archer.models.fold import make_expanding_folds


def make_dataset(
    n: int = 170,
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


def test_make_prediction_panel_records_fold_metadata() -> None:
    dates, ds = make_dataset()

    folds = make_expanding_folds(
        ds,
        first_cutoff=dates[90],
        refit_every=21,
    )

    panel = make_prediction_panel(
        ds,
        folds,
    )

    expected_idx = pd.DatetimeIndex(
        [
            date
            for fold in folds
            for date in fold.test_idx
        ],
        name="forecast_origin",
    )

    assert panel.horizon == ds.horizon
    assert panel.frame.index.equals(expected_idx)
    assert panel.frame.index.is_unique
    assert panel.frame.index.is_monotonic_increasing

    assert list(panel.frame.columns) == [
        "realized",
        "target_end",
        "fit_cutoff",
        "fold_id",
    ]

    assert panel.frame["realized"].equals(
        ds.y.loc[expected_idx].rename("realized")
    )

    assert panel.frame["target_end"].equals(
        ds.y_end.loc[expected_idx].rename("target_end")
    )

    for fold in folds:
        rows = panel.frame.loc[fold.test_idx]

        assert (rows["fit_cutoff"] == fold.cutoff).all()
        assert (rows["fold_id"] == fold.fold_id).all()


def test_prediction_panel_adds_aligned_forecast() -> None:
    dates, ds = make_dataset()

    folds = make_expanding_folds(
        ds,
        first_cutoff=dates[90],
        refit_every=21,
    )

    panel = make_prediction_panel(ds, folds)

    forecast = pd.Series(
        0.00010,
        index=panel.frame.index,
        name="har",
    )

    updated = panel.with_forecast(
        name="har",
        forecast=forecast,
    )

    assert "har" not in panel.frame.columns
    assert "har" in updated.frame.columns
    assert updated.frame["har"].equals(forecast)
    assert updated.model_names == ("har",)


def test_prediction_panel_accepts_multiple_models() -> None:
    dates, ds = make_dataset()

    folds = make_expanding_folds(
        ds,
        first_cutoff=dates[90],
        refit_every=21,
    )

    panel = make_prediction_panel(ds, folds)

    panel = panel.with_forecast(
        name="naive_ma22",
        forecast=pd.Series(
            0.00008,
            index=panel.frame.index,
        ),
    )

    panel = panel.with_forecast(
        name="har",
        forecast=pd.Series(
            0.00007,
            index=panel.frame.index,
        ),
    )

    assert panel.model_names == (
        "naive_ma22",
        "har",
    )


def test_prediction_panel_rejects_misaligned_forecast() -> None:
    dates, ds = make_dataset()

    folds = make_expanding_folds(
        ds,
        first_cutoff=dates[90],
        refit_every=21,
    )

    panel = make_prediction_panel(ds, folds)

    forecast = pd.Series(
        0.00010,
        index=panel.frame.index[:-1],
    )

    with pytest.raises(
        ValueError,
        match="forecast index must exactly match",
    ):
        panel.with_forecast(
            name="har",
            forecast=forecast,
        )


def test_prediction_panel_rejects_nonpositive_forecast() -> None:
    dates, ds = make_dataset()

    folds = make_expanding_folds(
        ds,
        first_cutoff=dates[90],
        refit_every=21,
    )

    panel = make_prediction_panel(ds, folds)

    forecast = pd.Series(
        0.00010,
        index=panel.frame.index,
    )
    forecast.iloc[0] = 0.0

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        panel.with_forecast(
            name="har",
            forecast=forecast,
        )


@pytest.mark.parametrize(
    "reserved_name",
    [
        "realized",
        "target_end",
        "fit_cutoff",
        "fold_id",
    ],
)
def test_prediction_panel_rejects_reserved_model_names(
    reserved_name: str,
) -> None:
    dates, ds = make_dataset()

    folds = make_expanding_folds(
        ds,
        first_cutoff=dates[90],
        refit_every=21,
    )

    panel = make_prediction_panel(ds, folds)

    forecast = pd.Series(
        0.00010,
        index=panel.frame.index,
    )

    with pytest.raises(
        ValueError,
        match="reserved",
    ):
        panel.with_forecast(
            name=reserved_name,
            forecast=forecast,
        )
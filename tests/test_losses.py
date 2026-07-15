from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.analytics.losses import (
    loss_frame,
    mse_loss,
    qlike_loss,
)
from archer.analytics.panel import PredictionPanel

from typing import Any


def test_qlike_is_zero_when_forecast_equals_realized() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)

    realized = pd.Series(
        [0.0001, 0.0002, 0.0003],
        index=idx,
        name="realized",
    )

    forecast = realized.copy()
    forecast.name = "har"

    loss = qlike_loss(
        realized=realized,
        forecast=forecast,
    )

    assert np.allclose(loss.to_numpy(), 0.0)
    assert loss.index.equals(idx)
    assert loss.name == "har"


def test_qlike_is_positive_when_forecast_differs_from_realized() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)

    realized = pd.Series(
        [0.0001, 0.0002, 0.0003],
        index=idx,
        name="realized",
    )

    forecast = pd.Series(
        [0.0002, 0.0001, 0.0004],
        index=idx,
        name="har",
    )

    loss = qlike_loss(
        realized=realized,
        forecast=forecast,
    )

    assert (loss > 0.0).all()


def test_qlike_penalizes_underprediction_more_than_equal_overprediction() -> None:
    idx = pd.bdate_range("2020-01-01", periods=1)

    realized = pd.Series(
        [0.0001],
        index=idx,
        name="realized",
    )

    under_forecast = pd.Series(
        [0.00005],
        index=idx,
        name="under",
    )

    over_forecast = pd.Series(
        [0.00020],
        index=idx,
        name="over",
    )

    under_loss = qlike_loss(
        realized=realized,
        forecast=under_forecast,
    )

    over_loss = qlike_loss(
        realized=realized,
        forecast=over_forecast,
    )

    assert under_loss.iloc[0] > over_loss.iloc[0]


def test_mse_loss_matches_squared_error() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)

    realized = pd.Series(
        [1.0, 2.0, 3.0],
        index=idx,
        name="realized",
    )

    forecast = pd.Series(
        [1.5, 1.0, 5.0],
        index=idx,
        name="har",
    )

    loss = mse_loss(
        realized=realized,
        forecast=forecast,
    )

    expected = pd.Series(
        [0.25, 1.0, 4.0],
        index=idx,
        name="har",
    )

    assert loss.equals(expected)


def test_losses_reject_misaligned_indexes() -> None:
    realized = pd.Series(
        [0.0001, 0.0002],
        index=pd.bdate_range("2020-01-01", periods=2),
        name="realized",
    )

    forecast = pd.Series(
        [0.0001, 0.0002],
        index=pd.bdate_range("2020-01-02", periods=2),
        name="har",
    )

    with pytest.raises(
        ValueError,
        match="indexes must match",
    ):
        qlike_loss(
            realized=realized,
            forecast=forecast,
        )


@pytest.mark.parametrize(
    "bad_realized,bad_forecast",
    [
        ([0.0, 0.0002], [0.0001, 0.0002]),
        ([0.0001, -0.0002], [0.0001, 0.0002]),
        ([0.0001, 0.0002], [0.0, 0.0002]),
        ([0.0001, 0.0002], [0.0001, -0.0002]),
    ],
)
def test_qlike_rejects_nonpositive_values(
    bad_realized: list[float],
    bad_forecast: list[float],
) -> None:
    idx = pd.bdate_range("2020-01-01", periods=2)

    realized = pd.Series(
        bad_realized,
        index=idx,
        name="realized",
    )

    forecast = pd.Series(
        bad_forecast,
        index=idx,
        name="har",
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        qlike_loss(
            realized=realized,
            forecast=forecast,
        )


def test_loss_frame_computes_one_column_per_model() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)

    frame = pd.DataFrame(
        {
            "realized": [0.0001, 0.0002, 0.0003],
            "target_end": idx + pd.offsets.BDay(21),
            "fit_cutoff": pd.Timestamp("2019-12-31"),
            "fold_id": [0, 0, 0],
            "naive_ma22": [0.0001, 0.0001, 0.0003],
            "har": [0.0001, 0.0002, 0.0002],
        },
        index=idx,
    )
    frame.index.name = "forecast_origin"

    panel = PredictionPanel(
        frame=frame,
        horizon=21,
    )

    losses = loss_frame(
        panel,
        loss="mse",
    )

    assert list(losses.columns) == [
        "naive_ma22",
        "har",
    ]

    assert losses.index.equals(panel.frame.index)

    assert losses["naive_ma22"].iloc[0] == 0.0
    assert losses["har"].iloc[0] == 0.0


def test_loss_frame_rejects_unknown_loss() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)

    frame = pd.DataFrame(
        {
            "realized": [0.0001, 0.0002, 0.0003],
            "target_end": idx + pd.offsets.BDay(21),
            "fit_cutoff": pd.Timestamp("2019-12-31"),
            "fold_id": [0, 0, 0],
            "har": [0.0001, 0.0002, 0.0002],
        },
        index=idx,
    )
    frame.index.name = "forecast_origin"

    panel = PredictionPanel(
        frame=frame,
        horizon=21,
    )

    invalid_loss : Any = 'mae'

    with pytest.raises(
        ValueError,
        match="unknown loss",
    ):
        loss_frame(
            panel,
            loss=invalid_loss,
        )
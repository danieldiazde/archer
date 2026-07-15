from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .panel import PredictionPanel


LossName = Literal["qlike", "mse"]


def qlike_loss(
    *,
    realized: pd.Series,
    forecast: pd.Series,
) -> pd.Series:
    """
    Compute normalized QLIKE loss in variance units.

        L(v_hat, v) = v / v_hat - log(v / v_hat) - 1

    The loss is zero when forecast equals realized and positive otherwise.

    Both inputs must be strictly positive variance series.
    """
    realized_values, forecast_values = _validate_pair(
        realized=realized,
        forecast=forecast,
        require_positive=True,
    )

    ratio = realized_values / forecast_values

    loss = ratio - np.log(ratio) - 1.0

    return pd.Series(
        loss,
        index=realized.index,
        name=forecast.name,
    )


def mse_loss(
    *,
    realized: pd.Series,
    forecast: pd.Series,
) -> pd.Series:
    """
    Compute squared-error loss in variance units.
    """
    realized_values, forecast_values = _validate_pair(
        realized=realized,
        forecast=forecast,
        require_positive=False,
    )

    loss = (realized_values - forecast_values) ** 2

    return pd.Series(
        loss,
        index=realized.index,
        name=forecast.name,
    )


def loss_frame(
    panel: PredictionPanel,
    *,
    loss: LossName,
) -> pd.DataFrame:
    """
    Compute one loss column per model forecast in a PredictionPanel.
    """
    if loss == "qlike":
        loss_fn = qlike_loss
    elif loss == "mse":
        loss_fn = mse_loss
    else:
        raise ValueError(f"unknown loss: {loss!r}")

    if not panel.model_names:
        raise ValueError("prediction panel contains no model forecasts.")

    realized = panel.frame["realized"].rename("realized")

    losses = {
        model_name: loss_fn(
            realized=realized,
            forecast=panel.frame[model_name].rename(model_name),
        )
        for model_name in panel.model_names
    }

    return pd.DataFrame(
        losses,
        index=panel.frame.index,
    )


def _validate_pair(
    *,
    realized: pd.Series,
    forecast: pd.Series,
    require_positive: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(realized, pd.Series):
        raise TypeError("realized must be a pandas Series.")

    if not isinstance(forecast, pd.Series):
        raise TypeError("forecast must be a pandas Series.")

    if not realized.index.equals(forecast.index):
        raise ValueError("realized and forecast indexes must match.")

    if not pd.api.types.is_numeric_dtype(realized):
        raise ValueError("realized values must be numeric.")

    if not pd.api.types.is_numeric_dtype(forecast):
        raise ValueError("forecast values must be numeric.")

    realized_values = realized.astype(float).to_numpy()
    forecast_values = forecast.astype(float).to_numpy()

    if np.isnan(realized_values).any() or np.isnan(forecast_values).any():
        raise ValueError("realized and forecast must not contain missing values.")

    if (
        not np.isfinite(realized_values).all()
        or not np.isfinite(forecast_values).all()
    ):
        raise ValueError("realized and forecast must contain finite values.")

    if require_positive and (
        (realized_values <= 0.0).any()
        or (forecast_values <= 0.0).any()
    ):
        raise ValueError(
            "realized and forecast values must be strictly positive."
        )

    return realized_values, forecast_values
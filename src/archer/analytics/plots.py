from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .panel import PredictionPanel


TRADING_DAYS = 252


def plot_forecasts_vs_realized(
    panel: PredictionPanel,
    *,
    model_names: Sequence[str] | None = None,
    trading_days: int = TRADING_DAYS,
) -> Figure:
    """
    Plot realized and forecast annualized volatility.

    The PredictionPanel stores daily variance. For readability, this
    function converts each series to annualized volatility:

        annualized_volatility = sqrt(daily_variance * trading_days)
    """
    if trading_days <= 0:
        raise ValueError(
            "trading_days must be strictly positive."
        )

    selected_models = _resolve_model_names(
        panel,
        model_names=model_names,
    )

    realized_vol = (
        panel.frame["realized"]
        * trading_days
    ).pow(0.5)

    figure, axis = plt.subplots(
        figsize=(12, 6),
    )

    axis.plot(
        panel.frame.index,
        realized_vol,
        label="realized",
        linewidth=2.0,
    )

    for model_name in selected_models:
        forecast_vol = (
            panel.frame[model_name]
            * trading_days
        ).pow(0.5)

        axis.plot(
            panel.frame.index,
            forecast_vol,
            label=model_name,
            linewidth=1.2,
            alpha=0.85,
        )

    axis.set_title(
        "Out-of-sample volatility forecasts"
    )

    axis.set_xlabel(
        "Forecast origin"
    )

    axis.set_ylabel(
        "Annualized volatility"
    )

    axis.legend()
    axis.grid(
        visible=True,
        alpha=0.25,
    )

    figure.tight_layout()

    return figure


def plot_vix_variance_spread(
    panel: PredictionPanel,
    *,
    vix_model_name: str = "vix_implied",
    rolling_window: int = 21,
) -> Figure:
    """
    Plot VIX-implied daily variance minus subsequent realized variance.

    A positive value means implied variance exceeded the subsequently
    realized average daily variance for that forecast origin.

    Both the raw spread and its trailing rolling average are displayed.
    """
    if rolling_window < 1:
        raise ValueError(
            "rolling_window must be at least 1."
        )

    if vix_model_name not in panel.model_names:
        raise ValueError(
            f"VIX model {vix_model_name!r} is not in the prediction panel."
        )

    spread = (
        panel.frame[vix_model_name]
        - panel.frame["realized"]
    ).rename("vix_variance_spread")

    rolling_spread = spread.rolling(
        window=rolling_window,
        min_periods=rolling_window,
    ).mean()

    figure, axis = plt.subplots(
        figsize=(12, 6),
    )

    axis.plot(
        panel.frame.index,
        spread,
        label="daily spread",
        linewidth=0.8,
        alpha=0.45,
    )

    axis.plot(
        panel.frame.index,
        rolling_spread,
        label=f"{rolling_window}-day rolling mean",
        linewidth=2.0,
    )

    axis.axhline(
        y=0.0,
        linewidth=1.0,
        linestyle="--",
    )

    axis.set_title(
        "VIX-implied minus subsequently realized variance"
    )

    axis.set_xlabel(
        "Forecast origin"
    )

    axis.set_ylabel(
        "Daily variance spread"
    )

    axis.legend()
    axis.grid(
        visible=True,
        alpha=0.25,
    )

    figure.tight_layout()

    return figure


def _resolve_model_names(
    panel: PredictionPanel,
    *,
    model_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if model_names is None:
        selected = panel.model_names
    else:
        selected = tuple(model_names)

    if not selected:
        raise ValueError(
            "at least one forecast model must be selected."
        )

    if len(set(selected)) != len(selected):
        raise ValueError(
            "model_names cannot contain duplicates."
        )

    missing = [
        name
        for name in selected
        if name not in panel.model_names
    ]

    if missing:
        raise ValueError(
            f"prediction panel does not contain models: {missing}"
        )

    return selected
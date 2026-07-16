from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from archer.analytics.panel import PredictionPanel
from archer.analytics.plots import (
    plot_forecasts_vs_realized,
    plot_vix_variance_spread,
)


def make_panel() -> PredictionPanel:
    idx = pd.bdate_range(
        "2020-01-01",
        periods=300,
    )

    realized = pd.Series(
        np.linspace(
            0.00005,
            0.00030,
            len(idx),
        ),
        index=idx,
    )

    frame = pd.DataFrame(
        {
            "realized": realized,
            "target_end": idx + pd.offsets.BDay(21),
            "fit_cutoff": pd.Timestamp("2019-12-31"),
            "fold_id": 0,
            "naive_ma22": realized * 1.05,
            "har": realized * 0.98,
            "garch": realized * 1.02,
            "vix_implied": realized * 1.20,
        },
        index=idx,
    )

    frame.index.name = "forecast_origin"

    return PredictionPanel(
        frame=frame,
        horizon=21,
    )


def test_plot_forecasts_vs_realized_returns_figure() -> None:
    panel = make_panel()

    figure = plot_forecasts_vs_realized(panel)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1

    axis = figure.axes[0]

    assert axis.get_title() == (
        "Out-of-sample volatility forecasts"
    )

    assert axis.get_ylabel() == (
        "Annualized volatility"
    )

    # Realized plus four models.
    assert len(axis.lines) == 5


def test_plot_forecasts_vs_realized_can_limit_models() -> None:
    panel = make_panel()

    figure = plot_forecasts_vs_realized(
        panel,
        model_names=[
            "har",
            "vix_implied",
        ],
    )

    axis = figure.axes[0]

    # Realized plus two selected models.
    assert len(axis.lines) == 3


def test_plot_vix_variance_spread_returns_figure() -> None:
    panel = make_panel()

    figure = plot_vix_variance_spread(
        panel,
        rolling_window=21,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1

    axis = figure.axes[0]

    assert axis.get_title() == (
        "VIX-implied minus subsequently realized variance"
    )

    assert axis.get_ylabel() == (
        "Daily variance spread"
    )

    # Raw spread, rolling mean, and horizontal zero line.
    assert len(axis.lines) == 3


def test_vix_spread_is_positive_for_upward_biased_vix() -> None:
    panel = make_panel()

    spread = (
        panel.frame["vix_implied"]
        - panel.frame["realized"]
    )

    assert (spread > 0.0).all()
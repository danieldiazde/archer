from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from archer.analytics.panel import PredictionPanel
from archer.analytics.report import (
    ForecastReport,
    build_forecast_report,
    summarize_losses,
    summarize_losses_by_year,
)


def make_panel() -> PredictionPanel:
    rng = np.random.default_rng(42)

    idx = pd.bdate_range(
        "2020-01-01",
        periods=500,
    )

    realized = pd.Series(
        rng.uniform(
            0.00005,
            0.00030,
            size=len(idx),
        ),
        index=idx,
    )

    frame = pd.DataFrame(
        {
            "realized": realized,
            "target_end": idx + pd.offsets.BDay(21),
            "fit_cutoff": pd.Timestamp("2019-12-31"),
            "fold_id": 0,
            "naive_ma22": (
                realized
                + rng.normal(
                    0.0,
                    0.00004,
                    size=len(idx),
                )
            ).clip(lower=1e-8),
            "har": (
                realized
                + rng.normal(
                    0.0,
                    0.00001,
                    size=len(idx),
                )
            ).clip(lower=1e-8),
            "garch": (
                realized
                + rng.normal(
                    0.0,
                    0.00002,
                    size=len(idx),
                )
            ).clip(lower=1e-8),
        },
        index=idx,
    )

    frame.index.name = "forecast_origin"

    return PredictionPanel(
        frame=frame,
        horizon=21,
    )


def test_summarize_losses_compares_models_to_baseline() -> None:
    idx = pd.bdate_range(
        "2020-01-01",
        periods=3,
    )

    losses = pd.DataFrame(
        {
            "naive_ma22": [2.0, 4.0, 6.0],
            "har": [1.0, 2.0, 3.0],
        },
        index=idx,
    )

    summary = summarize_losses(
        losses,
        baseline="naive_ma22",
    )

    assert list(summary.index) == [
        "har",
        "naive_ma22",
    ]

    assert summary.loc["naive_ma22", "mean_loss"] == 4.0
    assert summary.loc["har", "mean_loss"] == 2.0

    assert (
        summary.loc["har", "relative_to_baseline"]
        == 0.5
    )

    assert (
        summary.loc["har", "improvement_vs_baseline"]
        == 0.5
    )


def test_summarize_losses_by_year() -> None:
    idx = pd.to_datetime(
        [
            "2020-01-02",
            "2020-02-03",
            "2021-01-04",
            "2021-02-01",
        ]
    )

    losses = pd.DataFrame(
        {
            "naive_ma22": [2.0, 4.0, 6.0, 10.0],
            "har": [1.0, 2.0, 3.0, 5.0],
        },
        index=idx,
    )

    summary = summarize_losses_by_year(
        losses,
        baseline="naive_ma22",
    )

    assert list(summary.index) == [
        2020,
        2021,
    ]

    assert (
        summary.loc[
            2020,
            ("mean_loss", "naive_ma22"),
        ]
        == 3.0
    )

    assert (
        summary.loc[
            2020,
            ("mean_loss", "har"),
        ]
        == 1.5
    )

    assert (
        summary.loc[
            2020,
            ("relative_to_baseline", "har"),
        ]
        == 0.5
    )

    assert (
        summary.loc[
            2020,
            ("improvement_vs_baseline", "har"),
        ]
        == 0.5
    )


def test_build_forecast_report_produces_all_tables() -> None:
    panel = make_panel()

    report = build_forecast_report(
        panel,
        baseline="naive_ma22",
    )

    assert isinstance(report, ForecastReport)

    assert list(report.overall_qlike.index) == [
        "har",
        "garch",
        "naive_ma22",
    ]

    assert set(report.overall_mse.index) == {
        "naive_ma22",
        "har",
        "garch",
    }

    assert set(report.mz.index) == {
        "naive_ma22",
        "har",
        "garch",
    }

    expected_pairs = {
        ("naive_ma22", "har"),
        ("naive_ma22", "garch"),
        ("har", "garch"),
    }

    assert set(report.dm_qlike.index) == expected_pairs
    assert set(report.dm_mse.index) == expected_pairs

    assert not report.qlike_by_year.empty
    assert not report.mse_by_year.empty


def test_build_forecast_report_uses_horizon_for_inference() -> None:
    panel = make_panel()

    report = build_forecast_report(
        panel,
        baseline="naive_ma22",
    )

    assert (report.mz["maxlags"] == 20).all()
    assert (report.dm_qlike["maxlags"] == 20).all()
    assert (report.dm_mse["maxlags"] == 20).all()


def test_report_rejects_missing_baseline() -> None:
    panel = make_panel()

    with pytest.raises(
        ValueError,
        match="baseline",
    ):
        build_forecast_report(
            panel,
            baseline="missing_model",
        )
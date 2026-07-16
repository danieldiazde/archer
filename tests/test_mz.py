from __future__ import annotations

import numpy as np
import pandas as pd

from archer.analytics.mz import (
    mincer_zarnowitz,
    mz_table,
)
from archer.analytics.panel import PredictionPanel


def test_mincer_zarnowitz_recovers_unbiased_forecast() -> None:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=1_000)

    forecast = pd.Series(
        rng.uniform(0.00005, 0.00030, size=len(idx)),
        index=idx,
        name="unbiased",
    )

    raw_noise = rng.normal(
        loc=0.0,
        scale=0.000005,
        size=len(idx),
    )

    design = np.column_stack(
        [
            np.ones(len(idx)),
            forecast.to_numpy(),
        ]
    )

    projection_coefficients = np.linalg.lstsq(
        design,
        raw_noise,
        rcond=None,
    )[0]

    orthogonal_noise = (
        raw_noise
        - design @ projection_coefficients
    )

    realized = pd.Series(
        forecast.to_numpy() + orthogonal_noise,
        index=idx,
        name="realized",
    )

    result = mincer_zarnowitz(
        realized=realized,
        forecast=forecast,
        maxlags=20,
    )

    assert abs(result.intercept) < 0.00001
    assert abs(result.slope - 1.0) < 0.05
    assert result.r_squared > 0.90

    # The joint null (intercept, slope) = (0, 1) should not be rejected.
    assert result.wald_p_value > 0.05


def test_mincer_zarnowitz_rejects_biased_forecast() -> None:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=1_000)

    forecast = pd.Series(
        rng.uniform(0.00005, 0.00030, size=len(idx)),
        index=idx,
        name="biased",
    )

    noise = rng.normal(
        loc=0.0,
        scale=0.000002,
        size=len(idx),
    )

    realized = pd.Series(
        0.00003 + 0.60 * forecast.to_numpy() + noise,
        index=idx,
        name="realized",
    )

    result = mincer_zarnowitz(
        realized=realized,
        forecast=forecast,
        maxlags=20,
    )

    assert abs(result.intercept - 0.00003) < 0.00001
    assert abs(result.slope - 0.60) < 0.05
    assert result.wald_p_value < 0.001


def test_mincer_zarnowitz_uses_requested_hac_lags() -> None:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=300)

    forecast = pd.Series(
        rng.uniform(0.00005, 0.00030, size=len(idx)),
        index=idx,
        name="har",
    )

    realized = pd.Series(
        forecast.to_numpy()
        + rng.normal(0.0, 0.00001, size=len(idx)),
        index=idx,
        name="realized",
    )

    result = mincer_zarnowitz(
        realized=realized,
        forecast=forecast,
        maxlags=7,
    )

    assert result.maxlags == 7


def test_mz_table_returns_one_row_per_model() -> None:
    idx = pd.bdate_range("2020-01-01", periods=300)

    rng = np.random.default_rng(42)

    realized = rng.uniform(
        0.00005,
        0.00030,
        size=len(idx),
    )

    frame = pd.DataFrame(
        {
            "realized": realized,
            "target_end": idx + pd.offsets.BDay(21),
            "fit_cutoff": pd.Timestamp("2019-12-31"),
            "fold_id": 0,
            "har": realized + rng.normal(
                0.0,
                0.00001,
                size=len(idx),
            ),
            "vix_implied": 0.00005 + 1.30 * realized,
        },
        index=idx,
    )
    frame.index.name = "forecast_origin"

    panel = PredictionPanel(
        frame=frame,
        horizon=21,
    )

    table = mz_table(panel)

    assert list(table.index) == [
        "har",
        "vix_implied",
    ]

    assert list(table.columns) == [
        "intercept",
        "slope",
        "r_squared",
        "wald_statistic",
        "wald_p_value",
        "maxlags",
        "nobs",
    ]

    assert (table["maxlags"] == 20).all()
    assert (table["nobs"] == len(idx)).all()
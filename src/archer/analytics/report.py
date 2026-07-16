from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dm import pairwise_dm_table
from .losses import loss_frame
from .mz import mz_table
from .panel import PredictionPanel


@dataclass(frozen=True, slots=True)
class ForecastReport:
    """
    Complete statistical evaluation of a PredictionPanel.

    overall_qlike:
        Mean and median QLIKE loss by model, including performance
        relative to the selected baseline.

    overall_mse:
        Mean and median squared-error loss by model.

    qlike_by_year:
        Annual QLIKE results.

    mse_by_year:
        Annual MSE results.

    mz:
        Mincer-Zarnowitz calibration results.

    dm_qlike:
        Pairwise Diebold-Mariano tests using QLIKE losses.

    dm_mse:
        Pairwise Diebold-Mariano tests using MSE losses.
    """

    overall_qlike: pd.DataFrame
    overall_mse: pd.DataFrame
    qlike_by_year: pd.DataFrame
    mse_by_year: pd.DataFrame
    mz: pd.DataFrame
    dm_qlike: pd.DataFrame
    dm_mse: pd.DataFrame


def build_forecast_report(
    panel: PredictionPanel,
    *,
    baseline: str,
) -> ForecastReport:
    """
    Build all forecast-evaluation tables from one PredictionPanel.

    Every table is derived from the same persisted out-of-sample forecasts,
    ensuring that losses, calibration tests, and pairwise comparisons use
    identical dates and observations.
    """
    if baseline not in panel.model_names:
        raise ValueError(
            f"baseline {baseline!r} is not a model in the prediction panel."
        )

    qlike = loss_frame(
        panel,
        loss="qlike",
    )

    mse = loss_frame(
        panel,
        loss="mse",
    )

    return ForecastReport(
        overall_qlike=summarize_losses(
            qlike,
            baseline=baseline,
        ),
        overall_mse=summarize_losses(
            mse,
            baseline=baseline,
        ),
        qlike_by_year=summarize_losses_by_year(
            qlike,
            baseline=baseline,
        ),
        mse_by_year=summarize_losses_by_year(
            mse,
            baseline=baseline,
        ),
        mz=mz_table(panel),
        dm_qlike=pairwise_dm_table(
            qlike,
            horizon=panel.horizon,
        ),
        dm_mse=pairwise_dm_table(
            mse,
            horizon=panel.horizon,
        ),
    )


def summarize_losses(
    losses: pd.DataFrame,
    *,
    baseline: str,
) -> pd.DataFrame:
    """
    Summarize mean and median losses relative to a baseline model.

    Definitions:

        relative_to_baseline =
            model mean loss / baseline mean loss

        improvement_vs_baseline =
            1 - relative_to_baseline

    Therefore:

        relative_to_baseline < 1
            model beats the baseline.

        improvement_vs_baseline > 0
            model improves upon the baseline.
    """
    _validate_losses(
        losses,
        baseline=baseline,
    )

    mean_loss = losses.mean()
    median_loss = losses.median()

    baseline_mean = float(mean_loss.loc[baseline])

    if baseline_mean <= 0.0:
        raise ValueError(
            "baseline mean loss must be strictly positive."
        )

    summary = pd.DataFrame(
        {
            "mean_loss": mean_loss,
            "median_loss": median_loss,
        }
    )

    summary["relative_to_baseline"] = (
        summary["mean_loss"]
        / baseline_mean
    )

    summary["improvement_vs_baseline"] = (
        1.0
        - summary["relative_to_baseline"]
    )

    return summary.sort_values(
        "mean_loss",
        ascending=True,
    )


def summarize_losses_by_year(
    losses: pd.DataFrame,
    *,
    baseline: str,
) -> pd.DataFrame:
    """
    Summarize average model losses separately for each calendar year.
    """
    _validate_losses(
        losses,
        baseline=baseline,
    )

    datetime_index = pd.DatetimeIndex(
        pd.to_datetime(losses.index)
    )

    yearly = losses.groupby(
        datetime_index.year,
    ).mean()

    yearly.index.name = "forecast_origin"

    baseline_yearly = yearly[baseline]

    if (baseline_yearly <= 0.0).any():
        raise ValueError(
            "annual baseline mean losses must be strictly positive."
        )

    relative = yearly.div(
        baseline_yearly,
        axis=0,
    )

    improvement = (
        1.0
        - relative
    )

    return pd.concat(
        {
            "mean_loss": yearly,
            "relative_to_baseline": relative,
            "improvement_vs_baseline": improvement,
        },
        axis=1,
    )


def _validate_losses(
    losses: pd.DataFrame,
    *,
    baseline: str,
) -> None:
    if not isinstance(losses, pd.DataFrame):
        raise TypeError(
            "losses must be a pandas DataFrame."
        )

    if losses.empty:
        raise ValueError(
            "losses cannot be empty."
        )

    if losses.columns.has_duplicates:
        raise ValueError(
            "loss columns must contain unique model names."
        )

    if baseline not in losses.columns:
        raise ValueError(
            f"baseline {baseline!r} is not present in losses."
        )

    if losses.index.has_duplicates:
        raise ValueError(
            "loss index contains duplicate forecast origins."
        )

    if not losses.index.is_monotonic_increasing:
        raise ValueError(
            "loss index must be sorted chronologically."
        )

    non_numeric = [
        column
        for column in losses.columns
        if not pd.api.types.is_numeric_dtype(
            losses[column]
        )
    ]

    if non_numeric:
        raise ValueError(
            f"loss columns must be numeric: {non_numeric}"
        )

    values = losses.to_numpy(
        dtype=float,
    )

    if np.isnan(values).any():
        raise ValueError(
            "losses cannot contain missing values."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "losses must contain finite values."
        )

    if (values < 0.0).any():
        raise ValueError(
            "losses cannot contain negative values."
        )
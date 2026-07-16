from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t


@dataclass(frozen=True, slots=True)
class DMResult:
    """
    Diebold-Mariano forecast-comparison result.

    The loss differential is defined as:

        d_t = loss_a(t) - loss_b(t)

    Therefore:

        mean_loss_difference < 0:
            model A has lower average loss.

        mean_loss_difference > 0:
            model B has lower average loss.
    """

    mean_loss_difference: float
    statistic: float
    p_value: float
    maxlags: int
    nobs: int


def diebold_mariano(
    *,
    loss_a: pd.Series,
    loss_b: pd.Series,
    horizon: int,
) -> DMResult:
    """
    Compare two forecast-loss series using the Diebold-Mariano test.

    The long-run variance of the loss differential is estimated with a
    Newey-West/Bartlett HAC estimator using:

        maxlags = horizon - 1

    The statistic also receives the Harvey-Leybourne-Newbold finite-sample
    correction for overlapping h-step forecasts.

    The returned p-value is two-sided.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    a, b = _validate_losses(
        loss_a=loss_a,
        loss_b=loss_b,
    )

    differential = a - b
    nobs = len(differential)

    if nobs < 2:
        raise ValueError(
            "Diebold-Mariano test requires at least 2 observations."
        )

    mean_difference = float(differential.mean())

    # Exact ties need an explicit policy because both the numerator and
    # estimated variance are zero.
    if np.allclose(
        differential.to_numpy(),
        0.0,
        rtol=0.0,
        atol=0.0,
    ):
        return DMResult(
            mean_loss_difference=0.0,
            statistic=0.0,
            p_value=1.0,
            maxlags=min(horizon - 1, nobs - 1),
            nobs=nobs,
        )

    maxlags = min(
        horizon - 1,
        nobs - 1,
    )

    long_run_variance = _newey_west_long_run_variance(
        differential.to_numpy(dtype=float),
        maxlags=maxlags,
    )

    if not np.isfinite(long_run_variance):
        raise RuntimeError(
            "Diebold-Mariano long-run variance is non-finite."
        )

    if long_run_variance <= 0.0:
        raise RuntimeError(
            "Diebold-Mariano long-run variance must be positive."
        )

    standard_error = np.sqrt(
        long_run_variance / nobs
    )

    raw_statistic = mean_difference / standard_error

    correction = _hln_correction(
        nobs=nobs,
        horizon=horizon,
    )

    statistic = float(
        raw_statistic * correction
    )

    p_value = float(
        2.0
        * t.sf(
            abs(statistic),
            df=nobs - 1,
        )
    )

    return DMResult(
        mean_loss_difference=mean_difference,
        statistic=statistic,
        p_value=p_value,
        maxlags=maxlags,
        nobs=nobs,
    )


def pairwise_dm_table(
    losses: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    """
    Run Diebold-Mariano tests for every unique model pair.

    Rows are ordered according to the model-column order in ``losses``.
    """
    if not isinstance(losses, pd.DataFrame):
        raise TypeError("losses must be a pandas DataFrame.")

    if losses.empty:
        raise ValueError("losses cannot be empty.")

    if losses.shape[1] < 2:
        raise ValueError(
            "pairwise DM tests require at least two models."
        )

    if losses.columns.has_duplicates:
        raise ValueError(
            "loss columns must have unique model names."
        )

    rows: list[dict[str, float | int]] = []
    pairs: list[tuple[str, str]] = []

    for model_a, model_b in combinations(
        losses.columns,
        2,
    ):
        result = diebold_mariano(
            loss_a=losses[model_a].rename(model_a),
            loss_b=losses[model_b].rename(model_b),
            horizon=horizon,
        )

        pairs.append(
            (
                str(model_a),
                str(model_b),
            )
        )

        rows.append(
            {
                "mean_loss_difference": (
                    result.mean_loss_difference
                ),
                "statistic": result.statistic,
                "p_value": result.p_value,
                "maxlags": result.maxlags,
                "nobs": result.nobs,
            }
        )

    index = pd.MultiIndex.from_tuples(
        pairs,
        names=[
            "model_a",
            "model_b",
        ],
    )

    return pd.DataFrame(
        rows,
        index=index,
    )


def _newey_west_long_run_variance(
    values: np.ndarray,
    *,
    maxlags: int,
) -> float:
    """
    Estimate the long-run variance with Bartlett/Newey-West weights.

    For centered loss differentials x_t:

        LRV =
            gamma_0
            + 2 * sum(
                weight_lag * gamma_lag
            )

        weight_lag =
            1 - lag / (maxlags + 1)
    """
    if values.ndim != 1:
        raise ValueError(
            "Newey-West input must be one-dimensional."
        )

    nobs = len(values)

    if nobs < 2:
        raise ValueError(
            "Newey-West estimation requires at least 2 observations."
        )

    if maxlags < 0:
        raise ValueError("maxlags must be nonnegative.")

    if maxlags >= nobs:
        raise ValueError(
            "maxlags must be smaller than the number of observations."
        )

    centered = values - values.mean()

    gamma_zero = float(
        np.dot(
            centered,
            centered,
        )
        / nobs
    )

    long_run_variance = gamma_zero

    for lag in range(
        1,
        maxlags + 1,
    ):
        autocovariance = float(
            np.dot(
                centered[lag:],
                centered[:-lag],
            )
            / nobs
        )

        weight = (
            1.0
            - lag / (maxlags + 1.0)
        )

        long_run_variance += (
            2.0
            * weight
            * autocovariance
        )

    return float(long_run_variance)


def _hln_correction(
    *,
    nobs: int,
    horizon: int,
) -> float:
    """
    Harvey-Leybourne-Newbold finite-sample correction.

    The correction factor is:

        sqrt(
            (
                n + 1
                - 2h
                + h(h - 1) / n
            )
            / n
        )
    """
    correction_squared = (
        nobs
        + 1
        - 2 * horizon
        + horizon * (horizon - 1) / nobs
    ) / nobs

    if correction_squared <= 0.0:
        raise ValueError(
            "sample is too short for the requested forecast horizon."
        )

    return float(
        np.sqrt(correction_squared)
    )


def _validate_losses(
    *,
    loss_a: pd.Series,
    loss_b: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    if not isinstance(loss_a, pd.Series):
        raise TypeError(
            "loss_a must be a pandas Series."
        )

    if not isinstance(loss_b, pd.Series):
        raise TypeError(
            "loss_b must be a pandas Series."
        )

    if not loss_a.index.equals(loss_b.index):
        raise ValueError(
            "loss indexes must match."
        )

    if not pd.api.types.is_numeric_dtype(loss_a):
        raise ValueError(
            "loss_a values must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(loss_b):
        raise ValueError(
            "loss_b values must be numeric."
        )

    a = loss_a.astype(float)
    b = loss_b.astype(float)

    if a.isna().any() or b.isna().any():
        raise ValueError(
            "loss series cannot contain missing values."
        )

    if (
        not np.isfinite(a.to_numpy()).all()
        or not np.isfinite(b.to_numpy()).all()
    ):
        raise ValueError(
            "loss series must contain finite values."
        )

    return a, b
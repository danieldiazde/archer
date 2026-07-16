from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from .panel import PredictionPanel


@dataclass(frozen=True, slots=True)
class MZResult:
    """
    Result of a Mincer-Zarnowitz calibration regression.

    Regression:

        realized_t = intercept + slope * forecast_t + error_t

    An unbiased and correctly scaled forecast satisfies:

        intercept = 0
        slope = 1
    """

    intercept: float
    slope: float
    r_squared: float
    wald_statistic: float
    wald_p_value: float
    maxlags: int
    nobs: int


def mincer_zarnowitz(
    *,
    realized: pd.Series,
    forecast: pd.Series,
    maxlags: int,
) -> MZResult:
    """
    Run a Mincer-Zarnowitz calibration regression with HAC covariance.

    The joint Wald test evaluates:

        H0: intercept = 0 and slope = 1

    HAC covariance is used because overlapping multi-period volatility
    targets produce serially correlated regression errors.
    """
    if maxlags < 0:
        raise ValueError("maxlags must be nonnegative.")

    realized_values, forecast_values = _validate_inputs(
        realized=realized,
        forecast=forecast,
    )

    if len(realized_values) < 3:
        raise ValueError(
            "Mincer-Zarnowitz regression requires at least 3 observations."
        )

    effective_maxlags = min(
        maxlags,
        len(realized_values) - 1,
    )

    X = pd.DataFrame(
        {
            "forecast": forecast_values,
        },
        index=realized.index,
    )

    X = sm.add_constant(
        X,
        has_constant="add",
    )

    model = sm.OLS(
        realized_values,
        X,
        missing="raise",
    )

    result = model.fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": effective_maxlags,
        },
    )

    intercept = float(result.params["const"])
    slope = float(result.params["forecast"])

    # Joint restriction:
    #
    #     intercept = 0
    #     slope = 1
    #
    # The Wald statistic is:
    #
    #     d' Cov(beta)^(-1) d
    #
    # where d = [intercept, slope - 1].
    difference = np.array(
        [
            intercept,
            slope - 1.0,
        ],
        dtype=float,
    )

    covariance = np.asarray(
        result.cov_params(),
        dtype=float,
    )

    inverse_covariance = np.linalg.pinv(covariance)

    wald_statistic = float(
        difference.T
        @ inverse_covariance
        @ difference
    )

    wald_p_value = float(
        chi2.sf(
            wald_statistic,
            df=2,
        )
    )

    return MZResult(
        intercept=intercept,
        slope=slope,
        r_squared=float(result.rsquared),
        wald_statistic=wald_statistic,
        wald_p_value=wald_p_value,
        maxlags=effective_maxlags,
        nobs=int(result.nobs),
    )


def mz_table(
    panel: PredictionPanel,
) -> pd.DataFrame:
    """
    Run Mincer-Zarnowitz calibration tests for every model in a panel.

    The default HAC lag count is horizon - 1 because consecutive
    horizon-day targets overlap by that many observations.
    """
    if not panel.model_names:
        raise ValueError(
            "prediction panel contains no model forecasts."
        )

    maxlags = panel.horizon - 1
    realized = panel.frame["realized"].rename("realized")

    rows: dict[str, dict[str, float | int]] = {}

    for model_name in panel.model_names:
        result = mincer_zarnowitz(
            realized=realized,
            forecast=panel.frame[model_name].rename(model_name),
            maxlags=maxlags,
        )

        rows[model_name] = {
            "intercept": result.intercept,
            "slope": result.slope,
            "r_squared": result.r_squared,
            "wald_statistic": result.wald_statistic,
            "wald_p_value": result.wald_p_value,
            "maxlags": result.maxlags,
            "nobs": result.nobs,
        }

    return pd.DataFrame.from_dict(
        rows,
        orient="index",
    )


def _validate_inputs(
    *,
    realized: pd.Series,
    forecast: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    if not isinstance(realized, pd.Series):
        raise TypeError(
            "realized must be a pandas Series."
        )

    if not isinstance(forecast, pd.Series):
        raise TypeError(
            "forecast must be a pandas Series."
        )

    if not realized.index.equals(forecast.index):
        raise ValueError(
            "realized and forecast indexes must match."
        )

    if not pd.api.types.is_numeric_dtype(realized):
        raise ValueError(
            "realized values must be numeric."
        )

    if not pd.api.types.is_numeric_dtype(forecast):
        raise ValueError(
            "forecast values must be numeric."
        )

    realized_values = realized.astype(float)
    forecast_values = forecast.astype(float)

    if (
        realized_values.isna().any()
        or forecast_values.isna().any()
    ):
        raise ValueError(
            "realized and forecast cannot contain missing values."
        )

    if (
        not np.isfinite(realized_values.to_numpy()).all()
        or not np.isfinite(forecast_values.to_numpy()).all()
    ):
        raise ValueError(
            "realized and forecast must contain finite values."
        )

    if (
        (realized_values <= 0.0).any()
        or (forecast_values <= 0.0).any()
    ):
        raise ValueError(
            "realized and forecast must be strictly positive."
        )

    if forecast_values.nunique() < 2:
        raise ValueError(
            "forecast must contain variation for Mincer-Zarnowitz regression."
        )

    return realized_values, forecast_values
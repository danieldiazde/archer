from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import VolDataset
from .fold import ForecastFold


@dataclass(frozen=True, slots=True)
class NaiveMA22Forecaster:
    """
    Naive trailing-month variance forecaster.

    Forecast definition:

        y_hat_t = har_m(t)

    Since har_m is the trailing 22-day mean of daily variance, this is the
    simplest benchmark that says:

        "Next month's average variance will equal the last month's
        average variance."

    No parameters are estimated.
    """

    name: str = "naive_ma22"
    feature_name: str = "har_m"

    def fit(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> None:
        """
        No-op fit.

        The benchmark has no estimated parameters, but keeps the same
        interface as learned forecasters.
        """
        _ = ds
        _ = fold

    def predict(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> pd.Series:
        """
        Return the trailing-month HAR feature for the fold's test origins.
        """
        test_ds = fold.test_dataset(ds)

        if self.feature_name not in test_ds.X.columns:
            raise ValueError(
                f"dataset is missing required feature: {self.feature_name!r}"
            )

        forecast = test_ds.X[self.feature_name].astype(float)

        _validate_positive_forecast(
            forecast,
            name=self.name,
        )

        return forecast.rename(self.name)


@dataclass(frozen=True, slots=True)
class AlignedSeriesForecaster:
    """
    Adapter for externally constructed forecast series.

    Use this for forecasts that are already aligned to dataset dates, such
    as VIX-implied variance.

    The series must be in the same units as the target:

        daily variance
    """

    name: str
    series: pd.Series

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("forecast name cannot be empty.")

        if not isinstance(self.series, pd.Series):
            raise TypeError("series must be a pandas Series.")

        if self.series.index.has_duplicates:
            raise ValueError("series index contains duplicate dates.")

        if not pd.api.types.is_numeric_dtype(self.series):
            raise ValueError("series values must be numeric.")

        normalized = self.series.copy()
        normalized.index = pd.to_datetime(normalized.index)
        normalized = normalized.sort_index().astype(float)

        if normalized.isna().any():
            raise ValueError("series contains missing values.")

        if not np.isfinite(normalized.to_numpy()).all():
            raise ValueError("series contains non-finite values.")

        object.__setattr__(
            self,
            "series",
            normalized,
        )

    def fit(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> None:
        """
        No-op fit.

        The external forecast has already been constructed.
        """
        _ = ds
        _ = fold

    def predict(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> pd.Series:
        """
        Return the external forecast for the fold's test origins.
        """
        _ = ds

        missing = fold.test_idx.difference(self.series.index)

        if not missing.empty:
            raise ValueError(
                "aligned forecast series is missing required forecast origins: "
                f"{missing.tolist()}"
            )

        forecast = self.series.loc[fold.test_idx].astype(float)

        _validate_positive_forecast(
            forecast,
            name=self.name,
        )

        return forecast.rename(self.name)


def _validate_positive_forecast(
    forecast: pd.Series,
    *,
    name: str,
) -> None:
    if forecast.empty:
        raise ValueError(f"{name} forecast is empty.")

    if forecast.isna().any():
        raise ValueError(f"{name} forecast contains missing values.")

    if not np.isfinite(forecast.to_numpy()).all():
        raise ValueError(f"{name} forecast contains non-finite values.")

    if (forecast <= 0.0).any():
        raise ValueError(
            f"{name} forecast values must be strictly positive."
        )
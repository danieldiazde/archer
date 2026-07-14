from __future__ import annotations

from typing import Protocol

import pandas as pd

from .dataset import VolDataset
from .fold import ForecastFold


class Forecaster(Protocol):
    """
    Common interface for volatility forecasting models.

    Models receive the complete dataset plus an explicit forecasting fold.

    This allows different models to use the information available at the
    cutoff correctly:

        HAR:
            Fits on supervised rows whose target windows end by the cutoff.

        GARCH:
            Fits on returns observed through the cutoff.
    """

    name: str

    def fit(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> None:
        """
        Fit the model using only information available at the fold cutoff.
        """
        ...

    def predict(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> pd.Series:
        """
        Produce daily-variance forecasts for the fold's test origins.
        """
        ...
from __future__ import annotations

from typing import Protocol

import pandas as pd

from .dataset import VolDataset
from .fold import ForecastFold


class Forecaster(Protocol):
    @property
    def name(self) -> str:
        """
        Model name used as the forecast column name.
        """
        ...

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
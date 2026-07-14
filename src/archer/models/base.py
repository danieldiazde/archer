from __future__ import annotations

from typing import Protocol

import pandas as pd

from .dataset import VolDataset


class Forecaster(Protocol):
    """
    Common interface for volatility forecasting models.

    All forecasts must be daily variance predictions, indexed like ds.X.
    The evaluation harness can annualize or transform them later.
    """

    name: str

    def fit(self, ds: VolDataset) -> None:
        """
        Fit the model on a volatility dataset.
        """
        ...

    def predict(self, ds: VolDataset) -> pd.Series:
        """
        Predict daily variance for each row in ds.

        Returns:
            pd.Series indexed by ds.X.index.
        """
        ...
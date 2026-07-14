from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import statsmodels.api as sm

from .dataset import VolDataset


@dataclass
class HARForecaster:
    """
    HAR-RV forecaster.

    Fits:
        y_t = beta_0 + beta_d har_d(t) + beta_w har_w(t) + beta_m har_m(t) + error

    Forecasts are daily variance units, not annualized volatility.
    """

    name: str = "har"
    epsilon: float = 1e-12 #minimum prediction allowed
    hac_lags: int = 21
    feature_names: tuple[str, str, str] = ("har_d", "har_w", "har_m")
    result_: Any | None = field(default=None, init=False, repr=False)

    def fit(self, ds: VolDataset) -> None:
        """
        Fit OLS HAR model with HAC/Newey-West covariance.

        HAC affects standard errors and t-stats, not point forecasts.
        """
        self._validate_dataset(ds)

        X = ds.X.loc[:, list(self.feature_names)].astype(float)
        y = ds.y.astype(float)

        X_const = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, X_const, missing="raise")

        self.result_ = model.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": self.hac_lags},
        )

    def predict(self, ds: VolDataset) -> pd.Series:
        """
        Predict daily variance for every row in ds.
        """
        if self.result_ is None:
            raise RuntimeError("HARForecaster is not fitted.")

        self._validate_features(ds)

        X = ds.X.loc[:, list(self.feature_names)].astype(float)
        X_const = sm.add_constant(X, has_constant="add")

        raw_pred = self.result_.predict(X_const)

        pred = pd.Series(
            raw_pred,
            index=ds.X.index,
            name=self.name,
        )

        return pred.clip(lower=self.epsilon)

    def _validate_dataset(self, ds: VolDataset) -> None:
        self._validate_features(ds)

        if ds.X.empty:
            raise ValueError("Cannot fit HARForecaster on an empty dataset.")

        if not ds.X.index.equals(ds.y.index):
            raise ValueError("X and y indexes must be aligned.")

        if ds.y.isna().any():
            raise ValueError("Target y contains missing values.")

    def _validate_features(self, ds: VolDataset) -> None:
        missing = set(self.feature_names) - set(ds.X.columns)

        if missing:
            raise ValueError(f"Missing HAR feature columns: {sorted(missing)}")

        X = ds.X.loc[:, list(self.feature_names)]

        if X.isna().any().any():
            raise ValueError("HAR features contain missing values.")
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import statsmodels.api as sm

from .dataset import VolDataset
from .fold import ForecastFold


@dataclass
class HARForecaster:
    """
    HAR-RV volatility forecaster.

    Fits:

        y_t = beta_0
              + beta_d * har_d(t)
              + beta_w * har_w(t)
              + beta_m * har_m(t)
              + error_t

    Forecasts are expressed in daily variance units.

    The model receives the complete dataset and a ForecastFold:

        fit:
            Uses only supervised rows whose target windows end by the
            fold cutoff.

        predict:
            Produces forecasts for feature dates strictly after the
            fold cutoff.
    """

    name: str = "har"
    epsilon: float = 1e-12
    hac_lags: int = 21
    feature_names: tuple[str, str, str] = (
        "har_d",
        "har_w",
        "har_m",
    )

    result_: Any | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def fit(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> None:
        """
        Fit OLS using the fold's leakage-safe supervised training rows.

        HAC/Newey-West covariance changes standard errors and inference,
        not coefficient estimates or point forecasts.
        """
        train_ds = fold.train_dataset(ds)

        self._validate_training_dataset(train_ds)

        X = train_ds.X.loc[:, list(self.feature_names)].astype(float)
        y = train_ds.y.astype(float)

        X_with_constant = sm.add_constant(
            X,
            has_constant="add",
        )

        model = sm.OLS(
            y,
            X_with_constant,
            missing="raise",
        )

        maxlags = min(
            self.hac_lags,
            len(train_ds.X) - 1,
        )

        self.result_ = model.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maxlags},
        )

    def predict(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> pd.Series:
        """
        Predict future average daily variance for the fold's test origins.
        """
        if self.result_ is None:
            raise RuntimeError("HARForecaster is not fitted.")

        test_ds = fold.test_dataset(ds)

        self._validate_prediction_features(test_ds.X)

        X = test_ds.X.loc[:, list(self.feature_names)].astype(float)

        X_with_constant = sm.add_constant(
            X,
            has_constant="add",
        )

        raw_prediction = self.result_.predict(X_with_constant)

        prediction = pd.Series(
            raw_prediction,
            index=test_ds.X.index,
            name=self.name,
            dtype=float,
        )

        return prediction.clip(lower=self.epsilon)

    def _validate_training_dataset(
        self,
        ds: VolDataset,
    ) -> None:
        if ds.X.empty:
            raise ValueError("HAR training dataset is empty.")

        if not ds.X.index.equals(ds.y.index):
            raise ValueError("HAR training X and y indexes are not aligned.")

        self._validate_prediction_features(ds.X)

        if ds.y.isna().any():
            raise ValueError("HAR training target contains missing values.")

        if not pd.api.types.is_numeric_dtype(ds.y):
            raise ValueError("HAR training target must be numeric.")

    def _validate_prediction_features(
        self,
        X: pd.DataFrame,
    ) -> None:
        if X.empty:
            raise ValueError("HAR prediction dataset is empty.")

        missing = [
            feature
            for feature in self.feature_names
            if feature not in X.columns
        ]

        if missing:
            raise ValueError(
                f"HAR dataset is missing required features: {missing}"
            )

        features = X.loc[:, list(self.feature_names)]

        if features.isna().any().any():
            raise ValueError("HAR features contain missing values.")

        non_numeric = [
            column
            for column in features.columns
            if not pd.api.types.is_numeric_dtype(features[column])
        ]

        if non_numeric:
            raise ValueError(
                f"HAR features must be numeric: {non_numeric}"
            )
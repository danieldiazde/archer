from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from archer.models.base import Forecaster
from archer.models.dataset import VolDataset
from archer.models.fold import ForecastFold

from .panel import PredictionPanel, make_prediction_panel


ModelFactory = Callable[[], Forecaster]


def run_walkforward(
    *,
    ds: VolDataset,
    folds: Sequence[ForecastFold],
    model_factories: Sequence[ModelFactory],
) -> PredictionPanel:
    """
    Run an expanding-window walk-forward forecasting tournament.

    For each model factory:

        1. Create a fresh model instance per fold.
        2. Fit using only information allowed by that fold.
        3. Predict that fold's test origins.
        4. Stitch all fold predictions into one aligned forecast series.

    The returned PredictionPanel contains realized targets, fold metadata,
    and one forecast column per model.
    """
    if not folds:
        raise ValueError("at least one forecast fold is required.")

    if not model_factories:
        raise ValueError("at least one model factory is required.")

    panel = make_prediction_panel(
        ds=ds,
        folds=folds,
    )

    for model_factory in model_factories:
        model_name: str | None = None
        fold_predictions: list[pd.Series] = []

        for fold in folds:
            model = model_factory()

            if model_name is None:
                model_name = model.name

            model.fit(ds, fold)

            prediction = model.predict(ds, fold)

            _validate_fold_prediction(
                prediction=prediction,
                fold=fold,
                model_name=model_name,
            )

            fold_predictions.append(
                prediction.astype(float)
            )

        if model_name is None:
            raise RuntimeError("model factory did not produce a model.")

        forecast = pd.concat(
            fold_predictions,
            axis=0,
        )

        forecast = forecast.loc[panel.frame.index]
        forecast.name = model_name

        panel = panel.with_forecast(
            name=model_name,
            forecast=forecast,
        )

    return panel


def _validate_fold_prediction(
    *,
    prediction: pd.Series,
    fold: ForecastFold,
    model_name: str,
) -> None:
    if not isinstance(prediction, pd.Series):
        raise TypeError(
            f"{model_name} prediction must be a pandas Series."
        )

    if not prediction.index.equals(fold.test_idx):
        raise ValueError(
            f"{model_name} prediction index must equal fold test index."
        )

    if prediction.empty:
        raise ValueError(
            f"{model_name} prediction is empty."
        )

    if not pd.api.types.is_numeric_dtype(prediction):
        raise ValueError(
            f"{model_name} prediction values must be numeric."
        )

    values = prediction.astype(float)

    if values.isna().any():
        raise ValueError(
            f"{model_name} prediction contains missing values."
        )

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(
            f"{model_name} prediction contains non-finite values."
        )

    if (values <= 0.0).any():
        raise ValueError(
            f"{model_name} prediction values must be strictly positive."
        )
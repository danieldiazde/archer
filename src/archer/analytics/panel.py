from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Sequence

import numpy as np
import pandas as pd

from archer.models.dataset import VolDataset
from archer.models.fold import ForecastFold


@dataclass(frozen=True, slots=True)
class PredictionPanel:
    """
    Central out-of-sample forecasting artifact.

    The frame is indexed by forecast origin and contains:

        realized:
            Future mean daily variance observed after the forecast origin.

        target_end:
            Final date included in the realized target window.

        fit_cutoff:
            Information cutoff used to fit the model for this forecast.

        fold_id:
            Walk-forward fold that produced this forecast.

        model columns:
            One strictly positive daily-variance forecast per model.
    """

    RESERVED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "realized",
        "target_end",
        "fit_cutoff",
        "fold_id",
    )

    frame: pd.DataFrame
    horizon: int

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1.")

        if self.frame.empty:
            raise ValueError("prediction panel cannot be empty.")

        missing = [
            column
            for column in self.RESERVED_COLUMNS
            if column not in self.frame.columns
        ]

        if missing:
            raise ValueError(
                f"prediction panel is missing required columns: {missing}"
            )

        if not self.frame.index.is_unique:
            raise ValueError(
                "prediction panel forecast origins must be unique."
            )

        if not self.frame.index.is_monotonic_increasing:
            raise ValueError(
                "prediction panel forecast origins must be sorted."
            )

    @property
    def model_names(self) -> tuple[str, ...]:
        """
        Return forecast-model columns in insertion order.
        """
        return tuple(
            column
            for column in self.frame.columns
            if column not in self.RESERVED_COLUMNS
        )

    def with_forecast(
        self,
        name: str,
        forecast: pd.Series,
    ) -> "PredictionPanel":
        """
        Return a new panel containing one additional model forecast.

        Forecasts must:

            - exactly match the panel's forecast-origin index;
            - contain finite numeric values;
            - be strictly positive;
            - use a nonreserved, previously unused model name.
        """
        if not name:
            raise ValueError("forecast name cannot be empty.")

        if name in self.RESERVED_COLUMNS:
            raise ValueError(
                f"forecast name {name!r} is reserved."
            )

        if name in self.frame.columns:
            raise ValueError(
                f"forecast {name!r} already exists in the panel."
            )

        if not isinstance(forecast, pd.Series):
            raise TypeError("forecast must be a pandas Series.")

        if not forecast.index.equals(self.frame.index):
            raise ValueError(
                "forecast index must exactly match the prediction panel index."
            )

        if not pd.api.types.is_numeric_dtype(forecast):
            raise ValueError("forecast values must be numeric.")

        forecast_values = forecast.astype(float)

        if forecast_values.isna().any():
            raise ValueError("forecast contains missing values.")

        if not np.isfinite(forecast_values.to_numpy()).all():
            raise ValueError("forecast contains non-finite values.")

        if (forecast_values <= 0.0).any():
            raise ValueError(
                "forecast values must be strictly positive."
            )

        updated_frame = self.frame.copy()
        updated_frame[name] = forecast_values

        return PredictionPanel(
            frame=updated_frame,
            horizon=self.horizon,
        )


def make_prediction_panel(
    ds: VolDataset,
    folds: Sequence[ForecastFold],
) -> PredictionPanel:
    """
    Construct the empty walk-forward prediction panel.

    This function records realized targets and fold provenance. Model
    forecasts are added later through ``PredictionPanel.with_forecast()``.
    """
    if not folds:
        raise ValueError("at least one forecast fold is required.")

    pieces: list[pd.DataFrame] = []

    for fold in folds:
        if fold.test_idx.empty:
            raise ValueError(
                f"forecast fold {fold.fold_id} contains no test rows."
            )

        missing_origins = fold.test_idx.difference(ds.X.index)

        if not missing_origins.empty:
            raise ValueError(
                f"forecast fold {fold.fold_id} contains dates "
                "outside the dataset."
            )

        fold_index = pd.DatetimeIndex(
            fold.test_idx,
            name="forecast_origin",
        )

        piece = pd.DataFrame(
            {
                "realized": ds.y.loc[fold.test_idx].to_numpy(
                    dtype=float
                ),
                "target_end": pd.to_datetime(
                    ds.y_end.loc[fold.test_idx]
                ).to_numpy(),
                "fit_cutoff": [pd.Timestamp(fold.cutoff)] * len(fold_index),
                "fold_id": np.repeat(
                    fold.fold_id,
                    len(fold_index),
                ),
            },
            index=fold_index,
        )

        pieces.append(piece)

    frame = pd.concat(pieces, axis=0)

    frame.index = pd.DatetimeIndex(
        frame.index,
        name="forecast_origin",
    )

    if not frame.index.is_unique:
        duplicated = frame.index[
            frame.index.duplicated()
        ]

        raise ValueError(
            "forecast folds contain overlapping test origins: "
            f"{duplicated.unique().tolist()}"
        )

    if not frame.index.is_monotonic_increasing:
        raise ValueError(
            "forecast folds are not ordered chronologically."
        )

    realized = frame["realized"]

    if realized.isna().any():
        raise ValueError(
            "prediction panel realized targets contain missing values."
        )

    if not np.isfinite(realized.to_numpy()).all():
        raise ValueError(
            "prediction panel realized targets contain non-finite values."
        )

    if (realized <= 0.0).any():
        raise ValueError(
            "prediction panel realized targets must be strictly positive."
        )

    return PredictionPanel(
        frame=frame,
        horizon=ds.horizon,
    )
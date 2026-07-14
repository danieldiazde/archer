from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .dataset import VolDataset


@dataclass(frozen=True, slots=True)
class ForecastFold:
    """
    Dates and row indexes defining one leakage-safe forecasting fold.

    cutoff:
        Final date of information available when the model is fitted.

    train_idx:
        Supervised training rows whose target windows end on or before
        the cutoff.

    test_idx:
        Forecast origins strictly after the cutoff.

    test_end:
        Final forecast origin included in this fold.

    fold_id:
        Sequential identifier for the fold.
    """

    cutoff: pd.Timestamp
    train_idx: pd.Index
    test_idx: pd.Index
    test_end: pd.Timestamp
    fold_id: int = 0

    def train_dataset(self, ds: VolDataset) -> VolDataset:
        """
        Return the fold's purged supervised training dataset.
        """
        return ds.slice(self.train_idx)

    def test_dataset(self, ds: VolDataset) -> VolDataset:
        """
        Return the fold's out-of-sample forecasting rows.
        """
        return ds.slice(self.test_idx)


def make_forecast_fold(
    ds: VolDataset,
    *,
    cutoff: pd.Timestamp | str,
    test_end: pd.Timestamp | str | None = None,
    fold_id: int = 0,
) -> ForecastFold:
    """
    Construct one leakage-safe forecasting fold.

    Training rows are selected using the target-window end date:

        y_end <= cutoff

    Test rows are selected using the feature date:

        cutoff < feature_date <= test_end

    The complete dataset remains available to models that need observed
    information through the cutoff, such as GARCH.
    """
    cutoff_ts = pd.Timestamp(cutoff)

    if fold_id < 0:
        raise ValueError("fold_id must be nonnegative.")

    if test_end is not None:
        requested_test_end = pd.Timestamp(test_end)

        if requested_test_end <= cutoff_ts:
            raise ValueError("test_end must be after cutoff.")
    else:
        requested_test_end = None

    train_ds, test_ds = ds.split(cutoff_ts)

    test_idx = test_ds.X.index

    if requested_test_end is not None:
        test_idx = test_idx[test_idx <= requested_test_end]

    if test_idx.empty:
        raise ValueError("Forecast fold contains no test rows.")

    actual_test_end = pd.Timestamp(test_idx.max())

    return ForecastFold(
        cutoff=cutoff_ts,
        train_idx=train_ds.X.index,
        test_idx=test_idx,
        test_end=actual_test_end,
        fold_id=fold_id,
    )


def make_expanding_folds(
    ds: VolDataset,
    *,
    first_cutoff: pd.Timestamp | str,
    refit_every: int = 21,
    final_test_end: pd.Timestamp | str | None = None,
) -> list[ForecastFold]:
    """
    Construct an expanding-window walk-forward schedule.

    The first model is fitted using information available at
    ``first_cutoff``.

    It then predicts the next ``refit_every`` available dataset rows.
    The final forecast origin in that block becomes the next fold's cutoff.

    This produces:

        - expanding training history;
        - nonoverlapping test blocks;
        - no missing forecast origins;
        - no duplicated forecast origins;
        - one centralized source of time-split logic.

    No calendar dates are invented. The schedule partitions the dates already
    present in the dataset.
    """
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1.")

    first_cutoff_ts = pd.Timestamp(first_cutoff)

    if final_test_end is not None:
        final_test_end_ts = pd.Timestamp(final_test_end)

        if final_test_end_ts <= first_cutoff_ts:
            raise ValueError(
                "final_test_end must be after first_cutoff."
            )
    else:
        final_test_end_ts = None

    forecast_origins = ds.X.index[
        ds.X.index > first_cutoff_ts
    ]

    if final_test_end_ts is not None:
        forecast_origins = forecast_origins[
            forecast_origins <= final_test_end_ts
        ]

    if forecast_origins.empty:
        raise ValueError(
            "No forecast origins exist after first_cutoff."
        )

    folds: list[ForecastFold] = []
    current_cutoff = first_cutoff_ts

    for start in range(
        0,
        len(forecast_origins),
        refit_every,
    ):
        block_idx = forecast_origins[
            start : start + refit_every
        ]

        block_end = pd.Timestamp(block_idx.max())

        fold = make_forecast_fold(
            ds,
            cutoff=current_cutoff,
            test_end=block_end,
            fold_id=len(folds),
        )

        folds.append(fold)

        # The last forecast origin in this block becomes the next
        # parameter-refit cutoff.
        current_cutoff = fold.test_end

    return folds
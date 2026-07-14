from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .dataset import VolDataset


@dataclass(frozen=True, slots=True)
class ForecastFold:
    """
    Dates defining one leakage-safe forecasting fold.

    cutoff:
        Final date of information available during model fitting.

    train_idx:
        Supervised training rows whose target windows end on or before
        the cutoff.

    test_idx:
        Forecast origins strictly after the cutoff.
    """

    cutoff: pd.Timestamp
    train_idx: pd.Index
    test_idx: pd.Index

    def train_dataset(self, ds: VolDataset) -> VolDataset:
        return ds.slice(self.train_idx)

    def test_dataset(self, ds: VolDataset) -> VolDataset:
        return ds.slice(self.test_idx)


def make_forecast_fold(
    ds: VolDataset,
    *,
    cutoff: pd.Timestamp | str,
    test_end: pd.Timestamp | str | None = None,
) -> ForecastFold:
    """
    Construct one leakage-safe forecasting fold.

    The full dataset remains available to models that need historical
    information through the cutoff, such as GARCH.
    """
    cutoff_ts = pd.Timestamp(cutoff)

    train, test = ds.split(cutoff_ts)
    test_idx = test.X.index

    if test_end is not None:
        test_end_ts = pd.Timestamp(test_end)

        if test_end_ts <= cutoff_ts:
            raise ValueError("test_end must be after cutoff.")

        test_idx = test_idx[test_idx <= test_end_ts]

    return ForecastFold(
        cutoff=cutoff_ts,
        train_idx=train.X.index,
        test_idx=test_idx,
    )
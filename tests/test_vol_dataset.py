from __future__ import annotations

import numpy as np
import pandas as pd

from archer.models.dataset import build_vol_dataset


def test_build_vol_dataset_is_point_in_time_safe_for_har_features_and_target() -> None:
    dates = pd.bdate_range("2020-01-01", periods=80)

    variance = pd.Series(
        np.arange(len(dates), dtype=float),
        index=dates,
        name="variance",
    )

    returns = pd.Series(
        0.0,
        index=dates,
        name="returns",
    )

    ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=21,
        weekly_window=5,
        monthly_window=22,
    )

    # Pick a row safely away from the beginning and end.
    date = dates[30]
    t = 30.0

    assert ds.horizon == 21

    assert ds.X.loc[date, "har_d"] == t
    assert ds.X.loc[date, "har_w"] == t - 2.0   # (30 + 29 + 28 + 27 + 26) / 5 = 28
    assert ds.X.loc[date, "har_m"] == t - 10.5  # (9 + 30) / 2
    assert ds.y.loc[date] == t + 11.0 # for a 21 day window (31 + 51) / 2

    assert ds.X.index.equals(ds.y.index)
    assert ds.X.index.equals(ds.returns.index)
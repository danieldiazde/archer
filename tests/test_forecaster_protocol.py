from __future__ import annotations

import pandas as pd

from archer.models.base import Forecaster
from archer.models.dataset import VolDataset
from archer.models.fold import ForecastFold, make_forecast_fold


class DummyForecaster:
    name = "dummy"

    def fit(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> None:
        self.fitted_ = True

    def predict(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> pd.Series:
        test_ds = fold.test_dataset(ds)

        return pd.Series(
            1.0,
            index=test_ds.X.index,
            name=self.name,
        )


def test_dummy_forecaster_satisfies_protocol() -> None:
    model: Forecaster = DummyForecaster()

    horizon = 1
    idx = pd.bdate_range("2020-01-01", periods=6)

    ds = VolDataset(
        X=pd.DataFrame(
            {
                "har_d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "har_w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "har_m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
            index=idx,
        ),
        y=pd.Series(
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            index=idx,
            name="y",
        ),
        y_end=pd.Series(
            idx + pd.offsets.BDay(horizon),
            index=idx,
            name="y_end",
        ),
        returns=pd.Series(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=idx,
            name="returns",
        ),
        horizon=horizon,
    )

    fold = make_forecast_fold(
        ds,
        cutoff=idx[2],
    )

    model.fit(ds, fold)
    pred = model.predict(ds, fold)

    assert pred.index.equals(fold.test_idx)
    assert pred.name == "dummy"
    assert (pred == 1.0).all()
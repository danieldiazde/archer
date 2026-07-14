from __future__ import annotations

import pandas as pd

from archer.models.base import Forecaster
from archer.models.dataset import VolDataset


class DummyForecaster:
    name = "dummy"

    def fit(self, ds: VolDataset) -> None:
        self.fitted_ = True

    def predict(self, ds: VolDataset) -> pd.Series:
        return pd.Series(
            1.0,
            index=ds.X.index,
            name=self.name,
        )


def test_dummy_forecaster_satisfies_protocol() -> None:
    model: Forecaster = DummyForecaster()

    horizon = 21
    idx = pd.bdate_range("2020-01-01", periods=3)

    y_end = pd.Series(
        idx + pd.offsets.BDay(horizon),
        index=idx,
        name="y_end",
    )

    ds = VolDataset(
        X=pd.DataFrame(
            {
                "har_d": [1.0, 2.0, 3.0],
                "har_w": [1.0, 2.0, 3.0],
                "har_m": [1.0, 2.0, 3.0],
            },
            index=idx,
        ),
        y=pd.Series(
            [2.0, 3.0, 4.0],
            index=idx,
            name="y",
        ),
        y_end=y_end,
        returns=pd.Series(
            [0.0, 0.0, 0.0],
            index=idx,
            name="returns",
        ),
        horizon=horizon,
    )

    model.fit(ds)
    pred = model.predict(ds)

    assert pred.index.equals(ds.X.index)
    assert pred.name == "dummy"
    assert (pred == 1.0).all()
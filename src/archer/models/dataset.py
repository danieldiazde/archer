from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class VolDataset:
    """
    Leakage-safe volatility forecasting dataset.

    X:
        Features available at or before feature date t.

    y:
        Mean future daily variance over (t, t + horizon].

    y_end:
        Final date included in each row's future target window.

        This allows train/test splits to purge rows whose labels extend
        beyond the training cutoff.

    returns:
        Daily log returns aligned with X and y.
        These will later be consumed by return-based models such as GARCH.

    horizon:
        Forecast horizon in trading days.
    """

    X: pd.DataFrame
    y: pd.Series
    y_end: pd.Series
    returns: pd.Series
    horizon: int

    def slice(self, idx: pd.Index) -> "VolDataset":
        """
        Return a subset while preserving alignment among every component.
        """
        return VolDataset(
            X=self.X.loc[idx],
            y=self.y.loc[idx],
            y_end=self.y_end.loc[idx],
            returns=self.returns.loc[idx],
            horizon=self.horizon,
        )

    def split(
        self,
        cutoff: pd.Timestamp | str,
    ) -> tuple["VolDataset", "VolDataset"]:
        """
        Split the dataset at a cutoff without target leakage.

        Training rows:
            Keep only rows whose complete future target window ends on or
            before the cutoff.

        Test rows:
            Keep rows whose feature date is strictly after the cutoff.

        Test features may use pre-cutoff history. That is legitimate because
        this history would have been observable at prediction time.
        """
        cutoff_ts = pd.Timestamp(cutoff)

        train_idx = self.X.index[self.y_end <= cutoff_ts]
        test_idx = self.X.index[self.X.index > cutoff_ts]

        return self.slice(train_idx), self.slice(test_idx)


def build_vol_dataset(
    *,
    variance: pd.Series,
    returns: pd.Series,
    horizon: int = 21,
    weekly_window: int = 5,
    monthly_window: int = 22,
) -> VolDataset:
    """
    Build a point-in-time volatility forecasting dataset.

    Features use information available at or before feature date t:

        har_d(t) = v_t

        har_w(t) =
            mean(v_t, ..., v_{t - weekly_window + 1})

        har_m(t) =
            mean(v_t, ..., v_{t - monthly_window + 1})

    The target uses strictly future information:

        y_t = mean(v_{t + 1}, ..., v_{t + horizon})

    Rows without sufficient historical features or a complete future target
    window are dropped rather than filled.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    if weekly_window < 1:
        raise ValueError("weekly_window must be at least 1.")

    if monthly_window < weekly_window:
        raise ValueError(
            "monthly_window must be greater than or equal to weekly_window."
        )

    variance = variance.copy()
    returns = returns.copy()

    variance.index = pd.to_datetime(variance.index)
    returns.index = pd.to_datetime(returns.index)

    variance = variance.sort_index()
    returns = returns.sort_index()

    if variance.index.has_duplicates:
        raise ValueError("variance index contains duplicate dates.")

    if returns.index.has_duplicates:
        raise ValueError("returns index contains duplicate dates.")

    common_index = variance.index.intersection(returns.index)

    if common_index.empty:
        raise ValueError("variance and returns have no overlapping dates.")

    variance = variance.loc[common_index].astype(float)
    returns = returns.loc[common_index].astype(float)

    X = pd.DataFrame(index=variance.index)

    X["har_d"] = variance

    X["har_w"] = variance.rolling(
        window=weekly_window,
        min_periods=weekly_window,
    ).mean()

    X["har_m"] = variance.rolling(
        window=monthly_window,
        min_periods=monthly_window,
    ).mean()

    # At feature date t, this becomes:
    # mean(v_{t+1}, ..., v_{t+horizon})
    y = (
        variance.rolling(
            window=horizon,
            min_periods=horizon,
        )
        .mean()
        .shift(-horizon)
    )
    y.name = f"future_mean_variance_{horizon}"

    # At feature date t, this records the date of v_{t+horizon},
    # which is the final observation included in y_t.
    target_dates = pd.Series(
        variance.index,
        index=variance.index,
        name="y_end",
    )

    y_end = target_dates.shift(-horizon)

    panel = (
        X.join(y)
        .join(y_end)
        .join(returns.rename("returns"))
        .dropna()
    )

    X_out = panel[["har_d", "har_w", "har_m"]]
    y_out = panel[y.name]
    y_end_out = panel["y_end"]
    returns_out = panel["returns"]

    return VolDataset(
        X=X_out,
        y=y_out,
        y_end=y_end_out,
        returns=returns_out,
        horizon=horizon,
    )
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class VolDataset:
    """
    Leakage-safe volatility forecasting dataset.

    X:
        HAR features known at time t.

    y:
        Future average daily variance over (t, t+horizon].

    returns:
        Daily log returns aligned to the same dates as X and y.

    horizon:
        Forecast horizon in trading days.
    """

    X: pd.DataFrame
    y: pd.Series
    returns: pd.Series
    horizon: int

    def slice(self, idx: pd.DatetimeIndex) -> "VolDataset":
        """
        Return a date-indexed subset of the dataset.
        """
        return VolDataset(
            X=self.X.loc[idx],
            y=self.y.loc[idx],
            returns=self.returns.loc[idx],
            horizon=self.horizon,
        )


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

    Features use information available at or before t:
        har_d(t) = v_t
        har_w(t) = mean(v_t, ..., v_{t-weekly_window+1})
        har_m(t) = mean(v_t, ..., v_{t-monthly_window+1})

    Target uses strictly future information:
        y_t = mean(v_{t+1}, ..., v_{t+horizon})
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    if weekly_window < 1:
        raise ValueError("weekly_window must be at least 1.")

    if monthly_window < weekly_window:
        raise ValueError("monthly_window must be greater than or equal to weekly_window.")

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

    y = variance.rolling(
        window=horizon,
        min_periods=horizon,
    ).mean().shift(-horizon)

    y.name = f"future_mean_variance_{horizon}"

    panel = X.join(y).join(returns.rename("returns"))
    panel = panel.dropna()

    X_out = panel[["har_d", "har_w", "har_m"]]
    y_out = panel[y.name]
    returns_out = panel["returns"]

    return VolDataset(
        X=X_out,
        y=y_out,
        returns=returns_out,
        horizon=horizon,
    )
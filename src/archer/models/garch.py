from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model

from .dataset import VolDataset
from .fold import ForecastFold


@dataclass
class GarchForecaster:
    """
    Expanding-window GARCH(p, q) volatility forecaster.

    Model:

        r_t = epsilon_t

        epsilon_t = sigma_t * z_t

        sigma_t^2 =
            omega
            + alpha * epsilon_{t-1}^2
            + beta * sigma_{t-1}^2

    Returns are multiplied by ``return_scale`` before fitting because GARCH
    optimizers behave better when daily returns are expressed in percentage
    points rather than small decimal values.

    The model forecasts the variance path over the dataset horizon and returns
    the average forecast variance:

        y_hat_t =
            mean(
                sigma^2_{t+1|t},
                ...,
                sigma^2_{t+horizon|t},
            )

    Output units are decimal daily variance, matching VolDataset.y.
    """

    name: str = "garch"
    p: int = 1
    q: int = 1
    min_obs: int = 100
    return_scale: float = 100.0
    epsilon: float = 1e-12

    result_: Any | None = field(
        default=None,
        init=False,
        repr=False,
    )

    fit_end_: pd.Timestamp | None = field(
        default=None,
        init=False,
    )

    nobs_: int = field(
        default=0,
        init=False,
    )

    def fit(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> None:
        """
        Estimate GARCH parameters using returns observed through the cutoff.

        The full return series is passed to ``arch_model`` so the fitted result
        can update its conditional variance state using later observed returns.

        Parameter estimation itself stops at ``fold.cutoff``.
        """
        self._validate_configuration()
        returns = self._prepare_returns(ds)

        cutoff = pd.Timestamp(fold.cutoff)

        if cutoff not in returns.index:
            raise ValueError(
                "GARCH returns must include the fold cutoff."
            )

        cutoff_position = int(
            returns.index.get_indexer(pd.Index([cutoff]))[0]
        )

        if cutoff_position < 0:
            raise ValueError(
                "GARCH returns must include the fold cutoff."
            )

        fit_returns = returns.iloc[: cutoff_position + 1]

        if len(fit_returns) < self.min_obs:
            raise ValueError(
                "GARCH requires at least "
                f"{self.min_obs} return observations; "
                f"received {len(fit_returns)}."
            )

        # arch's last_obs is excluded from estimation. To include the cutoff,
        # pass the first observation after the cutoff as last_obs.
        next_position = cutoff_position + 1

        if next_position >= len(returns):
            raise ValueError(
                "GARCH requires at least one return after the fold cutoff."
            )

        last_obs_exclusive = returns.index[next_position]

        scaled_returns = returns * self.return_scale

        model = arch_model(
            scaled_returns,
            mean="Zero",
            vol="GARCH",
            p=self.p,
            o=0,
            q=self.q,
            dist="normal",
            rescale=False,
        )

        result = model.fit(
            last_obs=last_obs_exclusive,
            update_freq=0,
            disp="off",
            show_warning=False,
        )

        if result.convergence_flag != 0:
            raise RuntimeError(
                "GARCH optimization did not converge at cutoff "
                f"{cutoff.date()}."
            )

        self.result_ = result
        self.fit_end_ = cutoff
        self.nobs_ = len(fit_returns)

    def predict(
        self,
        ds: VolDataset,
        fold: ForecastFold,
    ) -> pd.Series:
        """
        Forecast average daily variance for every test origin in the fold.

        Parameters remain fixed within the fold, while the conditional state
        incorporates returns observed through each forecast origin.
        """
        if self.result_ is None or self.fit_end_ is None:
            raise RuntimeError("GarchForecaster is not fitted.")

        if pd.Timestamp(fold.cutoff) != self.fit_end_:
            raise ValueError(
                "GARCH prediction fold cutoff does not match fit cutoff."
            )

        if fold.test_idx.empty:
            raise ValueError("GARCH prediction fold contains no test rows.")

        forecasts = self.result_.forecast(
            horizon=ds.horizon,
            start=fold.test_idx.min(),
            align="origin",
            method="analytic",
            reindex=False,
        )

        variance_percent_squared = forecasts.variance

        missing_origins = fold.test_idx.difference(
            variance_percent_squared.index
        )

        if not missing_origins.empty:
            raise ValueError(
                "GARCH forecast output is missing required origins: "
                f"{missing_origins.tolist()}"
            )

        horizon_variances = variance_percent_squared.loc[
            fold.test_idx
        ]

        if horizon_variances.shape[1] != ds.horizon:
            raise RuntimeError(
                "GARCH returned an unexpected forecast horizon: "
                f"expected {ds.horizon}, "
                f"received {horizon_variances.shape[1]}."
            )

        # arch returns percentage-return variance because inputs were scaled
        # by 100. Divide by 100² to recover decimal-return variance.
        forecast = (
            horizon_variances.mean(axis=1)
            / self.return_scale**2
        )

        forecast = forecast.astype(float)

        if forecast.isna().any():
            raise ValueError(
                "GARCH forecast contains missing values."
            )

        if not np.isfinite(forecast.to_numpy()).all():
            raise ValueError(
                "GARCH forecast contains non-finite values."
            )

        return (
            forecast
            .clip(lower=self.epsilon)
            .rename(self.name)
        )

    def _prepare_returns(
        self,
        ds: VolDataset,
    ) -> pd.Series:
        if not isinstance(ds.returns, pd.Series):
            raise TypeError(
                "VolDataset returns must be a pandas Series."
            )

        if ds.returns.empty:
            raise ValueError("GARCH return series is empty.")

        if ds.returns.index.has_duplicates:
            raise ValueError(
                "GARCH return index contains duplicate dates."
            )

        if not pd.api.types.is_numeric_dtype(ds.returns):
            raise ValueError("GARCH returns must be numeric.")

        returns = ds.returns.copy()
        returns.index = pd.to_datetime(returns.index)
        returns = returns.sort_index().astype(float)

        if returns.isna().any():
            raise ValueError(
                "GARCH returns contain missing values."
            )

        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError(
                "GARCH returns contain non-finite values."
            )

        return returns

    def _validate_configuration(self) -> None:
        if self.p < 1:
            raise ValueError("GARCH p must be at least 1.")

        if self.q < 1:
            raise ValueError("GARCH q must be at least 1.")

        if self.min_obs < 1:
            raise ValueError("GARCH min_obs must be at least 1.")

        if self.return_scale <= 0.0:
            raise ValueError(
                "GARCH return_scale must be strictly positive."
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "GARCH epsilon must be strictly positive."
            )
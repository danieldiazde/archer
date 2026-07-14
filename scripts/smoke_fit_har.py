from __future__ import annotations

import numpy as np
import pandas as pd

from archer.data.ingest import load_ingest_config
from archer.data.returns import make_log_returns, make_price_matrix
from archer.data.store import ParquetStore
from archer.features.realized_vol import daily_variance
from archer.models.dataset import build_vol_dataset
from archer.models.har import HARForecaster


TRAIN_END = pd.Timestamp("2019-12-31")
TRADING_DAYS = 252


def main() -> None:
    cfg = load_ingest_config("config/data.yaml")

    store = ParquetStore(
        bronze_dir=cfg.bronze_dir,
        silver_dir=cfg.silver_dir,
    )

    spx = store.read_silver("^GSPC").copy()

    spx["date"] = (
        pd.to_datetime(spx["date"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )

    spx = (
        spx.sort_values("date")
        .reset_index(drop=True)
    )

    variance = daily_variance(
        spx,
        method="gk_total",
    )
    variance.name = "gk_total"

    prices = make_price_matrix(
        spx,
        field="adj_close",
    )

    returns = make_log_returns(prices)["^GSPC"]
    returns.name = "returns"

    full_ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=21,
        weekly_window=5,
        monthly_window=22,
    )

    train_ds, test_ds = full_ds.split(TRAIN_END)

    if train_ds.X.empty:
        raise RuntimeError("Purged HAR training dataset is empty.")

    if train_ds.y_end.max() > TRAIN_END:
        raise RuntimeError("Training target window crosses the cutoff.")

    model = HARForecaster(
        epsilon=1e-12,
        hac_lags=21,
    )

    model.fit(train_ds)

    if model.result_ is None:
        raise RuntimeError("HAR fit did not produce a result.")

    train_pred = model.predict(train_ds)

    params = model.result_.params
    standard_errors = model.result_.bse
    t_values = model.result_.tvalues
    p_values = model.result_.pvalues

    coefficient_table = pd.DataFrame(
        {
            "coefficient": params,
            "hac_std_error": standard_errors,
            "t_value": t_values,
            "p_value": p_values,
        }
    )

    persistence = float(
        params["har_d"]
        + params["har_w"]
        + params["har_m"]
    )

    # Convert daily variance predictions to annualized volatility
    predicted_annualized_vol = (
        train_pred * TRADING_DAYS
    ).pow(0.5)
    predicted_annualized_vol.name = "predicted_annualized_vol"

    actual_annualized_vol = (
        train_ds.y * TRADING_DAYS
    ).pow(0.5)
    actual_annualized_vol.name = "actual_annualized_vol"

    print("FULL DATASET")
    print(f"Rows: {len(full_ds.X):,}")
    print(
        f"Feature range: "
        f"{full_ds.X.index.min().date()} "
        f"to {full_ds.X.index.max().date()}"
    )

    print("\nPURGED TRAINING DATASET")
    print(f"Rows: {len(train_ds.X):,}")
    print(
        f"Feature range: "
        f"{train_ds.X.index.min().date()} "
        f"to {train_ds.X.index.max().date()}"
    )
    print(
        f"Latest target end: "
        f"{train_ds.y_end.max().date()}"
    )

    print("\nFUTURE TEST DATASET")
    print(f"Rows: {len(test_ds.X):,}")
    print(
        f"Feature range: "
        f"{test_ds.X.index.min().date()} "
        f"to {test_ds.X.index.max().date()}"
    )

    print("\nDAILY GK_TOTAL VARIANCE")
    print(variance.describe())

    print("\nHAR COEFFICIENTS WITH HAC INFERENCE")
    print(coefficient_table)

    print("\nHAR PERSISTENCE")
    print(f"beta_d + beta_w + beta_m = {persistence:.6f}")

    print("\nTRAINING PREDICTIONS: DAILY VARIANCE")
    print(train_pred.describe())

    print("\nTRAINING PREDICTIONS: ANNUALIZED VOLATILITY")
    print(predicted_annualized_vol.describe())

    print("\nACTUAL TARGET: ANNUALIZED VOLATILITY")
    print(actual_annualized_vol.describe())

    print("\nMODEL SUMMARY")
    print(model.result_.summary())


if __name__ == "__main__":
    main()
from __future__ import annotations

import pandas as pd

from archer.data.ingest import load_ingest_config
from archer.data.returns import make_log_returns, make_price_matrix
from archer.data.store import ParquetStore
from archer.features.realized_vol import daily_variance
from archer.models.dataset import build_vol_dataset
from archer.models.fold import make_forecast_fold
from archer.models.har import HARForecaster


TRAIN_END = pd.Timestamp("2019-12-31")
TRADING_DAYS = 252


def main() -> None:
    cfg = load_ingest_config("config/data.yaml")

    store = ParquetStore(
        bronze_dir=cfg.bronze_dir,
        silver_dir=cfg.silver_dir,
    )

    # ------------------------------------------------------------
    # 1. Load the complete clean SPX history
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 2. Estimate daily total realized variance
    # ------------------------------------------------------------
    variance = daily_variance(
        spx,
        method="gk_total",
    )
    variance.name = "gk_total"

    # ------------------------------------------------------------
    # 3. Calculate adjusted close-to-close log returns
    # ------------------------------------------------------------
    prices = make_price_matrix(
        spx,
        field="adj_close",
    )

    returns = make_log_returns(prices)["^GSPC"]
    returns.name = "returns"

    # ------------------------------------------------------------
    # 4. Build the complete dataset once
    # ------------------------------------------------------------
    full_ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=21,
        weekly_window=5,
        monthly_window=22,
    )

    # ------------------------------------------------------------
    # 5. Construct a leakage-safe forecasting fold
    # ------------------------------------------------------------
    fold = make_forecast_fold(
        full_ds,
        cutoff=TRAIN_END,
    )

    train_ds = fold.train_dataset(full_ds)
    test_ds = fold.test_dataset(full_ds)

    if train_ds.X.empty:
        raise RuntimeError("Purged HAR training dataset is empty.")

    if test_ds.X.empty:
        raise RuntimeError("HAR test dataset is empty.")

    if train_ds.y_end.max() > TRAIN_END:
        raise RuntimeError("Training target window crosses the cutoff.")

    if test_ds.X.index.min() <= TRAIN_END:
        raise RuntimeError("Test feature date is not after the cutoff.")

    # ------------------------------------------------------------
    # 6. Fit HAR using the fold's purged training rows
    # ------------------------------------------------------------
    model = HARForecaster(
        epsilon=1e-12,
        hac_lags=21,
    )

    model.fit(full_ds, fold)

    if model.result_ is None:
        raise RuntimeError("HAR fit did not produce a result.")

    # The fold-aware predict method returns only test forecasts.
    test_pred = model.predict(full_ds, fold)

    # ------------------------------------------------------------
    # 7. Inspect coefficients and HAC inference
    # ------------------------------------------------------------
    params = model.result_.params

    coefficient_table = pd.DataFrame(
        {
            "coefficient": params,
            "hac_std_error": model.result_.bse,
            "t_value": model.result_.tvalues,
            "p_value": model.result_.pvalues,
        }
    )

    persistence = float(
        params["har_d"]
        + params["har_w"]
        + params["har_m"]
    )

    # Model output remains in daily variance units.
    # Annualization is only for human-readable inspection.
    predicted_annualized_vol = (
        test_pred * TRADING_DAYS
    ).pow(0.5)
    predicted_annualized_vol.name = "predicted_annualized_vol"

    actual_annualized_vol = (
        test_ds.y * TRADING_DAYS
    ).pow(0.5)
    actual_annualized_vol.name = "actual_annualized_vol"

    # ------------------------------------------------------------
    # 8. Print smoke-fit evidence
    # ------------------------------------------------------------
    print("FULL DATASET")
    print(f"Rows: {len(full_ds.X):,}")
    print(
        "Feature range: "
        f"{full_ds.X.index.min().date()} "
        f"to {full_ds.X.index.max().date()}"
    )

    print("\nPURGED TRAINING DATASET")
    print(f"Rows: {len(train_ds.X):,}")
    print(
        "Feature range: "
        f"{train_ds.X.index.min().date()} "
        f"to {train_ds.X.index.max().date()}"
    )
    print(
        "Latest target end: "
        f"{train_ds.y_end.max().date()}"
    )

    print("\nFUTURE TEST DATASET")
    print(f"Rows: {len(test_ds.X):,}")
    print(
        "Feature range: "
        f"{test_ds.X.index.min().date()} "
        f"to {test_ds.X.index.max().date()}"
    )

    print("\nDAILY GK_TOTAL VARIANCE")
    print(variance.describe())

    print("\nHAR COEFFICIENTS WITH HAC INFERENCE")
    print(coefficient_table)

    print("\nHAR PERSISTENCE")
    print(
        "beta_d + beta_w + beta_m = "
        f"{persistence:.6f}"
    )

    print("\nOUT-OF-SAMPLE PREDICTIONS: DAILY VARIANCE")
    print(test_pred.describe())

    print("\nOUT-OF-SAMPLE PREDICTIONS: ANNUALIZED VOLATILITY")
    print(predicted_annualized_vol.describe())

    print("\nOUT-OF-SAMPLE ACTUAL TARGET: ANNUALIZED VOLATILITY")
    print(actual_annualized_vol.describe())

    print("\nMODEL SUMMARY")
    print(model.result_.summary())


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pandas as pd

from archer.analytics.report import build_forecast_report
from archer.analytics.walkforward import run_walkforward
from archer.data.ingest import load_ingest_config
from archer.data.returns import make_log_returns, make_price_matrix
from archer.data.store import ParquetStore
from archer.features.implied_vol import vix_to_daily_variance
from archer.features.realized_vol import daily_variance
from archer.models.benchmarks import (
    AlignedSeriesForecaster,
    NaiveMA22Forecaster,
)
from archer.models.dataset import build_vol_dataset
from archer.models.fold import make_expanding_folds
from archer.models.garch import GarchForecaster
from archer.models.har import HARForecaster


SPX_SYMBOL = "^GSPC"
VIX_SYMBOL = "^VIX"
VARIANCE_METHOD = "gk_total"

HORIZON = 21
WEEKLY_WINDOW = 5
MONTHLY_WINDOW = 22

FIRST_CUTOFF = pd.Timestamp("2013-12-31")
REFIT_EVERY = 21
BASELINE = "naive_ma22"

OUTPUT_DIR = Path("data/evals")

PANEL_PATH = OUTPUT_DIR / "forecast_panel.parquet"
MANIFEST_PATH = OUTPUT_DIR / "forecast_panel.manifest.json"

OVERALL_QLIKE_PATH = OUTPUT_DIR / "overall_qlike.csv"
OVERALL_MSE_PATH = OUTPUT_DIR / "overall_mse.csv"

QLIKE_BY_YEAR_PATH = OUTPUT_DIR / "qlike_by_year.csv"
MSE_BY_YEAR_PATH = OUTPUT_DIR / "mse_by_year.csv"

MZ_PATH = OUTPUT_DIR / "mz_calibration.csv"
DM_QLIKE_PATH = OUTPUT_DIR / "dm_qlike.csv"
DM_MSE_PATH = OUTPUT_DIR / "dm_mse.csv"


def main() -> None:
    cfg = load_ingest_config("config/data.yaml")

    store = ParquetStore(
        bronze_dir=cfg.bronze_dir,
        silver_dir=cfg.silver_dir,
    )

    # ------------------------------------------------------------
    # 1. Load and normalize SPX data
    # ------------------------------------------------------------
    spx = store.read_silver(SPX_SYMBOL).copy()

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
    # 2. Build realized-variance and return series
    # ------------------------------------------------------------
    variance = daily_variance(
        spx,
        method=VARIANCE_METHOD,
    )
    variance.name = VARIANCE_METHOD

    prices = make_price_matrix(
        spx,
        field="adj_close",
    )

    returns = make_log_returns(prices)[SPX_SYMBOL]
    returns.name = "returns"

    # ------------------------------------------------------------
    # 3. Build the complete point-in-time forecasting dataset
    # ------------------------------------------------------------
    ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=HORIZON,
        weekly_window=WEEKLY_WINDOW,
        monthly_window=MONTHLY_WINDOW,
    )

    # ------------------------------------------------------------
    # 4. Load and convert the VIX-implied forecast
    # ------------------------------------------------------------
    vix = store.read_silver(VIX_SYMBOL).copy()

    vix["date"] = (
        pd.to_datetime(vix["date"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )

    vix = (
        vix.sort_values("date")
        .reset_index(drop=True)
    )

    vix_levels = make_price_matrix(
        vix,
        field="close",
    )[VIX_SYMBOL]
    vix_levels.name = "vix"

    vix_implied = vix_to_daily_variance(
        vix_levels,
        calendar_horizon_days=30,
        trading_horizon_days=HORIZON,
        annual_calendar_days=365,
    )

    vix_implied = vix_implied.reindex(ds.X.index)

    missing_vix = vix_implied.index[
        vix_implied.isna()
    ]

    if not missing_vix.empty:
        raise RuntimeError(
            "VIX-implied forecast is missing required SPX dates. "
            f"First missing dates: {missing_vix[:10].tolist()}"
        )

    # partial creates a no-argument factory compatible with run_walkforward.
    vix_factory = partial(
        AlignedSeriesForecaster,
        name="vix_implied",
        series=vix_implied,
    )

    # ------------------------------------------------------------
    # 5. Generate the expanding-window schedule
    # ------------------------------------------------------------
    folds = make_expanding_folds(
        ds,
        first_cutoff=FIRST_CUTOFF,
        refit_every=REFIT_EVERY,
    )

    # ------------------------------------------------------------
    # 6. Run the forecasting tournament
    # ------------------------------------------------------------
    panel = run_walkforward(
        ds=ds,
        folds=folds,
        model_factories=[
            NaiveMA22Forecaster,
            HARForecaster,
            GarchForecaster,
            vix_factory,
        ],
    )

    # ------------------------------------------------------------
    # 7. Build every evaluation table from the same panel
    # ------------------------------------------------------------
    report = build_forecast_report(
        panel,
        baseline=BASELINE,
    )

    # ------------------------------------------------------------
    # 8. Persist the panel and report tables
    # ------------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.frame.to_parquet(PANEL_PATH)

    report.overall_qlike.to_csv(OVERALL_QLIKE_PATH)
    report.overall_mse.to_csv(OVERALL_MSE_PATH)

    report.qlike_by_year.to_csv(QLIKE_BY_YEAR_PATH)
    report.mse_by_year.to_csv(MSE_BY_YEAR_PATH)

    report.mz.to_csv(MZ_PATH)
    report.dm_qlike.to_csv(DM_QLIKE_PATH)
    report.dm_mse.to_csv(DM_MSE_PATH)

    manifest = {
        "spx_symbol": SPX_SYMBOL,
        "vix_symbol": VIX_SYMBOL,
        "variance_method": VARIANCE_METHOD,
        "horizon": HORIZON,
        "weekly_window": WEEKLY_WINDOW,
        "monthly_window": MONTHLY_WINDOW,
        "first_cutoff": str(FIRST_CUTOFF.date()),
        "refit_every": REFIT_EVERY,
        "baseline": BASELINE,
        "n_folds": len(folds),
        "n_forecasts": len(panel.frame),
        "models": list(panel.model_names),
        "dataset_start": str(ds.X.index.min().date()),
        "dataset_end": str(ds.X.index.max().date()),
        "oos_start": str(panel.frame.index.min().date()),
        "oos_end": str(panel.frame.index.max().date()),
        "vix_conversion": {
            "calendar_horizon_days": 30,
            "trading_horizon_days": HORIZON,
            "annual_calendar_days": 365,
        },
        "artifacts": {
            "panel": str(PANEL_PATH),
            "overall_qlike": str(OVERALL_QLIKE_PATH),
            "overall_mse": str(OVERALL_MSE_PATH),
            "qlike_by_year": str(QLIKE_BY_YEAR_PATH),
            "mse_by_year": str(MSE_BY_YEAR_PATH),
            "mz_calibration": str(MZ_PATH),
            "dm_qlike": str(DM_QLIKE_PATH),
            "dm_mse": str(DM_MSE_PATH),
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # 9. Print the research report
    # ------------------------------------------------------------
    print_header("DATASET")
    print(f"Symbol: {SPX_SYMBOL}")
    print(f"VIX benchmark: {VIX_SYMBOL}")
    print(f"Variance method: {VARIANCE_METHOD}")
    print(f"Full dataset rows: {len(ds.X):,}")
    print(
        "Feature range: "
        f"{ds.X.index.min().date()} "
        f"to {ds.X.index.max().date()}"
    )

    print_header("WALK-FORWARD")
    print(f"First cutoff: {FIRST_CUTOFF.date()}")
    print(f"Refit every: {REFIT_EVERY} forecast origins")
    print(f"Number of folds: {len(folds):,}")
    print(f"Out-of-sample forecasts: {len(panel.frame):,}")
    print(
        "OOS range: "
        f"{panel.frame.index.min().date()} "
        f"to {panel.frame.index.max().date()}"
    )

    print_header("MODEL COLUMNS")
    print(panel.model_names)

    print_header("OVERALL QLIKE, LOWER IS BETTER")
    print(report.overall_qlike)

    print_header("OVERALL MSE, LOWER IS BETTER")
    print(report.overall_mse)

    print_header("QLIKE BY YEAR, LOWER IS BETTER")
    print(report.qlike_by_year)

    print_header("MSE BY YEAR, LOWER IS BETTER")
    print(report.mse_by_year)

    print_header("MINCER-ZARNOWITZ CALIBRATION")
    print(report.mz)

    print_header("DIEBOLD-MARIANO: QLIKE")
    print(report.dm_qlike)

    print_header("DIEBOLD-MARIANO: MSE")
    print(report.dm_mse)

    print_header("PERSISTED OUTPUT")
    print(f"Panel: {PANEL_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Overall QLIKE: {OVERALL_QLIKE_PATH}")
    print(f"Overall MSE: {OVERALL_MSE_PATH}")
    print(f"QLIKE by year: {QLIKE_BY_YEAR_PATH}")
    print(f"MSE by year: {MSE_BY_YEAR_PATH}")
    print(f"MZ calibration: {MZ_PATH}")
    print(f"DM QLIKE: {DM_QLIKE_PATH}")
    print(f"DM MSE: {DM_MSE_PATH}")


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


if __name__ == "__main__":
    main()
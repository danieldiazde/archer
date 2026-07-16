from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import numbers

from archer.analytics.losses import loss_frame
from archer.analytics.walkforward import run_walkforward
from archer.analytics.dm import pairwise_dm_table
from archer.analytics.mz import mz_table
from archer.data.ingest import load_ingest_config
from archer.data.returns import make_log_returns, make_price_matrix
from archer.data.store import ParquetStore
from archer.features.realized_vol import daily_variance
from archer.models.dataset import build_vol_dataset
from archer.models.fold import make_expanding_folds
from archer.models.har import HARForecaster
from archer.models.garch import GarchForecaster

from functools import partial

from archer.features.implied_vol import vix_to_daily_variance
from archer.models.benchmarks import (
    AlignedSeriesForecaster,
    NaiveMA22Forecaster,
)


SPX_SYMBOL = "^GSPC"
VIX_SYMBOL = "^VIX"

VARIANCE_METHOD = "gk_total"

HORIZON = 21
WEEKLY_WINDOW = 5
MONTHLY_WINDOW = 22

FIRST_CUTOFF = pd.Timestamp("2013-12-31")
REFIT_EVERY = 21

OUTPUT_DIR = Path("data/evals")
PANEL_PATH = OUTPUT_DIR / "forecast_panel.parquet"
MANIFEST_PATH = OUTPUT_DIR / "forecast_panel.manifest.json"

MZ_PATH = OUTPUT_DIR / "mz_calibration.csv"
DM_QLIKE_PATH = OUTPUT_DIR / "dm_qlike.csv"
DM_MSE_PATH = OUTPUT_DIR / "dm_mse.csv"
QLIKE_BY_YEAR_PATH = OUTPUT_DIR / "qlike_by_year.csv"
MSE_BY_YEAR_PATH = OUTPUT_DIR / "mse_by_year.csv"


def main() -> None:
    cfg = load_ingest_config("config/data.yaml")

    store = ParquetStore(
        bronze_dir=cfg.bronze_dir,
        silver_dir=cfg.silver_dir,
    )

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

    ds = build_vol_dataset(
        variance=variance,
        returns=returns,
        horizon=HORIZON,
        weekly_window=WEEKLY_WINDOW,
        monthly_window=MONTHLY_WINDOW,
    )

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

    # Require a VIX observation for every forecastable SPX row.
    vix_implied = vix_implied.reindex(ds.X.index)

    missing_vix = vix_implied.index[
        vix_implied.isna()
    ]

    if not missing_vix.empty:
        raise RuntimeError(
            "VIX-implied forecast is missing required SPX dates. "
            f"First missing dates: {missing_vix[:10].tolist()}"
        )

    folds = make_expanding_folds(
        ds,
        first_cutoff=FIRST_CUTOFF,
        refit_every=REFIT_EVERY,
    )

    vix_factory = partial(
        AlignedSeriesForecaster,
        name="vix_implied",
        series=vix_implied,
    )

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

    qlike = loss_frame(
        panel,
        loss="qlike",
    )

    mse = loss_frame(
        panel,
        loss="mse",
    )

    mz_summary = mz_table(panel)

    dm_qlike = pairwise_dm_table(
        qlike,
        horizon=panel.horizon,
    )

    dm_mse = pairwise_dm_table(
        mse,
        horizon=panel.horizon,
    )

    qlike_summary = summarize_losses(
        qlike,
        baseline="naive_ma22",
    )

    mse_summary = summarize_losses(
        mse,
        baseline="naive_ma22",
    )

    qlike_by_year = summarize_losses_by_year(
        qlike,
        baseline="naive_ma22",
    )

    mse_by_year = summarize_losses_by_year(
        mse,
        baseline="naive_ma22",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.frame.to_parquet(PANEL_PATH)
    mz_summary.to_csv(MZ_PATH)
    dm_qlike.to_csv(DM_QLIKE_PATH)
    dm_mse.to_csv(DM_MSE_PATH)
    qlike_by_year.to_csv(QLIKE_BY_YEAR_PATH)
    mse_by_year.to_csv(MSE_BY_YEAR_PATH)

    manifest = {
        "symbol": SPX_SYMBOL,
        "vix_symbol": VIX_SYMBOL,
        "variance_method": VARIANCE_METHOD,
        "vix_conversion": {
            "calendar_horizon_days": 30,
            "trading_horizon_days": HORIZON,
            "annual_calendar_days": 365,
        },
        "horizon": HORIZON,
        "weekly_window": WEEKLY_WINDOW,
        "monthly_window": MONTHLY_WINDOW,
        "first_cutoff": str(FIRST_CUTOFF.date()),
        "refit_every": REFIT_EVERY,
        "n_folds": len(folds),
        "n_forecasts": len(panel.frame),
        "panel_path": str(PANEL_PATH),
        "models": list(panel.model_names),
        "dataset_start": str(ds.X.index.min().date()),
        "dataset_end": str(ds.X.index.max().date()),
        "oos_start": str(panel.frame.index.min().date()),
        "oos_end": str(panel.frame.index.max().date()),
        "artifacts": {
        "panel": str(PANEL_PATH),
        "mz_calibration": str(MZ_PATH),
        "dm_qlike": str(DM_QLIKE_PATH),
        "dm_mse": str(DM_MSE_PATH),
        "qlike_by_year": str(QLIKE_BY_YEAR_PATH),
        "mse_by_year": str(MSE_BY_YEAR_PATH),
    },
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

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
    print(qlike_summary)

    print_header("OVERALL MSE, LOWER IS BETTER")
    print(mse_summary)

    print_header("QLIKE BY YEAR, LOWER IS BETTER")
    print(qlike_by_year)

    print_header("MSE BY YEAR, LOWER IS BETTER")
    print(mse_by_year)

    print_header("MINCER-ZARNOWITZ CALIBRATION")
    print(mz_summary)

    print_header("DIEBOLD-MARIANO: QLIKE")
    print(dm_qlike)

    print_header("DIEBOLD-MARIANO: MSE")
    print(dm_mse)

    print_header("PERSISTED OUTPUT")
    print(f"Panel: {PANEL_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")

    print(f"MZ calibration: {MZ_PATH}")
    print(f"DM QLIKE: {DM_QLIKE_PATH}")
    print(f"DM MSE: {DM_MSE_PATH}")
    print(f"QLIKE by year: {QLIKE_BY_YEAR_PATH}")
    print(f"MSE by year: {MSE_BY_YEAR_PATH}")


def summarize_losses(
    losses: pd.DataFrame,
    *,
    baseline: str,
) -> pd.DataFrame:
    if baseline not in losses.columns:
        raise ValueError(
            f"baseline {baseline!r} is not in losses."
        )

    mean_loss = losses.mean()
    median_loss = losses.median()

    summary = pd.DataFrame(
        {
            "mean_loss": mean_loss,
            "median_loss": median_loss,
        }
    )

    baseline_loss_raw = summary.at[baseline, "mean_loss"]

    if not isinstance(baseline_loss_raw, numbers.Real):
        raise TypeError(
            f"baseline mean_loss must be a real number, got {type(baseline_loss_raw).__name__}"
        )

    baseline_loss = float(baseline_loss_raw)

    summary["relative_to_baseline"] = (
        summary["mean_loss"].astype(float).div(baseline_loss)
    )

    summary["improvement_vs_baseline"] = (
        1.0 - summary["relative_to_baseline"]
    )

    return summary.sort_values("mean_loss")


def summarize_losses_by_year(
    losses: pd.DataFrame,
    *,
    baseline: str,
) -> pd.DataFrame:
    if baseline not in losses.columns:
        raise ValueError(
            f"baseline {baseline!r} is not in losses."
        )

    loss_dates = pd.DatetimeIndex(losses.index)

    yearly = losses.groupby(
        loss_dates.year,
    ).mean()

    relative = yearly.div(
        yearly[baseline],
        axis=0,
    )

    improvement = 1.0 - relative

    out = pd.concat(
        {
            "mean_loss": yearly,
            "relative_to_baseline": relative,
            "improvement_vs_baseline": improvement,
        },
        axis=1,
    )

    return out


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


if __name__ == "__main__":
    main()
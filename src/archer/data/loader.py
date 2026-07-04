import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)


def load_universe(path: str) -> list[str]:
    """Load ticker universe from a YAML config file."""

    config_path = Path(path)

    logger.info("Loading universe from %s", config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Universe config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if "universe" not in config:
        raise ValueError("Universe config must contain a 'universe' key.")

    symbols = []

    for asset_class, tickers in config["universe"].items():
        logger.debug("Loading %s tickers: %s", asset_class, tickers)

        if not isinstance(tickers, list):
            raise ValueError(f"Universe group '{asset_class}' must be a list.")

        symbols.extend(tickers)

    symbols = [symbol.upper() for symbol in symbols]

    if len(symbols) == 0:
        raise ValueError("Universe is empty.")

    if len(symbols) != len(set(symbols)):
        raise ValueError("Universe contains duplicate symbols.")

    logger.info("Loaded %d symbols", len(symbols))

    return symbols


def download_ohlcv(
    symbols: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame | None:
    """Download OHLCV data from Yahoo Finance."""

    logger.info(
        "Downloading OHLCV data for %d symbols from %s to %s",
        len(symbols),
        start,
        end,
    )

    data: pd.DataFrame | None = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )

    if data is None:
        logger.warning("Yahoo Finance returned no data for given symbols/date range")
    else:
        logger.info("Raw Yahoo Finance shape: %s", data.shape)

    return data


def save_raw_ohlcv(df: pd.DataFrame, path: str) -> None:
    """Save raw OHLCV data to parquet."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving raw OHLCV data to %s", output_path)

    df.to_parquet(output_path, index=False)

    logger.info("Saved %d rows and %d columns", df.shape[0], df.shape[1])
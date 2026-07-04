import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

YAHOO_TO_ARCHER_COLUMNS = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def load_universe(path: str) -> list[str]:
    """
    Load ticker symbols from a YAML universe config.

    Expected YAML format:
        universe:
          equities:
            - SPY
            - QQQ
          bonds:
            - IEF
            - TLT
    """

    config_path = Path(path)

    logger.info("Loading universe from %s", config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Universe config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Universe config must be a YAML dictionary.")

    if "universe" not in config:
        raise ValueError("Universe config must contain a 'universe' key.")

    universe = config["universe"]

    if not isinstance(universe, dict):
        raise ValueError("'universe' must be a dictionary of asset groups.")

    symbols = []

    for asset_class, tickers in universe.items():
        if not isinstance(tickers, list):
            raise ValueError(f"Universe group '{asset_class}' must be a list.")

        for ticker in tickers:
            if not isinstance(ticker, str):
                raise ValueError(f"Ticker {ticker} in group '{asset_class}' must be a string.")

            symbol = ticker.strip().upper()

            if symbol == "":
                raise ValueError(f"Found empty ticker in group '{asset_class}'.")

            symbols.append(symbol)

    if len(symbols) == 0:
        raise ValueError("Universe is empty.")

    duplicate_symbols = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
    if duplicate_symbols:
        raise ValueError(f"Universe contains duplicate symbols: {duplicate_symbols}")

    logger.info("Loaded %d symbols: %s", len(symbols), symbols)

    return symbols


def download_ohlcv(
    symbols: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance and return Archer's long-format schema.

    Output schema:
        date, symbol, open, high, low, close, adj_close, volume
    """

    if len(symbols) == 0:
        raise ValueError("Cannot download OHLCV data for an empty symbol list.")

    logger.info(
        "Downloading OHLCV data for %d symbols from %s to %s",
        len(symbols),
        start,
        end,
    )

    frames = []

    for symbol in symbols:
        logger.info("Downloading %s", symbol)

        raw_symbol_data = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )

        if raw_symbol_data is None or raw_symbol_data.empty:
            logger.warning("No data returned for %s", symbol)
            continue

        symbol_data = _standardize_yahoo_frame(raw_symbol_data, symbol)

        logger.info("Downloaded %d rows for %s", len(symbol_data), symbol)

        frames.append(symbol_data)

    if len(frames) == 0:
        raise ValueError("No OHLCV data was downloaded for any symbol.")

    ohlcv = pd.concat(frames, ignore_index=True)

    ohlcv = ohlcv[REQUIRED_COLUMNS]
    ohlcv = ohlcv.sort_values(["symbol", "date"]).reset_index(drop=True)

    logger.info(
        "Finished downloading OHLCV data with %d rows and %d columns",
        ohlcv.shape[0],
        ohlcv.shape[1],
    )

    return ohlcv


def save_raw_ohlcv(df: pd.DataFrame, path: str) -> None:
    """
    Save raw OHLCV data to a parquet file.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving raw OHLCV data to %s", output_path)

    df.to_parquet(output_path, index=False)

    logger.info("Saved raw OHLCV data with %d rows", len(df))


def _standardize_yahoo_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Convert a single-symbol Yahoo Finance DataFrame into Archer's long format.
    """

    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    data = data.rename(columns=YAHOO_TO_ARCHER_COLUMNS)

    missing_columns = set(REQUIRED_COLUMNS) - (set(data.columns) | {"symbol"})
    if missing_columns:
        raise ValueError(f"Downloaded data for {symbol} is missing columns: {sorted(missing_columns)}")

    data["symbol"] = symbol

    return data[REQUIRED_COLUMNS]
import logging

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]
NUMERIC_COLUMNS = PRICE_COLUMNS + ["volume"]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw OHLCV data into Archer's standard schema.

    Final schema:
        date, symbol, open, high, low, close, adj_close, volume
    """

    logger.info("Cleaning OHLCV data with shape %s", df.shape)

    clean = df.copy()

    clean.columns = (
        clean.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    missing_columns = set(EXPECTED_COLUMNS) - set(clean.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    clean = clean[EXPECTED_COLUMNS]

    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["symbol"] = clean["symbol"].astype(str).str.strip().str.upper()

    for column in NUMERIC_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    before = len(clean)
    clean = clean.drop_duplicates(subset=["date", "symbol"], keep="last")
    dropped = before - len(clean)

    if dropped > 0:
        logger.warning("Dropped %d duplicate date-symbol rows", dropped)

    clean = clean.sort_values(["symbol", "date"]).reset_index(drop=True)

    logger.info("Finished cleaning OHLCV data with shape %s", clean.shape)

    return clean


def validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Validate that OHLCV data is structurally and financially sane.

    Raises:
        ValueError: if the data violates required OHLCV rules.
    """

    logger.info("Validating OHLCV data with shape %s", df.shape)

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df["date"].isna().any():
        raise ValueError("Found missing or invalid dates.")

    if df["symbol"].isna().any():
        raise ValueError("Found missing symbols.")

    if (df["symbol"].astype(str).str.strip() == "").any():
        raise ValueError("Found empty symbols.")

    duplicate_count = df.duplicated(subset=["date", "symbol"]).sum()
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate date-symbol rows.")

    if df[PRICE_COLUMNS].isna().any().any():
        raise ValueError("Found missing price values.")

    if df["volume"].isna().any():
        raise ValueError("Found missing volume values.")

    if (df[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("Found non-positive prices.")

    if (df["volume"] < 0).any():
        raise ValueError("Found negative volume.")

    if (df["high"] < df["low"]).any():
        raise ValueError("Found rows where high < low.")

    if (df["high"] < df["open"]).any():
        raise ValueError("Found rows where high < open.")

    if (df["high"] < df["close"]).any():
        raise ValueError("Found rows where high < close.")

    if (df["low"] > df["open"]).any():
        raise ValueError("Found rows where low > open.")

    if (df["low"] > df["close"]).any():
        raise ValueError("Found rows where low > close.")

    sorted_df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    current_df = df.reset_index(drop=True)

    if not current_df[["symbol", "date"]].equals(sorted_df[["symbol", "date"]]):
        raise ValueError("Data must be sorted by symbol and date.")

    zero_volume_count = (df["volume"] == 0).sum()
    if zero_volume_count > 0:
        logger.warning("Found %d rows with zero volume", zero_volume_count)

    logger.info("OHLCV validation passed")


def save_clean_ohlcv(df: pd.DataFrame, path: str) -> None:
    """
    Save clean OHLCV data to parquet.
    """

    logger.info("Saving clean OHLCV data to %s", path)

    df.to_parquet(path, index=False)

    logger.info("Saved clean OHLCV data with %d rows", len(df))
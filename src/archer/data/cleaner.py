from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .universe import Instrument

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CoercionIssue:
    symbol: str
    column: str
    row_index: object
    raw_value: object
    message: str


@dataclass(frozen=True, slots=True)
class CleanResult:
    df: pd.DataFrame
    issues: list[CoercionIssue]


def clean_ohlcv(raw: pd.DataFrame, inst: Instrument) -> CleanResult:
    df = raw.copy()
    issues: list[CoercionIssue] = []

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    required_cols = {"date", "open", "high", "low", "close"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing_cols)}")

    df["symbol"] = inst.symbol

    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    if "volume" not in df.columns:
        df["volume"] = np.nan

    raw_date = df["date"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    bad_dates = raw_date.notna() & df["date"].isna()

    for idx in df.index[bad_dates]:
        issues.append(
            CoercionIssue(
                symbol=inst.symbol,
                column="date",
                row_index=idx,
                raw_value=raw_date.loc[idx],
                message="Could not parse date.",
            )
        )

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]

    for col in numeric_cols:
        raw_values = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")

        bad_values = raw_values.notna() & df[col].isna()

        for idx in df.index[bad_values]:
            issues.append(
                CoercionIssue(
                    symbol=inst.symbol,
                    column=col,
                    row_index=idx,
                    raw_value=raw_values.loc[idx],
                    message=f"Could not parse numeric value in {col}.",
                )
            )

    df = df[
        ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
    ]

    df = df.drop_duplicates()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return CleanResult(df=df, issues=issues)
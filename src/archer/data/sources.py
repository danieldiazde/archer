from __future__ import annotations

from typing import Protocol
from universe import Instrument
from datetime import date
from dataclasses import dataclass
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import logging

logger = logging.getLogger(__name__)

class SourceError(RuntimeError):
    """A data source cannot fetch usable data."""

@dataclass(frozen=True, slots=True)
class FetchResult:
    source : str
    df : pd.DataFrame

class Source(Protocol):
    name : str

    def fetch(
            self,
            inst : Instrument,
            start : date,
            end : date | None,
    ) -> pd.DataFrame:
        ...

class YFinanceSource:
    name = "yfinance"

    def fetch(
            self,
            inst: Instrument,
            start: date,
            end: date | None = None,
        ) -> pd.DataFrame:
            raw = yf.download(
                tickers=inst.symbol,
                start=start.isoformat(),
                end=end.isoformat() if end is not None else None,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                group_by="column",
            )

            if not isinstance(raw, pd.DataFrame) or raw.empty:
                raise SourceError(f"yfinance failed to return data or returned empty for {inst.symbol}.")

            processed_df = self._flatten_columns(raw)
            processed_df = processed_df.reset_index()

            return processed_df
    
    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This keeps the price field level and drops the ticker level.
        """
        if not isinstance(df.columns, pd.MultiIndex):
            return df

        # Common shape for one ticker:
        # level 0 = price fields, level 1 = ticker
        if len(df.columns.levels) >= 2:
            flattened = df.copy()
            flattened.columns = [
                str(col[0]) if isinstance(col, tuple) else str(col)
                for col in flattened.columns
            ]
            return flattened

        return df
    
class CboeVixSource:
    name = "cboe"

    vix_csv_url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

    def fetch(
        self,
        inst: Instrument,
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame:
        if inst.symbol.upper() != "^VIX":
            raise SourceError(
                f"CBOE VIX source only supports ^VIX, got {inst.symbol}."
            )

        response = requests.get(self.vix_csv_url, timeout=30)

        if response.status_code != 200:
            raise SourceError(
                f"CBOE request failed for {inst.symbol}: "
                f"HTTP {response.status_code}"
            )

        raw = pd.read_csv(StringIO(response.text))

        if raw.empty:
            raise SourceError("CBOE returned no VIX rows.")

        raw = self._normalize_cboe_columns(raw)

        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw[raw["date"].dt.date >= start]

        if end is not None:
            raw = raw[raw["date"].dt.date < end]

        if raw.empty:
            raise SourceError(
                f"CBOE returned no VIX rows in requested range "
                f"{start.isoformat()} to {end.isoformat() if end else 'latest'}."
            )

        # CBOE VIX has no meaningful volume and usually no adjusted close.
        raw["adj_close"] = raw["close"]
        raw["volume"] = pd.NA

        return raw

    def _normalize_cboe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = df.copy()

        clean.columns = (
            clean.columns
            .str.lower()
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        rename_map = {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
        }

        clean = clean.rename(columns=rename_map)

        required = {"date", "open", "high", "low", "close"}
        missing = required - set(clean.columns)

        if missing:
            raise SourceError(f"CBOE VIX data missing columns: {sorted(missing)}")

        return clean[["date", "open", "high", "low", "close"]]


class SourceResolver:
    def __init__(self, sources: list[Source]) -> None:
        self.sources = {source.name: source for source in sources}

    def fetch(
        self,
        inst: Instrument,
        start: date,
        end: date | None = None,
    ) -> FetchResult:
        errors: list[str] = []

        for source_name in inst.source_priority:
            source = self.sources.get(source_name)

            if source is None:
                errors.append(f"{source_name}: source is not registered")
                continue

            try:
                df = source.fetch(inst=inst, start=start, end=end)
                return FetchResult(source=source.name, df=df)
            except Exception as exc:
                errors.append(f"{source_name}: {exc}")

        joined_errors = "; ".join(errors)

        raise SourceError(
            f"All sources failed for {inst.symbol}. Errors: {joined_errors}"
        )


def build_default_resolver() -> SourceResolver:
    return SourceResolver(
        sources=[
            YFinanceSource(),
            CboeVixSource(),
        ]
    )
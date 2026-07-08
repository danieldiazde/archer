from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Literal, Protocol, Any, cast
from .universe import Instrument
import logging
import pandas_market_calendars as mcal
from datetime import date, datetime
import yaml
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class GateResult:
    gate : str
    symbol : str
    status : Literal["ok", "flagged", "failed"]
    detail : str = ""
    bad_rows : pd.Index | None = None

class Gate(Protocol):
    name : str
    def check(self, inst: Instrument, df: pd.DataFrame) -> GateResult: ...

class UniqueKeyGate:
    name = "unique_key"

    def check(self, inst: Instrument, df: pd.DataFrame) -> GateResult:
        duplicated = df["date"].duplicated(keep=False)

        if duplicated.any():
            return GateResult(
                gate = self.name,
                symbol= inst.symbol,
                status = "failed",
                detail = "Duplicate dates found.",
                bad_rows = df.index[duplicated]
            )

        return GateResult(
            gate = self.name,
            symbol = inst.symbol,
            status = "ok",
        )

class OhlcCoherentGate:
    name = "ohlc_coherent"

    def check(self, inst: Instrument, df: pd.DataFrame) -> GateResult:
        required_cols = ["open", "high", "low", "close"]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail=f"Missing OHLC columns: {missing_cols}",
            )

        high_bad = (
            (df["high"] < df["open"])
            | (df["high"] < df["low"])
            | (df["high"] < df["close"])
        )

        low_bad = (
            (df["low"] > df["open"])
            | (df["low"] > df["high"])
            | (df["low"] > df["close"])
        )

        bad_rows = high_bad | low_bad

        if bad_rows.any():
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail="OHLC values are incoherent: high must be >= open/low/close and low must be <= open/high/close.",
                bad_rows=df.index[bad_rows],
            )

        return GateResult(
            gate=self.name,
            symbol=inst.symbol,
            status="ok",
        )

class PositivePricesGate:
    name = "positive_prices"

    def check(self, inst: Instrument, df: pd.DataFrame) -> GateResult:
        price_cols = ["open", "high", "low", "close", "adj_close"]

        missing_cols = [col for col in price_cols if col not in df.columns]
        if missing_cols:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail=f"Missing price columns: {missing_cols}",
            )

        bad_masks: list[pd.Series] = []

        for col in price_cols:
            values = pd.to_numeric(df[col], errors="coerce")

            non_numeric = df[col].notna() & values.isna()
            if non_numeric.any():
                return GateResult(
                    gate=self.name,
                    symbol=inst.symbol,
                    status="failed",
                    detail=f"Column {col!r} contains non-numeric price values.",
                    bad_rows=df.index[non_numeric],
                )

            non_positive = values <= 0
            bad_masks.append(non_positive)

        bad_rows = bad_masks[0]
        for mask in bad_masks[1:]:
            bad_rows = bad_rows | mask

        if bad_rows.any():
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail="Price columns must be strictly positive.",
                bad_rows=df.index[bad_rows],
            )

        return GateResult(
            gate=self.name,
            symbol=inst.symbol,
            status="ok",
        )

class SortedKeyGate:
    name = 'sorted_key'

    def check(self, inst: Instrument, df: pd.DataFrame) -> GateResult:
        unsorted = not df['date'].is_monotonic_increasing

        if unsorted:
            bad_mask = df['date'] < df['date'].shift(1)

            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status='failed',
                detail='Dates must be sorted in increasing order.',
                bad_rows=df.index[bad_mask]
            )
        
        return GateResult(
            gate=self.name,
            symbol=inst.symbol,
            status='ok'
        )

class VolumeNonnegGate:
    name = 'nonneg_volume'

    def check(self, inst: Instrument, df: pd.DataFrame) -> GateResult:
        negative_volume = df['volume'] < 0

        if negative_volume.any():
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status='failed',
                detail='Volume must be greater or equal to zero.',
                bad_rows=df.index[negative_volume]
            )
        
        return GateResult(
            gate=self.name,
            symbol = inst.symbol,
            status='ok'
        )

class CalendarCompleteGate:
    name = 'calendar_complete'

    def __init__(self, calendar_name : str = 'XNYS') -> None:
        self.calendar_name = calendar_name
        self.calendar = mcal.get_calendar(calendar_name)

    def check(self, inst:Instrument, df: pd.DataFrame) -> GateResult:
        if "date" not in df.columns:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail="Missing required column: date.",
            )
        if df.empty:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail="No rows found for instrument.",
            )
        
        dates = pd.to_datetime(df["date"], errors="coerce")

        bad_dates = dates.isna()
        if bad_dates.any():
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail="Some dates could not be parsed.",
                bad_rows=df.index[bad_dates],
            )
        
        observed_dates = pd.DatetimeIndex(
            dates.dt.normalize().unique()
        ).sort_values()

        if len(observed_dates) <= 1:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="ok",
                detail="One or zero observed dates; no internal calendar gaps to check.",
            )

        start_date = observed_dates.min().date()
        end_date = observed_dates.max().date()

        schedule = self.calendar.schedule(
            start_date=start_date,
            end_date=end_date,
        )

        expected_dates = pd.DatetimeIndex(schedule.index).normalize()

        missing_dates = expected_dates.difference(observed_dates)

        if len(missing_dates) > 0:
            preview = [d.date().isoformat() for d in missing_dates[:5]]

            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail=(
                    f"Missing {len(missing_dates)} expected {self.calendar_name} "
                    f"sessions between {start_date} and {end_date}. "
                    f"First missing dates: {preview}"
                ),
                bad_rows=missing_dates,
            )

        return GateResult(
            gate=self.name,
            symbol=inst.symbol,
            status="ok",
            detail=f"All expected {self.calendar_name} sessions are present.",
        )

class CoverageGate:
    name = "coverage"

    def __init__(self, calendar_name: str = "XNYS") -> None:
        self.calendar_name = calendar_name
        self.calendar = mcal.get_calendar(calendar_name)

    def check(self, inst: Instrument, df: pd.DataFrame) -> GateResult:
        if "date" not in df.columns:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail="Missing required column: date.",
            )

        if df.empty:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail=(
                    f"No rows found. Expected coverage starting from "
                    f"{inst.expected_start.isoformat()}."
                ),
            )

        dates = pd.to_datetime(df["date"], errors="coerce")

        bad_dates = dates.isna()
        if bad_dates.any():
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail="Some dates could not be parsed.",
                bad_rows=df.index[bad_dates],
            )

        observed_dates = pd.DatetimeIndex(
            dates.dt.normalize().unique()
        ).sort_values()

        first_observed = observed_dates.min().date()
        last_observed = observed_dates.max().date()

        schedule = self.calendar.schedule(
            start_date=inst.expected_start,
            end_date=last_observed,
        )

        expected_sessions = pd.DatetimeIndex(schedule.index).normalize()

        if len(expected_sessions) == 0:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="flagged",
                detail=(
                    f"No expected {self.calendar_name} sessions found between "
                    f"{inst.expected_start.isoformat()} and {last_observed.isoformat()}."
                ),
            )

        expected_first = expected_sessions.min().date()

        if first_observed > expected_first:
            missing_before_start = expected_sessions[
                expected_sessions.date < first_observed
            ]

            preview = [
                d.date().isoformat()
                for d in missing_before_start[:5]
            ]

            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail=(
                    f"Coverage starts late. Expected first session on "
                    f"{expected_first.isoformat()}, but first observed date is "
                    f"{first_observed.isoformat()}. "
                    f"Missing {len(missing_before_start)} expected sessions before "
                    f"first observation. First missing dates: {preview}"
                ),
                bad_rows=missing_before_start,
            )

        if first_observed < expected_first:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="flagged",
                detail=(
                    f"Coverage starts earlier than configured. Expected first session "
                    f"on {expected_first.isoformat()}, but first observed date is "
                    f"{first_observed.isoformat()}. Check expected_start in universe.yaml."
                ),
            )

        return GateResult(
            gate=self.name,
            symbol=inst.symbol,
            status="ok",
            detail=(
                f"Coverage starts on expected first session: "
                f"{expected_first.isoformat()}."
            ),
        )

class OutliersVsWhiteListGate:
    name = "outliers_vs_whitelist"

    def __init__(
        self,
        thresholds: dict[str, float],
        events_path: str | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.whitelist: set[tuple[str, date]] = (
            self._load_whitelist(events_path)
            if events_path is not None
            else set()
        )

    def _parse_date(self, value: Any) -> date:
        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        if isinstance(value, str):
            try:
                return datetime.strptime(value.strip(), "%Y-%m-%d").date()
            except ValueError as exc:
                raise ValueError(
                    f"Invalid event date format: {value!r}. Expected YYYY-MM-DD."
                ) from exc

        raise ValueError(f"Invalid event date: {value!r}")

    def _load_whitelist(self, events_path: str) -> set[tuple[str, date]]:
        path = Path(events_path)

        if not path.exists():
            raise FileNotFoundError(f"Events whitelist not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            config: Any = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError("events.yaml must be a YAML dictionary.")

        raw_events = config.get("events", [])

        if not isinstance(raw_events, list):
            raise ValueError("'events' must be a list.")

        whitelist: set[tuple[str, date]] = set()

        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                raise ValueError("Each event must be a dictionary.")

            if "date" not in raw_event:
                raise ValueError("Each event must contain a date.")

            if "symbols" not in raw_event:
                raise ValueError("Each event must contain symbols.")

            event_date = self._parse_date(raw_event["date"])
            raw_symbols = raw_event["symbols"]

            if not isinstance(raw_symbols, list):
                raise ValueError("Event symbols must be a list.")

            for raw_symbol in raw_symbols:
                if not isinstance(raw_symbol, str):
                    raise ValueError("Event symbols must be strings.")

                symbol = raw_symbol.strip().upper()
                if not symbol:
                    raise ValueError("Event symbols cannot be empty.")

                whitelist.add((symbol, event_date))

        return whitelist

    def check(self, inst: Instrument, df: pd.DataFrame) -> GateResult:
        required_cols = ["date", "adj_close"]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail=f"Missing required columns: {missing_cols}",
            )

        threshold = self.thresholds.get(inst.exposure_family)
        if threshold is None:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail=(
                    f"No outlier threshold configured for "
                    f"exposure_family={inst.exposure_family!r}."
                ),
            )

        if df.empty:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail="No rows found for instrument.",
            )

        dates = cast(
            pd.Series,
            pd.to_datetime(df["date"], errors="coerce"),
        )

        bad_dates = dates.isna()
        if bad_dates.any():
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail="Some dates could not be parsed.",
                bad_rows=df.index[bad_dates],
            )

        adj_close = cast(
            pd.Series,
            pd.to_numeric(df["adj_close"], errors="coerce"),
        )

        bad_adj_close = adj_close.isna() | (adj_close <= 0)
        if bad_adj_close.any():
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail="adj_close must be numeric and positive to compute log returns.",
                bad_rows=df.index[bad_adj_close],
            )

        work = df.copy()
        work["_date"] = dates.dt.normalize()
        work["_adj_close"] = adj_close.astype(float)
        work = work.sort_values("_date")

        price = cast(pd.Series, work["_adj_close"])
        log_return = cast(
            pd.Series,
            np.log(price / price.shift(1)),
        )

        outlier_mask = cast(pd.Series, log_return.abs() > threshold)

        if not outlier_mask.any():
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="ok",
                detail=f"No abs(log return) above threshold {threshold}.",
            )

        outliers = work.loc[outlier_mask].copy()

        outlier_dates = cast(pd.Series, outliers["_date"]).dt.date

        unwhitelisted_mask = pd.Series(
            [
                (inst.symbol.upper(), outlier_date) not in self.whitelist
                for outlier_date in outlier_dates
            ],
            index=outliers.index,
        )

        unwhitelisted_rows = outliers.index[unwhitelisted_mask.to_numpy()]
        whitelisted_count = len(outliers) - len(unwhitelisted_rows)

        if len(unwhitelisted_rows) > 0:
            preview_dates = outlier_dates[unwhitelisted_mask].head(5).tolist()
            preview = [d.isoformat() for d in preview_dates]

            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="flagged",
                detail=(
                    f"Found {len(unwhitelisted_rows)} unwhitelisted outlier returns "
                    f"above threshold {threshold}. "
                    f"Whitelisted outliers: {whitelisted_count}. "
                    f"First unwhitelisted dates: {preview}"
                ),
                bad_rows=unwhitelisted_rows,
            )

        return GateResult(
            gate=self.name,
            symbol=inst.symbol,
            status="ok",
            detail=(
                f"All {len(outliers)} outlier returns above threshold "
                f"{threshold} are whitelisted."
            ),
        )

class PlausibilityGate:
    name = "plausibility"

    def __init__(
        self,
        bounds_by_symbol: dict[str, tuple[float, float]] | None = None,
        price_cols: tuple[str, ...] = ("open", "high", "low", "close", "adj_close"),
    ) -> None:
        self.bounds_by_symbol = bounds_by_symbol or {
            "^VIX": (5.0, 150.0),
        }
        self.price_cols = price_cols

    def check(self, inst: Instrument, df: pd.DataFrame) -> GateResult:
        symbol = inst.symbol.upper()

        if symbol not in self.bounds_by_symbol:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="ok",
                detail="No plausibility bounds configured for this symbol.",
            )

        if df.empty:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail="No rows found for instrument.",
            )

        lower, upper = self.bounds_by_symbol[symbol]

        available_cols = [col for col in self.price_cols if col in df.columns]

        if not available_cols:
            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="failed",
                detail=(
                    f"No price columns available for plausibility check. "
                    f"Expected at least one of {list(self.price_cols)}."
                ),
            )

        bad_masks: list[pd.Series] = []

        for col in available_cols:
            values = cast(
                pd.Series,
                pd.to_numeric(df[col], errors="coerce"),
            )

            non_numeric = values.isna() & df[col].notna()
            if non_numeric.any():
                return GateResult(
                    gate=self.name,
                    symbol=inst.symbol,
                    status="failed",
                    detail=f"Column {col!r} contains non-numeric values.",
                    bad_rows=df.index[non_numeric],
                )

            outside_bounds = (values < lower) | (values > upper)
            bad_masks.append(outside_bounds)

        bad_rows_mask = bad_masks[0]
        for mask in bad_masks[1:]:
            bad_rows_mask = bad_rows_mask | mask

        if bad_rows_mask.any():
            bad_rows = df.index[bad_rows_mask]

            preview_cols = ["date", *available_cols]
            preview_cols = [col for col in preview_cols if col in df.columns]

            preview = (
                df.loc[bad_rows, preview_cols]
                .head(5)
                .to_dict(orient="records")
            )

            return GateResult(
                gate=self.name,
                symbol=inst.symbol,
                status="flagged",
                detail=(
                    f"Values outside plausible range [{lower}, {upper}] "
                    f"for {symbol}. First examples: {preview}"
                ),
                bad_rows=bad_rows,
            )

        return GateResult(
            gate=self.name,
            symbol=inst.symbol,
            status="ok",
            detail=f"All checked values are within plausible range [{lower}, {upper}].",
        )

def run_gates(df : pd.DataFrame, instruments : list[Instrument], gates : list[Gate]) -> list[GateResult]:

    logger.info("Running gates on data with shape %s", df.shape)

    results : list[GateResult] = []

    # unique_key
    for inst in instruments:
        symbol_df = df[df["symbol"] == inst.symbol]

        for gate in gates:
            result = gate.check(inst, symbol_df)
            results.append(result)
    
    return results
   

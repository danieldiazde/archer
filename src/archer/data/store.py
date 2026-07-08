from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Any
import pandas as pd
from .universe import Instrument
import logging

logger = logging.getLogger(__name__)


class StoreError(Exception):
    """Base error for storage-layer failures."""


class MissingDataError(StoreError):
    """Raised when expected data or manifest files are missing."""


class IntegrityError(StoreError):
    """Raised when a file does not match its manifest hash."""


@dataclass(frozen=True, slots=True)
class Manifest:
    symbol: str
    layer: Literal["bronze", "silver"]
    source: str                     # "yfinance" | "cboe" | "silver-build"
    requested_start: str
    requested_end: str | None
    retrieved_start: str
    retrieved_end: str
    rows: int
    sha256: str
    created_at: str                 # UTC ISO-8601
    schema_version: int = 1
    inputs: tuple[str, ...] = ()    # silver only: sha256s of bronze parents
    gates: str = ""                 # silver only: "12 ok · 1 flagged"


def slug(symbol: str) -> str:
    clean = (
        symbol.strip()
        .upper()
        .replace("^", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )

    if not clean:
        raise ValueError(f"Invalid empty symbol: {symbol!r}")

    return clean


def assert_unique_slugs(instruments: list[Instrument]) -> None:
    seen: dict[str, str] = {}

    for inst in instruments:
        s = slug(inst.symbol)

        if s in seen:
            raise StoreError(
                f"Slug collision: {seen[s]!r} and {inst.symbol!r} both map to {s!r}"
            )

        seen[s] = inst.symbol


def utc_timestamp_for_path() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def manifest_path_for(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".manifest.json")


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """
    Write a parquet file atomically.

    The file is first written to a temporary sibling path, then moved into
    place with os.replace(). This prevents readers from seeing half-written
    files after a crash.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")

    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    """
    Write a JSON file atomically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")

    try:
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_manifest(manifest: Manifest, path: Path) -> None:
    """
    Write a manifest beside a parquet file.
    """
    payload = asdict(manifest)

    write_json_atomic(payload, path)


def read_manifest(path: Path) -> Manifest:
    """
    Read a manifest JSON file and reconstruct the Manifest dataclass.
    """
    path = Path(path)

    if not path.exists():
        raise MissingDataError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload: dict[str, Any] = json.load(file)

    if "inputs" in payload and isinstance(payload["inputs"], list):
        payload["inputs"] = tuple(payload["inputs"])

    return Manifest(**payload)


def read_verified_parquet(path: Path) -> tuple[pd.DataFrame, Manifest]:
    """
    Read a parquet file only if its manifest exists and its hash matches.
    """
    path = Path(path)
    manifest_path = manifest_path_for(path)

    if not path.exists():
        raise MissingDataError(f"Parquet file not found: {path}")

    if not manifest_path.exists():
        raise MissingDataError(
            f"Manifest missing for {path}. Refusing to read uncommitted data."
        )

    manifest = read_manifest(manifest_path)

    actual_hash = sha256_file(path)

    if actual_hash != manifest.sha256:
        raise IntegrityError(
            f"Hash mismatch for {path}. "
            f"Manifest has {manifest.sha256}, actual file has {actual_hash}."
        )

    df = pd.read_parquet(path)

    return df, manifest


class ParquetStore:
    def __init__(self, bronze_dir: Path, silver_dir: Path) -> None:
        self.bronze_dir = Path(bronze_dir)
        self.silver_dir = Path(silver_dir)

        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self.silver_dir.mkdir(parents=True, exist_ok=True)

    def bronze_symbol_dir(self, symbol: str) -> Path:
        return self.bronze_dir / slug(symbol)

    def silver_path(self, symbol: str) -> Path:
        return self.silver_dir / f"{slug(symbol)}.parquet"

    def write_bronze(
        self,
        df: pd.DataFrame,
        inst: Instrument,
        *,
        source: str,
        requested_start: str,
        requested_end: str | None = None,
    ) -> Manifest:
        timestamp = utc_timestamp_for_path()

        symbol_dir = self.bronze_symbol_dir(inst.symbol)
        path = symbol_dir / f"{timestamp}_{source}.parquet"

        write_parquet_atomic(df, path)

        manifest = build_manifest(
            df=df,
            path=path,
            symbol=inst.symbol,
            layer="bronze",
            source=source,
            requested_start=requested_start,
            requested_end=requested_end,
        )

        write_manifest(manifest, manifest_path_for(path))

        return manifest

    def write_silver(
        self,
        df: pd.DataFrame,
        inst: Instrument,
        *,
        inputs: tuple[str, ...],
        gate_summary: str,
    ) -> Manifest:
        path = self.silver_path(inst.symbol)

        write_parquet_atomic(df, path)

        manifest = build_manifest(
            df=df,
            path=path,
            symbol=inst.symbol,
            layer="silver",
            source="silver-build",
            requested_start=inst.expected_start.isoformat(),
            requested_end=None,
            inputs=inputs,
            gates=gate_summary,
        )

        write_manifest(manifest, manifest_path_for(path))

        return manifest
    
    def latest_bronze(self, symbol: str) -> tuple[pd.DataFrame, Manifest]:
        symbol_dir = self.bronze_symbol_dir(symbol)

        if not symbol_dir.exists():
            raise MissingDataError(
                f"No bronze directory found for {symbol}. Run ingest first."
            )

        files = sorted(symbol_dir.glob("*.parquet"))

        if not files:
            raise MissingDataError(
                f"No bronze parquet files found for {symbol}. Run ingest first."
            )

        latest_path = files[-1]

        return read_verified_parquet(latest_path)
    
    def read_silver(self, symbol: str) -> pd.DataFrame:
        path = self.silver_path(symbol)

        if not path.exists():
            raise MissingDataError(
                f"Silver data for {symbol} not found. Run ingest first."
            )

        df, _manifest = read_verified_parquet(path)

        return df

    def load_prices(
        self,
        symbols: list[str],
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for symbol in symbols:
            df = self.read_silver(symbol)
            frames.append(df)

        if not frames:
            raise MissingDataError("No symbols were provided to load_prices().")

        prices = pd.concat(frames, ignore_index=True)

        required_cols = {
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        }

        missing_cols = required_cols - set(prices.columns)
        if missing_cols:
            raise StoreError(
                f"Loaded silver data is missing required columns: {sorted(missing_cols)}"
            )

        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")

        bad_dates = prices["date"].isna()
        if bad_dates.any():
            raise StoreError("Loaded silver data contains unparseable dates.")

        if start is not None:
            start_ts = pd.Timestamp(start)
            prices = prices[prices["date"] >= start_ts]

        if end is not None:
            end_ts = pd.Timestamp(end)
            prices = prices[prices["date"] < end_ts]

        prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)

        return prices
    

def infer_date_range(df: pd.DataFrame) -> tuple[str, str]:
    """
    Infer retrieved start/end from a DataFrame with a date or Date column.
    """
    date_col = None

    for candidate in ("date", "Date"):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        raise StoreError("Cannot infer date range: no date column found.")

    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()

    if dates.empty:
        raise StoreError("Cannot infer date range: no valid dates found.")

    start = dates.min().date().isoformat()
    end = dates.max().date().isoformat()

    return start, end

def build_manifest(
    *,
    df: pd.DataFrame,
    path: Path,
    symbol: str,
    layer: Literal["bronze", "silver"],
    source: str,
    requested_start: str,
    requested_end: str | None,
    inputs: tuple[str, ...] = (),
    gates: str = "",
) -> Manifest:
    retrieved_start, retrieved_end = infer_date_range(df)

    return Manifest(
        symbol=symbol,
        layer=layer,
        source=source,
        requested_start=requested_start,
        requested_end=requested_end,
        retrieved_start=retrieved_start,
        retrieved_end=retrieved_end,
        rows=len(df),
        sha256=sha256_file(path),
        created_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
        gates=gates,
    )


from dataclasses import dataclass, fields, MISSING
from datetime import date, datetime
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast
import logging
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    name: str
    instrument_type: str
    exposure_family: str
    role: str
    tradable: bool
    source_priority: tuple[str, ...]
    expected_start: date
    direction: str | None = None


class InstrumentKwargs(TypedDict):
    symbol: str
    name: str
    instrument_type: str
    exposure_family: str
    role: str
    tradable: bool
    source_priority: tuple[str, ...]
    expected_start: date
    direction: NotRequired[str | None]


VALID_FIELDS = {f.name for f in fields(Instrument)}
REQUIRED_FIELDS = {
    f.name
    for f in fields(Instrument)
    if f.default is MISSING and f.default_factory is MISSING
}

VALID_GROUPS = {"signals", "tradables"}

VALID_ROLES = {
    "realized_vol_leg",
    "implied_vol_leg",
    "long_vol_trade",
    "short_vol_trade",
}

VALID_INSTRUMENT_TYPES = {"index", "etf", "etn"}

VALID_EXPOSURE_FAMILIES = {"equity", "volatility"}

VALID_SOURCES = {"yfinance", "cboe", "fred", "stooq"}


def _parse_date(value: Any, field_name: str) -> date:
    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"{field_name} must be in YYYY-MM-DD format.") from exc

    raise ValueError(f"{field_name} must be a date or YYYY-MM-DD string.")


def _require_str(value: Any, field_name: str, *, lowercase: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")

    value = value.strip()

    if not value:
        raise ValueError(f"{field_name} cannot be empty.")

    return value.lower() if lowercase else value


def _parse_source_priority(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("source_priority must be a list of strings.")

    if not value:
        raise ValueError("source_priority cannot be empty.")

    sources: list[str] = []

    for source in value:
        if not isinstance(source, str):
            raise ValueError("Every source_priority item must be a string.")

        clean_source = source.strip().lower()

        if clean_source not in VALID_SOURCES:
            raise ValueError(f"Invalid source: {clean_source}")

        sources.append(clean_source)

    return tuple(sources)


def _normalize_keys(raw: dict[Any, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}

    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"Universe field names must be strings. Got: {key!r}")

        clean_key = key.strip().lower()

        if clean_key in clean:
            raise ValueError(f"Duplicate field after normalization: {clean_key}")

        clean[clean_key] = value

    return clean


def _parse_instrument(raw: dict[Any, Any], asset_class: str) -> Instrument:
    ticker = _normalize_keys(raw)

    unknown_fields = set(ticker) - VALID_FIELDS
    if unknown_fields:
        raise ValueError(f"Unknown fields: {sorted(unknown_fields)}")

    missing_fields = REQUIRED_FIELDS - set(ticker)
    if missing_fields:
        raise ValueError(f"Missing required fields: {sorted(missing_fields)}")

    symbol = _require_str(ticker["symbol"], "symbol", lowercase=False).upper()
    name = _require_str(ticker["name"], "name", lowercase=False)
    instrument_type = _require_str(ticker["instrument_type"], "instrument_type")
    exposure_family = _require_str(ticker["exposure_family"], "exposure_family")
    role = _require_str(ticker["role"], "role")

    tradable_raw = ticker["tradable"]
    if not isinstance(tradable_raw, bool):
        raise ValueError("tradable must be a boolean.")
    tradable = tradable_raw

    source_priority = _parse_source_priority(ticker["source_priority"])
    expected_start = _parse_date(ticker["expected_start"], "expected_start")

    direction_raw = ticker.get("direction")
    if direction_raw is None:
        direction = None
    else:
        direction = _require_str(direction_raw, "direction")

    if instrument_type not in VALID_INSTRUMENT_TYPES:
        raise ValueError(f"Invalid instrument_type for {symbol}: {instrument_type}")

    if exposure_family not in VALID_EXPOSURE_FAMILIES:
        raise ValueError(f"Invalid exposure_family for {symbol}: {exposure_family}")

    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role for {symbol}: {role}")

    if asset_class == "signals" and tradable:
        raise ValueError(f"{symbol} is under signals but has tradable: true")

    if asset_class == "tradables" and not tradable:
        raise ValueError(f"{symbol} is under tradables but has tradable: false")

    clean_kwargs: dict[str, Any] = {
        "symbol": symbol,
        "name": name,
        "instrument_type": instrument_type,
        "exposure_family": exposure_family,
        "role": role,
        "tradable": tradable,
        "source_priority": source_priority,
        "expected_start": expected_start,
        "direction": direction,
    }

    kwargs = cast(InstrumentKwargs, clean_kwargs)
    return Instrument(**kwargs)


def load_universe(path: str) -> list[Instrument]:
    config_path = Path(path)

    logging.info("Loading universe from %s", config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Universe config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Universe config must be a YAML dictionary.")

    if "universe" not in config:
        raise ValueError("Universe must contain a 'universe' key.")

    universe = config["universe"]

    if not isinstance(universe, dict):
        raise ValueError("'universe' must be a dictionary of asset groups.")

    instruments: list[Instrument] = []
    seen_symbols: set[str] = set()

    for asset_class, tickers in universe.items():
        if not isinstance(asset_class, str):
            raise ValueError(f"Universe group name must be a string. Got: {asset_class!r}")

        asset_class = asset_class.strip().lower()

        if asset_class not in VALID_GROUPS:
            raise ValueError(f"Invalid universe group: {asset_class}")

        if not isinstance(tickers, list):
            raise ValueError(f"Universe group {asset_class} must be a list.")

        for raw_ticker in tickers:
            if not isinstance(raw_ticker, dict):
                raise ValueError(f"Entries under {asset_class} must be dictionaries.")

            instrument = _parse_instrument(raw_ticker, asset_class)

            if instrument.symbol in seen_symbols:
                raise ValueError(f"Duplicate symbol in universe: {instrument.symbol}")

            seen_symbols.add(instrument.symbol)
            instruments.append(instrument)

    return instruments
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from cleaner import clean_ohlcv
from gates import (
    CalendarCompleteGate,
    CoverageGate,
    Gate,
    GateResult,
    OhlcCoherentGate,
    OutliersVsWhiteListGate,
    PlausibilityGate,
    PositivePricesGate,
    SortedKeyGate,
    UniqueKeyGate,
    VolumeNonnegGate,
    run_gates,
)
from sources import SourceError, SourceResolver, build_default_resolver
from store import ParquetStore, assert_unique_slugs, infer_date_range
from universe import Instrument, load_universe

logger = logging.getLogger(__name__)

ReportStatus = Literal["ok", "flagged", "failed", "fetch_error"]


@dataclass(frozen=True, slots=True)
class IngestConfig:
    universe_path: Path
    bronze_dir: Path
    silver_dir: Path
    start_date: date
    end_date: date | None
    calendar: str
    events_whitelist: Path | None
    outlier_abs_log_return: dict[str, float]


@dataclass(frozen=True, slots=True)
class SymbolReport:
    symbol: str
    status: ReportStatus
    source: str | None
    rows: int
    retrieved_start: str | None = None
    retrieved_end: str | None = None
    gate_results: tuple[GateResult, ...] = ()
    silver_written: bool = False
    error: str | None = None


def load_ingest_config(path: str | Path) -> IngestConfig:
    """
    Load data-layer ingest configuration.

    Expected shape:

    data:
      universe: "config/universe.yaml"
      start_date: "2010-01-01"
      end_date:
      calendar: "XNYS"
      bronze_dir: "data/bronze"
      silver_dir: "data/silver"
      events_whitelist: "config/events.yaml"
      gates:
        outlier_abs_log_return:
          equity: 0.15
          volatility: 0.70
    """
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("data.yaml must be a YAML dictionary.")

    data = raw.get("data")
    if not isinstance(data, dict):
        raise ValueError("data.yaml must contain a 'data' dictionary.")

    gates_cfg = data.get("gates", {})
    if not isinstance(gates_cfg, dict):
        raise ValueError("data.gates must be a dictionary.")

    thresholds_raw = gates_cfg.get("outlier_abs_log_return")
    if not isinstance(thresholds_raw, dict):
        raise ValueError("Missing data.gates.outlier_abs_log_return.")

    thresholds: dict[str, float] = {}
    for key, value in thresholds_raw.items():
        if not isinstance(key, str):
            raise ValueError("Outlier threshold keys must be strings.")

        if not isinstance(value, int | float):
            raise ValueError(f"Outlier threshold for {key!r} must be numeric.")

        thresholds[key.strip().lower()] = float(value)

    events_raw = data.get("events_whitelist")
    events_whitelist = Path(events_raw) if isinstance(events_raw, str) and events_raw.strip() else None

    return IngestConfig(
        universe_path=Path(_require_str(data, "universe")),
        bronze_dir=Path(_require_str(data, "bronze_dir")),
        silver_dir=Path(_require_str(data, "silver_dir")),
        start_date=_parse_required_date(data.get("start_date"), "start_date"),
        end_date=_parse_optional_date(data.get("end_date"), "end_date"),
        calendar=_require_str(data, "calendar"),
        events_whitelist=events_whitelist,
        outlier_abs_log_return=thresholds,
    )


def build_default_gates(cfg: IngestConfig) -> list[Gate]:
    """
    Construct the production gate list.

    The order is intentional:
    - cheap structural checks first
    - calendar/coverage checks after basic date sanity
    - statistical/plausibility checks last
    """
    return [
        UniqueKeyGate(),
        SortedKeyGate(),
        PositivePricesGate(),
        OhlcCoherentGate(),
        VolumeNonnegGate(),
        CalendarCompleteGate(calendar_name=cfg.calendar),
        CoverageGate(calendar_name=cfg.calendar),
        OutliersVsWhiteListGate(
            thresholds=cfg.outlier_abs_log_return,
            events_path=str(cfg.events_whitelist) if cfg.events_whitelist is not None else None,
        ),
        PlausibilityGate(),
    ]


def ingest_symbol(
    inst: Instrument,
    cfg: IngestConfig,
    store: ParquetStore,
    resolver: SourceResolver,
    gates: list[Gate],
    *,
    from_bronze: bool = False,
) -> SymbolReport:
    """
    Ingest one symbol.

    Pipeline:
        fetch or load latest bronze
        write bronze immediately if fetched
        clean
        run gates
        write silver only if no gate failed
        return a report either way

    Important property:
        failed gates never overwrite existing silver.
    """
    logger.info("Ingesting %s", inst.symbol)

    requested_start = max(cfg.start_date, inst.expected_start)
    requested_end = cfg.end_date

    if from_bronze:
        raw_frame, bronze_manifest = store.latest_bronze(inst.symbol)
        source = bronze_manifest.source
        logger.info("Using latest bronze for %s from %s", inst.symbol, source)
    else:
        fetch_result = resolver.fetch(
            inst=inst,
            start=requested_start,
            end=requested_end,
        )

        source = fetch_result.source
        raw_frame = fetch_result.df

        bronze_manifest = store.write_bronze(
            raw_frame,
            inst,
            source=source,
            requested_start=requested_start.isoformat(),
            requested_end=requested_end.isoformat() if requested_end is not None else None,
        )

    clean_result = clean_ohlcv(raw_frame, inst)

    if clean_result.issues:
        logger.warning(
            "%s had %d coercion issue(s) during cleaning.",
            inst.symbol,
            len(clean_result.issues),
        )

    gate_results = tuple(
        run_gates(
            clean_result.df,
            instruments=[inst],
            gates=gates,
        )
    )

    retrieved_start, retrieved_end = _safe_date_range(clean_result.df)

    if any(result.status == "failed" for result in gate_results):
        logger.warning(
            "Gate failure for %s. Silver will not be written.",
            inst.symbol,
        )

        return SymbolReport(
            symbol=inst.symbol,
            status="failed",
            source=source,
            rows=len(clean_result.df),
            retrieved_start=retrieved_start,
            retrieved_end=retrieved_end,
            gate_results=gate_results,
            silver_written=False,
        )

    gate_summary = summarize_gate_results(gate_results)

    store.write_silver(
        clean_result.df,
        inst,
        inputs=(bronze_manifest.sha256,),
        gate_summary=gate_summary,
    )

    status: ReportStatus = (
        "flagged"
        if any(result.status == "flagged" for result in gate_results)
        else "ok"
    )

    return SymbolReport(
        symbol=inst.symbol,
        status=status,
        source=source,
        rows=len(clean_result.df),
        retrieved_start=retrieved_start,
        retrieved_end=retrieved_end,
        gate_results=gate_results,
        silver_written=True,
    )


def run_ingest(
    cfg: IngestConfig,
    universe: list[Instrument],
    store: ParquetStore,
    resolver: SourceResolver,
    gates: list[Gate],
    *,
    symbols: list[str] | None = None,
    from_bronze: bool = False,
) -> list[SymbolReport]:
    """
    Run ingest for all selected symbols.

    Isolation doctrine:
        one symbol's failure does not abort the rest.
    """
    selected = _select_symbols(universe, symbols)

    reports: list[SymbolReport] = []

    for inst in selected:
        try:
            report = ingest_symbol(
                inst=inst,
                cfg=cfg,
                store=store,
                resolver=resolver,
                gates=gates,
                from_bronze=from_bronze,
            )
        except SourceError as exc:
            logger.exception("Fetch failed for %s", inst.symbol)

            report = SymbolReport(
                symbol=inst.symbol,
                status="fetch_error",
                source=None,
                rows=0,
                error=str(exc),
            )
        except Exception as exc:
            logger.exception("Ingest failed for %s", inst.symbol)

            report = SymbolReport(
                symbol=inst.symbol,
                status="failed",
                source=None,
                rows=0,
                error=str(exc),
            )

        reports.append(report)

    return reports


def run_ingest_from_config(
    config_path: str | Path,
    *,
    symbols: list[str] | None = None,
    from_bronze: bool = False,
) -> list[SymbolReport]:
    """
    Composition root for the data ingest pipeline.

    This is the one place where real objects are constructed:
        config
        universe
        store
        source resolver
        gates

    CLI should call this function.
    Tests can bypass it and inject fake stores/sources/gates into run_ingest().
    """
    cfg = load_ingest_config(config_path)

    universe = load_universe(str(cfg.universe_path))
    assert_unique_slugs(universe)

    store = ParquetStore(
        bronze_dir=cfg.bronze_dir,
        silver_dir=cfg.silver_dir,
    )

    resolver = build_default_resolver()
    gates = build_default_gates(cfg)

    return run_ingest(
        cfg=cfg,
        universe=universe,
        store=store,
        resolver=resolver,
        gates=gates,
        symbols=symbols,
        from_bronze=from_bronze,
    )


def summarize_gate_results(results: tuple[GateResult, ...] | list[GateResult]) -> str:
    counts = Counter(result.status for result in results)

    parts: list[str] = []

    for status in ("ok", "flagged", "failed"):
        count = counts.get(status, 0)

        if count:
            parts.append(f"{count} {status}")

    return " · ".join(parts) if parts else "no gates"


def ingest_exit_code(reports: list[SymbolReport]) -> int:
    """
    Exit-code convention:
        0 -> all symbols ok or flagged
        1 -> at least one failed or fetch_error
    """
    bad_statuses = {"failed", "fetch_error"}

    return 1 if any(report.status in bad_statuses for report in reports) else 0


def format_reports(reports: list[SymbolReport]) -> str:
    """
    Render a compact terminal report.
    """
    lines = [
        f"{'symbol':<8} {'status':<12} {'source':<12} {'rows':>8} "
        f"{'range':<24} {'gates':<24} {'silver':<8}"
    ]

    for report in reports:
        date_range = _format_range(report.retrieved_start, report.retrieved_end)
        gate_summary = summarize_gate_results(report.gate_results)
        source = report.source or "-"
        silver = "written" if report.silver_written else "-"

        lines.append(
            f"{report.symbol:<8} {report.status:<12} {source:<12} "
            f"{report.rows:>8} {date_range:<24} {gate_summary:<24} {silver:<8}"
        )

        if report.error:
            lines.append(f"  error: {report.error}")

    return "\n".join(lines)


def _select_symbols(
    universe: list[Instrument],
    symbols: list[str] | None,
) -> list[Instrument]:
    if symbols is None:
        return universe

    wanted = {_normalize_symbol(symbol) for symbol in symbols}

    selected = [
        inst
        for inst in universe
        if _normalize_symbol(inst.symbol) in wanted
    ]

    found = {_normalize_symbol(inst.symbol) for inst in selected}
    missing = wanted - found

    if missing:
        raise ValueError(
            f"Requested symbols not found in universe: {sorted(missing)}"
        )

    return selected


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"data.{key} must be a non-empty string.")

    return value.strip()


def _parse_required_date(value: Any, field_name: str) -> date:
    parsed = _parse_optional_date(value, field_name)

    if parsed is None:
        raise ValueError(f"{field_name} is required.")

    return parsed


def _parse_optional_date(value: Any, field_name: str) -> date | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"{field_name} must be YYYY-MM-DD.") from exc

    raise ValueError(f"{field_name} must be a date, YYYY-MM-DD string, or null.")


def _format_range(start: str | None, end: str | None) -> str:
    if start is None or end is None:
        return "-"

    return f"{start}..{end}"


def _safe_date_range(df: pd.DataFrame) -> tuple[str | None, str | None]:
    try:
        return infer_date_range(df)
    except Exception:
        return None, None
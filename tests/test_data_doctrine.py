from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from archer.data.gates import GateResult, OutliersVsWhiteListGate, Gate
from archer.data.ingest import IngestConfig, ingest_symbol
from archer.data.sources import FetchResult, SourceResolver
from archer.data.store import ParquetStore
from archer.data.universe import Instrument


def make_inst(symbol: str = "TEST", exposure_family: str = "equity") -> Instrument:
    return Instrument(
        symbol=symbol,
        name="Test Instrument",
        instrument_type="etf",
        exposure_family=exposure_family,
        role="realized_vol_leg",
        tradable=False,
        source_priority=("fake",),
        expected_start=date(2020, 1, 2),
    )


def make_ohlcv(adj_closes: list[float], symbol: str = "TEST") -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=len(adj_closes))

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": adj_closes,
            "High": [x * 1.01 for x in adj_closes],
            "Low": [x * 0.99 for x in adj_closes],
            "Close": adj_closes,
            "Adj Close": adj_closes,
            "Volume": [100] * len(adj_closes),
        }
    )


def test_outlier_gate_no_outlier_is_ok() -> None:
    inst = make_inst()
    df = make_ohlcv([100.0, 101.0, 102.0])
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    df["symbol"] = inst.symbol

    gate = OutliersVsWhiteListGate(thresholds={"equity": 0.15})

    result = gate.check(inst, df)

    assert result.status == "ok"


def test_outlier_gate_unwhitelisted_outlier_fails() -> None:
    inst = make_inst()
    df = make_ohlcv([100.0, 101.0, 150.0])
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    df["symbol"] = inst.symbol

    gate = OutliersVsWhiteListGate(thresholds={"equity": 0.15})

    result = gate.check(inst, df)

    assert result.status == "failed"
    assert result.bad_rows is not None
    assert len(result.bad_rows) == 1


def test_outlier_gate_whitelisted_outlier_is_flagged(tmp_path: Path) -> None:
    inst = make_inst()
    df = make_ohlcv([100.0, 101.0, 150.0])
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    df["symbol"] = inst.symbol

    events_path = tmp_path / "events.yaml"
    events_path.write_text(
        """
events:
  - date: "2020-01-06"
    symbols: ["TEST"]
""",
        encoding="utf-8",
    )

    gate = OutliersVsWhiteListGate(
        thresholds={"equity": 0.15},
        events_path=str(events_path),
    )

    result = gate.check(inst, df)

    assert result.status == "flagged"
    assert result.bad_rows is not None
    assert len(result.bad_rows) == 1


class FakeSource:
    name = "fake"

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def fetch(
        self,
        inst: Instrument,
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame:
        return self.df.copy()


def make_cfg(tmp_path: Path) -> IngestConfig:
    return IngestConfig(
        universe_path=tmp_path / "universe.yaml",
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        start_date=date(2020, 1, 2),
        end_date=None,
        calendar="XNYS",
        events_whitelist=None,
        outlier_abs_log_return={"equity": 0.15},
    )


def test_ingest_blocks_silver_on_failed_gate(tmp_path: Path) -> None:
    inst = make_inst()
    raw = make_ohlcv([100.0, 101.0, 150.0])

    cfg = make_cfg(tmp_path)
    store = ParquetStore(
        bronze_dir=cfg.bronze_dir,
        silver_dir=cfg.silver_dir,
    )
    resolver = SourceResolver([FakeSource(raw)])

    gates : list[Gate] = [
        OutliersVsWhiteListGate(thresholds={"equity": 0.15}),
    ]

    report = ingest_symbol(
        inst=inst,
        cfg=cfg,
        store=store,
        resolver=resolver,
        gates=gates,
    )

    assert report.status == "failed"
    assert report.silver_written is False
    assert not store.silver_path(inst.symbol).exists()


def test_ingest_allows_silver_on_flagged_whitelisted_outlier(tmp_path: Path) -> None:
    inst = make_inst()
    raw = make_ohlcv([100.0, 101.0, 150.0])

    events_path = tmp_path / "events.yaml"
    events_path.write_text(
        """
events:
  - date: "2020-01-06"
    symbols: ["TEST"]
""",
        encoding="utf-8",
    )

    cfg = make_cfg(tmp_path)
    store = ParquetStore(
        bronze_dir=cfg.bronze_dir,
        silver_dir=cfg.silver_dir,
    )
    resolver = SourceResolver([FakeSource(raw)])

    gates : list[Gate] = [
        OutliersVsWhiteListGate(
            thresholds={"equity": 0.15},
            events_path=str(events_path),
        ),
    ]

    report = ingest_symbol(
        inst=inst,
        cfg=cfg,
        store=store,
        resolver=resolver,
        gates=gates,
    )

    assert report.status == "flagged"
    assert report.silver_written is True
    assert store.silver_path(inst.symbol).exists()
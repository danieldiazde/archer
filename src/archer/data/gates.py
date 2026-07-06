import pandas as pd
from dataclasses import dataclass
from typing import Literal, Protocol
from universe import Instrument
import logging

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
                symbol=inst.symbol,
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
    pass

class PositivePricesGate:
    name = "positive_prices"

    def check(self, inst: Instrument, df: pd.DataFrame) -> GateResult:
        negatives = df[]

class SortedKeyGate:
    pass

class VolumeNonnegGate:
    pass

class CalendarCompleteGate:
    pass

class CoverageGate:
    pass

class OutliersVsWhiteListGate:
    pass

class PlausibilityGate:
    pass

def run_gates(df : pd.DataFrame, instruments : list[Instrument], gates : list[Gate]) -> list[GateResult]:

    logger.info("Running gates on data with dape %s", df.shape)

    results : list[GateResult] = []

    # unique_key
    for inst in instruments:
        symbol_df = df[df["symbol"] == inst.symbol]

        for gate in gates:
            result = gate.check(inst, symbol_df)
            results.append(result)
    
    return results
   

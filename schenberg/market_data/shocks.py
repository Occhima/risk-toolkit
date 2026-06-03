from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource


@dataclass(frozen=True, slots=True)
class ParallelZeroRateShock:
    source_name: str = "di_curve"
    shift: float = 0.0001

    def apply(self, market: MarketSnapshot) -> MarketSnapshot:
        source = market.source(self.source_name)
        bumped = MarketSource(
            name=source.name,
            data=source.data.with_columns(zero_rate=pl.col("zero_rate") + self.shift),
            schema=source.schema,
        )
        return market.with_source(bumped)

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.domain.schemas.forward import ForwardTrade
from schenberg.domain.schemas.position import InstrumentPrice
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.position.functions import price_forward_instruments


@dataclass(frozen=True, slots=True)
class InstrumentCatalog:
    forwards: LazyFrame[ForwardTrade] | None = None

    def prices(self, market: MarketSnapshot) -> LazyFrame[InstrumentPrice]:
        parts: list[pl.LazyFrame] = []

        if self.forwards is not None:
            parts.append(price_forward_instruments(self.forwards, market))

        if not parts:
            raise ValueError("instrument catalog has no instruments")

        return cast(
            LazyFrame[InstrumentPrice],
            pl.concat(parts, how="diagonal_relaxed"),
        )

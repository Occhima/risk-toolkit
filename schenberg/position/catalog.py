from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.domain.schemas.forward import ForwardTrade
from schenberg.domain.schemas.position import InstrumentValue
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.position.functions import mtm_forward


@dataclass(frozen=True, slots=True)
class InstrumentCatalog:
    forwards: LazyFrame[ForwardTrade] | None = None

    def value(self, market: MarketSnapshot) -> LazyFrame[InstrumentValue]:
        parts: list[pl.LazyFrame] = []

        if self.forwards is not None:
            parts.append(mtm_forward(self.forwards, market))

        if not parts:
            raise ValueError("instrument catalog has no instruments")

        return cast(
            LazyFrame[InstrumentValue],
            pl.concat(parts, how="diagonal_relaxed"),
        )

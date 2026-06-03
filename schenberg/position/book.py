from __future__ import annotations

from dataclasses import dataclass

from pandera.typing.polars import LazyFrame

from schenberg.domain.schemas.position import InstrumentPrice, Position, PricedPosition
from schenberg.position.functions import with_prices


@dataclass(frozen=True, slots=True)
class Book:
    positions: LazyFrame[Position]

    def with_prices(
        self,
        prices: LazyFrame[InstrumentPrice],
    ) -> LazyFrame[PricedPosition]:
        return with_prices(self.positions, prices)

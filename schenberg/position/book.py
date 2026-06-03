from __future__ import annotations

from dataclasses import dataclass

from pandera.typing.polars import LazyFrame

from schenberg.domain.schemas.position import InstrumentValue, Position, PositionValue
from schenberg.position.functions import value_positions


@dataclass(frozen=True, slots=True)
class Book:
    positions: LazyFrame[Position]

    def value(
        self,
        instrument_values: LazyFrame[InstrumentValue],
    ) -> LazyFrame[PositionValue]:
        return value_positions(self.positions, instrument_values)

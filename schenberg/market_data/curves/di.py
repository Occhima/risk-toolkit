from __future__ import annotations

from dataclasses import dataclass

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import ColumnLike
from schenberg.core.market import MarketRequirement
from schenberg.domain.schemas.market_data import DiCurveContract
from schenberg.market_data.calendar.conventions import Calendar
from schenberg.market_data.curves import CurveSpec
from schenberg.market_data.sources import MarketSource


@dataclass(frozen=True, slots=True)
class DiCurveSpec:
    name: str = "di_curve"

    def zero_rate(
        self,
        *,
        indexer: ColumnLike = "id_indexador",
        tenor: ColumnLike = "payment_days",
        output: str = "zero_rate",
    ) -> MarketRequirement:
        return CurveSpec(self.name).value("zero_rate", indexer=indexer, tenor=tenor, output=output)


@dataclass(frozen=True, slots=True)
class DiCurve:
    data: LazyFrame[DiCurveContract]
    calendar: Calendar
    name: str = "di_curve"

    @classmethod
    @pa.check_types(lazy=True)
    def build(
        cls,
        data: LazyFrame[DiCurveContract],
        calendar: Calendar,
        *,
        name: str = "di_curve",
    ) -> DiCurve:
        return cls(data=data, calendar=calendar, name=name)

    def source(self) -> MarketSource:
        return MarketSource(name=self.name, data=self.data, schema=DiCurveContract)

    def spec(self) -> DiCurveSpec:
        return DiCurveSpec(name=self.name)

from __future__ import annotations

from dataclasses import dataclass

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import ColumnSet
from schenberg.core.market import MarketRequirement
from schenberg.domain.schemas.market_data import FixingContract
from schenberg.market_data.sources import MarketSource


@dataclass(frozen=True, slots=True)
class FixingsSpec:
    name: str = "fixings"

    def fixing(
        self,
        *,
        indexer_col: str = "id_indexador",
        date_col: str = "base_date",
        output: str = "base_index",
    ) -> MarketRequirement:
        return MarketRequirement(
            table=self.name,
            on=ColumnSet.from_pairs(
                (indexer_col, "id_indexador"),
                (date_col, "fixing_date"),
            ),
            outputs={"fixing_value": output},
        )


@dataclass(frozen=True, slots=True)
class Fixings:
    data: LazyFrame[FixingContract]
    name: str = "fixings"

    @classmethod
    @pa.check_types(lazy=True)
    def build(cls, data: LazyFrame[FixingContract], *, name: str = "fixings") -> Fixings:
        return cls(data=data, name=name)

    def source(self) -> MarketSource:
        return MarketSource(name=self.name, data=self.data, schema=FixingContract)

    def spec(self) -> FixingsSpec:
        return FixingsSpec(name=self.name)

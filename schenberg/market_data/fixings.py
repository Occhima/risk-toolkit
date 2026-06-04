from __future__ import annotations

from dataclasses import dataclass

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import ColumnLike
from schenberg.core.market import MarketRequirement
from schenberg.domain.schemas.market_data import FixingContract
from schenberg.market_data.sources import MarketSource
from schenberg.market_data.specs import JoinSpec


@dataclass(frozen=True, slots=True)
class FixingsSpec:
    name: str = "fixings"

    def fixing(
        self,
        *,
        indexer: ColumnLike = "id_indexador",
        date: ColumnLike = "base_date",
        output: str = "base_index",
    ) -> MarketRequirement:
        return self.value(indexer=indexer, date=date, output=output)

    def value(
        self,
        *,
        indexer: ColumnLike = "id_indexador",
        date: ColumnLike = "fixing_date",
        output: str = "fixing_value",
    ) -> MarketRequirement:
        """Fetch a fixing value by ``(indexer, date)``.

        Use with a pre-computed join-key column, e.g. one derived via
        ``schenberg.market_data.date_rules``. :meth:`fixing` is the same read with
        the ``base_date``/``base_index`` defaults used by inflation legs.
        """
        return JoinSpec(self.name).read(
            "fixing_value",
            (indexer, "id_indexador"),
            (date, "fixing_date"),
            output=output,
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

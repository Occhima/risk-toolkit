from __future__ import annotations

from dataclasses import dataclass

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import ColumnLike
from schenberg.core.market import MarketRequirement
from schenberg.domain.schemas.market_data import FxRatesContract
from schenberg.market_data.sources import MarketSource
from schenberg.market_data.specs import JoinSpec


@dataclass(frozen=True, slots=True)
class FxRatesSpec:
    name: str = "fx_rates"

    def fx_rate(
        self,
        *,
        currency: ColumnLike = "currency",
        output: str = "fx_rate",
    ) -> MarketRequirement:
        return JoinSpec(self.name).read("fx_rate", (currency, "currency"), output=output)


@dataclass(frozen=True, slots=True)
class FxRates:
    data: LazyFrame[FxRatesContract]
    name: str = "fx_rates"

    @classmethod
    @pa.check_types(lazy=True)
    def build(cls, data: LazyFrame[FxRatesContract], *, name: str = "fx_rates") -> FxRates:
        return cls(data=data, name=name)

    def source(self) -> MarketSource:
        return MarketSource(name=self.name, data=self.data, schema=FxRatesContract)

    def spec(self) -> FxRatesSpec:
        return FxRatesSpec(name=self.name)

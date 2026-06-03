from __future__ import annotations

from dataclasses import dataclass

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.core.market import MarketRequirement
from schenberg.domain.schemas.market_data import FxRatesContract
from schenberg.market_data.requirements import require
from schenberg.market_data.sources import MarketSource


@dataclass(frozen=True, slots=True)
class FxRatesSpec:
    name: str = "fx_rates"

    def fx_rate(
        self,
        *,
        currency_col: str = "currency",
        output: str = "fx_rate",
        optional: bool = False,
    ) -> MarketRequirement:
        return require(
            self.name,
            (currency_col, "currency"),
            outputs={"fx_rate": output},
            optional=optional,
        )


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

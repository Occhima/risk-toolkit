from __future__ import annotations

from dataclasses import dataclass

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.core.market import MarketRequirement
from schenberg.domain.schemas.market_data import EnergyForwardCurveContract
from schenberg.market_data.requirements import require
from schenberg.market_data.sources import MarketSource


@dataclass(frozen=True, slots=True)
class EnergyForwardCurveSpec:
    name: str = "energy_forward_curve"

    def forward_price(
        self,
        *,
        submarket_col: str = "submarket",
        period_col: str = "delivery_period",
        price_output: str = "forward_price",
    ) -> MarketRequirement:
        return require(
            self.name,
            (submarket_col, "submarket"),
            (period_col, "delivery_period"),
            outputs={"forward_price": price_output},
        )


@dataclass(frozen=True, slots=True)
class EnergyForwardCurve:
    data: LazyFrame[EnergyForwardCurveContract]
    name: str = "energy_forward_curve"

    @classmethod
    @pa.check_types(lazy=True)
    def build(
        cls,
        data: LazyFrame[EnergyForwardCurveContract],
        *,
        name: str = "energy_forward_curve",
    ) -> EnergyForwardCurve:
        return cls(data=data, name=name)

    def source(self) -> MarketSource:
        return MarketSource(name=self.name, data=self.data, schema=EnergyForwardCurveContract)

    def spec(self) -> EnergyForwardCurveSpec:
        return EnergyForwardCurveSpec(name=self.name)

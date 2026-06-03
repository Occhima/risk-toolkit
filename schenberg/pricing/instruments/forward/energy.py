"""Energy forward pricing for already-normalized delivery-period rows."""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.core.graph import ExprGraph
from schenberg.domain.enums import ForwardFamily, InstrumentType, SettlementType
from schenberg.domain.schemas.forward import EnergyForwardLeg, ForwardPricing, ForwardTrade
from schenberg.domain.schemas.position import InstrumentPrice
from schenberg.market_data.curves.di import DiCurveSpec
from schenberg.market_data.forwards import EnergyForwardCurveSpec
from schenberg.market_data.fx import FxRatesSpec
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.forward.generic import forward_valuation_graph
from schenberg.pricing.instruments.forward.router import forward_router

F = cols(ForwardTrade)
E = cols(EnergyForwardLeg)
P = cols(ForwardPricing)
PX = cols(InstrumentPrice)

DI = DiCurveSpec("di_curve")
ENERGY = EnergyForwardCurveSpec("energy_forward_curve")
FX = FxRatesSpec("fx_rates")


@forward_router.register(
    F.instrument_type == InstrumentType.FORWARD.value,
    F.forward_family == ForwardFamily.ENERGY.value,
    F.settlement_type == SettlementType.PHYSICAL.value,
)
def energy_forward_graph() -> ExprGraph:
    return (
        ExprGraph.compose(
            "energy_forward",
            forward_valuation_graph,
        )
        .with_market(
            ENERGY.forward_price(),
            DI.zero_rate(),
            FX.fx_rate(),
        )
        .with_outputs("pricing", ForwardPricing)
    )


@pa.check_types(lazy=True)
def price_energy_forward(
    legs: LazyFrame[EnergyForwardLeg],
    market: MarketSnapshot,
) -> LazyFrame[InstrumentPrice]:
    priced = energy_forward_graph.compute_for(
        legs,
        market=market,
        output_profile="pricing",
    )

    result = (
        priced.group_by(E.instrument_id.name)
        .agg(price=P.value.expr().sum())
        .with_columns(instrument_type=pl.lit(InstrumentType.FORWARD.value))
        .select(
            PX.instrument_type.name,
            PX.instrument_id.name,
            PX.price.name,
        )
    )

    return cast(LazyFrame[InstrumentPrice], result)

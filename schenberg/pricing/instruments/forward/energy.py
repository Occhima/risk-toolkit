"""Energy forward pricing.

Energy forwards settle on the 6th business day of the ANBIMA calendar after the
last calendar day of the delivery month, so the discounting tenor is *derived*
from the delivery period and the snapshot's ``as_of`` rather than supplied as a
raw ``payment_days`` column. :class:`EnergyForwardPricer` materializes the
settlement date and that business-day count before handing the rows to the
otherwise-generic forward valuation graph; it is registered on the forward
router so the dedicated and generic entry points price energy identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.core.graph import ExprGraph
from schenberg.domain.enums import ForwardFamily, InstrumentType, SettlementType
from schenberg.domain.schemas.forward import EnergyForwardLeg, ForwardPricing, ForwardTrade
from schenberg.domain.schemas.position import InstrumentPrice
from schenberg.market_data.calendar.anbima import ANBIMA_HOLIDAYS
from schenberg.market_data.curves.di import DiCurveSpec
from schenberg.market_data.date_rules import business_day_count_expr, energy_settlement_date
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

SETTLEMENT_DATE = "settlement_date"

# The energy forward graph is the plain generic valuation graph with energy
# market bindings; payment_days is fed in as a derived column (see the pricer).
energy_forward_graph = (
    ExprGraph.compose("energy_forward", forward_valuation_graph)
    .with_market(
        ENERGY.forward_price(),
        DI.zero_rate(),
        FX.fx_rate(),
    )
    .with_outputs("pricing", ForwardPricing)
)


@dataclass(frozen=True, slots=True)
class EnergyForwardPricer:
    """Derive the ANBIMA fixing tenor, then delegate to ``graph``.

    ``payment_days`` is the BUS/252 business-day count from ``as_of`` to the
    settlement date (6th ANBIMA business day after the delivery month), so it is
    both the discount horizon and the DI-curve join key — materialized here,
    before the graph attaches market data.
    """

    graph: ExprGraph

    def compute_for(
        self,
        lf: pl.LazyFrame,
        *,
        market: MarketSnapshot | None = None,
        output_profile: str = "pricing",
    ) -> pl.LazyFrame:
        if market is None:
            raise ValueError("energy forward pricing needs a MarketSnapshot")
        prepared = lf.with_columns(
            energy_settlement_date(
                period_col=E.delivery_period.name,
                output_col=SETTLEMENT_DATE,
                holidays=ANBIMA_HOLIDAYS,
            )
        ).with_columns(
            business_day_count_expr(
                pl.lit(market.as_of),
                pl.col(SETTLEMENT_DATE),
                holidays=ANBIMA_HOLIDAYS,
            ).alias(F.payment_days.name)
        )
        return self.graph.compute_for(prepared, market=market, output_profile=output_profile)


energy_forward_pricer = EnergyForwardPricer(energy_forward_graph)

forward_router.register(
    F.instrument_type == InstrumentType.FORWARD.value,
    F.forward_family == ForwardFamily.ENERGY.value,
    F.settlement_type == SettlementType.PHYSICAL.value,
)(lambda: energy_forward_pricer)


@pa.check_types(lazy=True)
def price_energy_forward(
    legs: LazyFrame[EnergyForwardLeg],
    market: MarketSnapshot,
) -> LazyFrame[InstrumentPrice]:
    priced = energy_forward_pricer.compute_for(
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

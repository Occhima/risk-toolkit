"""Energy forward pricing for already-normalized delivery-period rows.

The energy forward reuses the generic forward backbone and only adds *where the
price comes from*: it looks up ``forward_price`` on the energy curve by
``(submarket, delivery_period)``, discounts on the DI curve and converts via FX.
The math graph never mentions "energy".
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.core.graph import PricingGraph, Term
from schenberg.domain.enums import ForwardFamily, InstrumentType, SettlementType
from schenberg.domain.schemas.forward import EnergyForwardLeg, ForwardPricing, ForwardTrade
from schenberg.domain.schemas.position import InstrumentPrice
from schenberg.market_data.calendar.anbima import ANBIMA_HOLIDAYS
from schenberg.market_data.date_rules import nth_business_day_of_following_month
from schenberg.market_data.requirements import MarketRequirements, requires
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.forward.generic import assemble_forward, forward_payoff_term
from schenberg.pricing.instruments.forward.prices import aggregate_forward_prices
from schenberg.pricing.instruments.forward.router import forward_router
from schenberg.pricing.market import DI, ENERGY_FWD, FX

F = cols(ForwardTrade)
E = cols(EnergyForwardLeg)

# Brazilian energy forwards fix on the 6th ANBIMA business day of the month after
# the delivery month.
_FIXING_BUSINESS_DAY = 6


class EnergyForwardRequirements(MarketRequirements[EnergyForwardLeg]):
    forward_price: Term[float] = requires(ENERGY_FWD.price())
    zero_rate: Term[float] = requires(DI.zero_rate())
    fx_rate: Term[float] = requires(FX.rate())


def with_fixing_date(legs: pl.LazyFrame) -> pl.LazyFrame:
    """Attach each row's settlement/fixing date: the 6th ANBIMA business day of the
    month following the contract's delivery month. A pure normalization step."""
    fixing_date = nth_business_day_of_following_month(
        E.delivery_period.expr(),
        n=_FIXING_BUSINESS_DAY,
        holidays=ANBIMA_HOLIDAYS,
    ).alias(E.fixing_date.name)
    return legs.with_columns(fixing_date)


@forward_router.when(
    F.instrument_type == InstrumentType.FORWARD.value,
    F.forward_family == ForwardFamily.ENERGY.value,
    F.settlement_type == SettlementType.PHYSICAL.value,
)
def energy_forward_graph() -> PricingGraph:
    g = PricingGraph[EnergyForwardLeg, EnergyForwardRequirements, ForwardPricing]("energy_forward")
    c, m = g.contract, g.market
    future_value = forward_payoff_term(g, forward_price=m.forward_price, strike=c.strike)
    return assemble_forward(
        g,
        future_value=future_value,
        zero_rate=m.zero_rate,
        payment_days=c.payment_days,
        fx_rate=m.fx_rate,
    )


@pa.check_types(lazy=True)
def price_energy_forward(
    legs: LazyFrame[EnergyForwardLeg],
    market: MarketSnapshot,
) -> LazyFrame[InstrumentPrice]:
    priced = energy_forward_graph.compute(legs, market=market, view="output")
    return aggregate_forward_prices(priced)

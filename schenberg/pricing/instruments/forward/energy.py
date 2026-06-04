"""Energy forward pricing for already-normalized delivery-period rows."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.core.graph import FormulaGraph
from schenberg.domain.enums import ForwardFamily, InstrumentType, SettlementType
from schenberg.domain.schemas.forward import EnergyForwardLeg, ForwardTrade
from schenberg.domain.schemas.position import InstrumentPrice
from schenberg.market_data.calendar.anbima import ANBIMA_HOLIDAYS
from schenberg.market_data.curves.di import DiCurveSpec
from schenberg.market_data.date_rules import nth_business_day_of_following_month
from schenberg.market_data.forwards import EnergyForwardCurveSpec
from schenberg.market_data.fx import FxRatesSpec
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.forward.generic import assemble_forward, forward_payoff_term
from schenberg.pricing.instruments.forward.prices import aggregate_forward_prices
from schenberg.pricing.instruments.forward.router import forward_router

F = cols(ForwardTrade)
E = cols(EnergyForwardLeg)

DI = DiCurveSpec("di_curve")
ENERGY = EnergyForwardCurveSpec("energy_forward_curve")
FX = FxRatesSpec("fx_rates")

# Brazilian energy forwards fix on the 6th ANBIMA business day of the month after
# the delivery month.
_FIXING_BUSINESS_DAY = 6


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
def energy_forward_graph() -> FormulaGraph:
    g = FormulaGraph("energy_forward", input=EnergyForwardLeg)
    t = g.input
    # The energy curve writes two columns at once (forward_price + settle days as
    # payment_days), so it keeps its own output names; the curve-supplied
    # payment_days then drives discounting.
    m = g.market(
        energy=ENERGY.forward_price(submarket=t.submarket, period=t.delivery_period),
        zero_rate=DI.zero_rate(indexer=t.id_indexador),
        fx_rate=FX.fx_rate(currency=t.currency),
    )
    future_value = forward_payoff_term(g, forward_price=m.forward_price, strike=t.strike)
    return assemble_forward(
        g,
        future_value=future_value,
        zero_rate=m.zero_rate,
        payment_days=m.payment_days,
        fx_rate=m.fx_rate,
    )


@pa.check_types(lazy=True)
def price_energy_forward(
    legs: LazyFrame[EnergyForwardLeg],
    market: MarketSnapshot,
) -> LazyFrame[InstrumentPrice]:
    priced = energy_forward_graph.compute(legs, market=market, view="pricing")
    return aggregate_forward_prices(priced, id_col=E.instrument_id.name)

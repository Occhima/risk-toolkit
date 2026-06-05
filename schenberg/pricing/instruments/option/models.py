"""Option model graphs and contract-oriented routers.

Each ``(model, kind)`` is one self-contained :class:`PricingGraph` over
``OptionTrade``: its market is declared by a
:class:`~schenberg.market_data.requirements.MarketRequirements` schema, it wires
the generalized-BSM price terms and the closed-form Greek terms, and publishes
three typed views -- ``output`` (price + Greeks), ``price`` and ``state``. The two
routers are ArrowChoice case splits over those graphs: every branch satisfies the
same view contract.

The models differ only in how the cost of carry ``b`` is formed: GENERALIZED
reads it from a curve; MERTON derives ``b = r - q`` from a dividend yield.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.columns import cols
from schenberg.core.graph import PricingGraph, Term, uses
from schenberg.core.router import Router
from schenberg.domain.enums import OptionKind, OptionModel
from schenberg.domain.schemas.option import (
    OptionPrice,
    OptionPricedState,
    OptionPriceWithGreeks,
    OptionTrade,
)
from schenberg.market_data.requirements import MarketRequirements, requires
from schenberg.pricing.instruments.option import core
from schenberg.pricing.market import CARRY, CURVES, DIVIDENDS, VOL
from schenberg.risk.greeks.graph import bsm_greeks_terms

OPT = cols(OptionTrade)


class GeneralizedOptionRequirements(MarketRequirements[OptionTrade]):
    """Rate, an explicit cost-of-carry curve, and the implied-vol surface."""

    rate: Term[float] = requires(CURVES.zero_rate())
    cost_of_carry: Term[float] = requires(CARRY.cost_of_carry())
    vol: Term[float] = requires(VOL.implied_vol())


class MertonOptionRequirements(MarketRequirements[OptionTrade]):
    """Rate, a dividend yield (carry is derived ``b = r - q``), and the vol surface."""

    rate: Term[float] = requires(CURVES.zero_rate())
    div_yield: Term[float] = requires(DIVIDENDS.div_yield())
    vol: Term[float] = requires(VOL.implied_vol())


def _build(model: OptionModel, kind: OptionKind) -> PricingGraph:
    """Build the price+risk graph for one ``(model, kind)`` with its three views."""
    name = f"{model.value.lower()}_{kind.value.lower()}"

    if model is OptionModel.MERTON:
        g = PricingGraph[OptionTrade, MertonOptionRequirements, OptionPriceWithGreeks](name)
        c, m = g.contract, g.market

        @g.formula(
            tags=("merton",),
            symbol="b",
            latex=r"r - q",
            description="Merton cost of carry b = r - q.",
        )
        def cost_of_carry(r: pl.Expr = uses(m.rate), q: pl.Expr = uses(m.div_yield)) -> pl.Expr:
            return r - q

        carry = cost_of_carry
    else:
        g = PricingGraph[OptionTrade, GeneralizedOptionRequirements, OptionPriceWithGreeks](name)
        c, m = g.contract, g.market
        carry = m.cost_of_carry

    rate, vol = m.rate, m.vol
    year_fraction = core.year_fraction_term(g, payment_days=c.payment_days)
    p = core.bsm_terms(
        g,
        kind=kind,
        spot=c.spot,
        strike=c.strike,
        cost_of_carry=carry,
        rate=rate,
        vol=vol,
        year_fraction=year_fraction,
    )
    bsm_greeks_terms(
        g,
        option_kind=c.option_kind,
        spot=c.spot,
        strike=c.strike,
        rate=rate,
        cost_of_carry=carry,
        vol=vol,
        year_fraction=year_fraction,
        d1=p.d1,
        d2=p.d2,
        carry_spot=p.carry_spot,
        disc_strike=p.disc_strike,
    )

    g.returns(OptionPriceWithGreeks)  # primary "output": price + Greeks
    g.view("price", OptionPrice)
    g.view("state", OptionPricedState)
    return g


# One graph per (model, kind); both routers choose among the same graphs.
_GRAPHS = {
    (model, kind): _build(model, kind)
    for model in (OptionModel.GENERALIZED, OptionModel.MERTON)
    for kind in (OptionKind.CALL, OptionKind.PUT)
}

option_price_router = (
    Router.on(OPT.option_model, OPT.option_kind).returns("price", OptionPrice).exclusive()
)
option_risk_router = (
    Router.on(OPT.option_model, OPT.option_kind)
    .returns("output", OptionPriceWithGreeks)
    .exclusive()
)

for (model, kind), graph in _GRAPHS.items():
    option_price_router.case(model, kind)(lambda bound=graph: bound)
    option_risk_router.case(model, kind)(lambda bound=graph: bound)

"""Option model graphs and contract-oriented routers.

Each ``(model, kind)`` is one self-contained :class:`FormulaGraph` over
``OptionTrade``: it names the market it reads (``g.market``), wires the
generalized-BSM price terms and the closed-form Greek terms, and publishes three
views — ``price``, ``state`` and ``risk``. The two routers are ArrowChoice case
splits over those graphs: every branch satisfies the same view contract.

The models differ only in how the cost of carry ``b`` is formed: GENERALIZED
joins it from a curve; MERTON derives ``b = r - q`` from a dividend yield.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.columns import cols
from schenberg.core.graph import FormulaGraph, uses
from schenberg.core.router import Router
from schenberg.domain.enums import OptionKind, OptionModel
from schenberg.domain.schemas.option import (
    OptionPrice,
    OptionPricedState,
    OptionPriceWithGreeks,
    OptionTrade,
)
from schenberg.market_data.curves import CurveSpec
from schenberg.market_data.volatility import VolSurfaceSpec
from schenberg.pricing.instruments.option import core
from schenberg.risk.greeks.graph import bsm_greeks_terms

OPT = cols(OptionTrade)

CURVES = CurveSpec("curves")
CARRY = CurveSpec("carry_curve")
DIVS = CurveSpec("dividend_curve")
VOL = VolSurfaceSpec("vol_surface")


def _build(model: OptionModel, kind: OptionKind) -> FormulaGraph:
    """Build the price+risk graph for one ``(model, kind)`` with its three views."""
    name = f"{model.value.lower()}_{kind.value.lower()}"
    g = FormulaGraph(name, input=OptionTrade)
    t = g.input

    if model is OptionModel.MERTON:
        m = g.market(
            rate=CURVES.value("zero_rate", indexer=t.id_indexador, tenor=t.payment_days),
            div_yield=DIVS.value("div_yield", indexer=t.id_indexador, tenor=t.payment_days),
            vol=VOL.implied_vol(indexer=t.id_indexador, tenor=t.payment_days, strike=t.strike),
        )
        rate, vol = m.rate, m.vol

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
        m = g.market(
            rate=CURVES.value("zero_rate", indexer=t.id_indexador, tenor=t.payment_days),
            cost_of_carry=CARRY.value(
                "cost_of_carry", indexer=t.id_indexador, tenor=t.payment_days
            ),
            vol=VOL.implied_vol(indexer=t.id_indexador, tenor=t.payment_days, strike=t.strike),
        )
        rate, vol, carry = m.rate, m.vol, m.cost_of_carry

    year_fraction = core.year_fraction_term(g, payment_days=t.payment_days)
    p = core.bsm_terms(
        g,
        kind=kind,
        spot=t.spot,
        strike=t.strike,
        cost_of_carry=carry,
        rate=rate,
        vol=vol,
        year_fraction=year_fraction,
    )
    greeks = bsm_greeks_terms(
        g,
        option_kind=t.option_kind,
        spot=t.spot,
        strike=t.strike,
        rate=rate,
        cost_of_carry=carry,
        vol=vol,
        year_fraction=year_fraction,
        d1=p.d1,
        d2=p.d2,
        carry_spot=p.carry_spot,
        disc_strike=p.disc_strike,
    )

    g.returns(
        "price",
        OptionPrice,
        option_id=t.option_id,
        instrument_type=t.instrument_type,
        price=p.price,
    )
    g.returns(
        "state",
        OptionPricedState,
        option_id=t.option_id,
        instrument_type=t.instrument_type,
        option_model=t.option_model,
        option_kind=t.option_kind,
        id_indexador=t.id_indexador,
        spot=t.spot,
        strike=t.strike,
        payment_days=t.payment_days,
        rate=rate,
        cost_of_carry=carry,
        vol=vol,
        year_fraction=year_fraction,
        d1=p.d1,
        d2=p.d2,
        price=p.price,
    )
    g.returns(
        "risk",
        OptionPriceWithGreeks,
        option_id=t.option_id,
        instrument_type=t.instrument_type,
        price=p.price,
        delta=greeks.delta,
        gamma=greeks.gamma,
        vega=greeks.vega,
        theta=greeks.theta,
        rho=greeks.rho,
    )
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
    Router.on(OPT.option_model, OPT.option_kind).returns("risk", OptionPriceWithGreeks).exclusive()
)

for (model, kind), graph in _GRAPHS.items():
    option_price_router.case(model, kind)(lambda bound=graph: bound)
    option_risk_router.case(model, kind)(lambda bound=graph: bound)

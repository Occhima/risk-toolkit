"""Option model graphs and routers.

Volatility, rates, carry and dividends are declared as graph market data via
``for_market``. Price-only leaves compose just the pricing core and expose the
``price`` and ``state`` views; closed-form risk leaves compose the Greek graph on
top and expose the ``risk`` view, routed separately and used only when Greeks are
requested.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.columns import cols
from schenberg.core.graph import FormulaGraph
from schenberg.core.market import MarketRead
from schenberg.core.router import Router
from schenberg.domain.enums import OptionKind, OptionModel
from schenberg.domain.schemas.option import OptionPricedState, OptionTrade
from schenberg.market_data.curves import CurveSpec
from schenberg.market_data.volatility import VolSurfaceSpec
from schenberg.pricing.instruments.option.core import generalized_bsm_core
from schenberg.risk.greeks import bsm_greeks_graph

OPT = cols(OptionTrade)
STATE = cols(OptionPricedState)

CURVES = CurveSpec("curves")
CARRY = CurveSpec("carry_curve")
DIVS = CurveSpec("dividend_curve")
VOL = VolSurfaceSpec("vol_surface")


def _price_leaf(name: str, *graphs: FormulaGraph, price_node: str) -> FormulaGraph:
    """Compose the BSM core with any extra graphs and declare price/state views."""
    return (
        generalized_bsm_core.compose_with(*graphs, name=name)
        .returns("price", price=price_node)
        .returns(
            "state",
            year_fraction=STATE.year_fraction,
            d1=STATE.d1,
            d2=STATE.d2,
            price=price_node,
            rate=STATE.rate,
            cost_of_carry=STATE.cost_of_carry,
            vol=STATE.vol,
        )
    )


def _risk_leaf(name: str, price_graph: FormulaGraph, *, price_node: str) -> FormulaGraph:
    """Compose closed-form Greeks onto a priced leaf and declare the risk view."""
    return price_graph.compose_with(bsm_greeks_graph, name=name).returns(
        "risk",
        price=price_node,
        delta="delta",
        gamma="gamma",
        vega="vega",
        theta="theta",
        rho="rho",
    )


# Market reads are immutable, so one declaration each is shared across every
# leaf; ``for_market`` binds a fresh dependency from them per graph.
RATE = CURVES.value("zero_rate", indexer=OPT.id_indexador, tenor=OPT.payment_days)
COST_OF_CARRY = CARRY.value("cost_of_carry", indexer=OPT.id_indexador, tenor=OPT.payment_days)
DIV_YIELD = DIVS.value("div_yield", indexer=OPT.id_indexador, tenor=OPT.payment_days)
IMPLIED_VOL = VOL.implied_vol(indexer=OPT.id_indexador, tenor=OPT.payment_days, strike=OPT.strike)

_PRICE_NODE = {OptionKind.CALL: "call_price", OptionKind.PUT: "put_price"}


# MERTON derives the cost of carry as b = r - q rather than joining it.
merton_carry = FormulaGraph("merton_carry")


@merton_carry.formula(
    tags=("merton",),
    symbol="b",
    latex=r"r - q",
    description="Merton cost of carry b = r - q.",
)
def cost_of_carry(rate: pl.Expr, div_yield: pl.Expr) -> pl.Expr:
    return rate - div_yield


option_price_router = Router.on(OPT.option_model, OPT.option_kind)
option_risk_router = Router.on(OPT.option_model, OPT.option_kind)


def _model(
    model: OptionModel,
    kind: OptionKind,
    *carry_graphs: FormulaGraph,
    **carry_market: MarketRead,
) -> None:
    """Declare one (model, kind): build its price and risk leaves and route both.

    Every leaf reads ``rate`` and ``vol``; ``carry_graphs`` and ``carry_market``
    supply the model-specific cost of carry (GENERALIZED joins it from a curve,
    MERTON composes ``merton_carry`` and reads a dividend yield instead).
    """
    name = f"{model.value.lower()}_{kind.value.lower()}"
    node = _PRICE_NODE[kind]
    price = _price_leaf(name, *carry_graphs, price_node=node).for_market(
        rate=RATE, vol=IMPLIED_VOL, **carry_market
    )
    risk = _risk_leaf(f"{name}_risk", price, price_node=node)
    option_price_router.case(model.value, kind.value)(lambda: price)
    option_risk_router.case(model.value, kind.value)(lambda: risk)


_model(OptionModel.GENERALIZED, OptionKind.CALL, cost_of_carry=COST_OF_CARRY)
_model(OptionModel.GENERALIZED, OptionKind.PUT, cost_of_carry=COST_OF_CARRY)
_model(OptionModel.MERTON, OptionKind.CALL, merton_carry, div_yield=DIV_YIELD)
_model(OptionModel.MERTON, OptionKind.PUT, merton_carry, div_yield=DIV_YIELD)

option_router = option_price_router

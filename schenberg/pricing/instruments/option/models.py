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


# --- GENERALIZED: cost of carry is a joined market column --------------------
generalized_call = _price_leaf("generalized_call", price_node="call_price").for_market(
    rate=CURVES.value("zero_rate", indexer=OPT.id_indexador, tenor=OPT.payment_days),
    cost_of_carry=CARRY.value("cost_of_carry", indexer=OPT.id_indexador, tenor=OPT.payment_days),
    vol=VOL.implied_vol(indexer=OPT.id_indexador, tenor=OPT.payment_days, strike=OPT.strike),
)
generalized_put = _price_leaf("generalized_put", price_node="put_price").for_market(
    rate=CURVES.value("zero_rate", indexer=OPT.id_indexador, tenor=OPT.payment_days),
    cost_of_carry=CARRY.value("cost_of_carry", indexer=OPT.id_indexador, tenor=OPT.payment_days),
    vol=VOL.implied_vol(indexer=OPT.id_indexador, tenor=OPT.payment_days, strike=OPT.strike),
)


# --- MERTON: cost of carry derived as b = r - q ------------------------------
merton_carry = FormulaGraph("merton_carry")


@merton_carry.formula(
    tags=("merton",),
    symbol="b",
    latex=r"r - q",
    description="Merton cost of carry b = r - q.",
)
def cost_of_carry(rate: pl.Expr, div_yield: pl.Expr) -> pl.Expr:
    return rate - div_yield


merton_call = _price_leaf("merton_call", merton_carry, price_node="call_price").for_market(
    rate=CURVES.value("zero_rate", indexer=OPT.id_indexador, tenor=OPT.payment_days),
    div_yield=DIVS.value("div_yield", indexer=OPT.id_indexador, tenor=OPT.payment_days),
    vol=VOL.implied_vol(indexer=OPT.id_indexador, tenor=OPT.payment_days, strike=OPT.strike),
)
merton_put = _price_leaf("merton_put", merton_carry, price_node="put_price").for_market(
    rate=CURVES.value("zero_rate", indexer=OPT.id_indexador, tenor=OPT.payment_days),
    div_yield=DIVS.value("div_yield", indexer=OPT.id_indexador, tenor=OPT.payment_days),
    vol=VOL.implied_vol(indexer=OPT.id_indexador, tenor=OPT.payment_days, strike=OPT.strike),
)

generalized_call_risk = _risk_leaf(
    "generalized_call_risk", generalized_call, price_node="call_price"
)
generalized_put_risk = _risk_leaf("generalized_put_risk", generalized_put, price_node="put_price")
merton_call_risk = _risk_leaf("merton_call_risk", merton_call, price_node="call_price")
merton_put_risk = _risk_leaf("merton_put_risk", merton_put, price_node="put_price")


# Routers stay decorator-based: ``case`` builds equality predicates on the route
# columns, while complex forks can use ``when``.
option_price_router = Router.on(OPT.option_model, OPT.option_kind)
option_risk_router = Router.on(OPT.option_model, OPT.option_kind)


@option_price_router.case(OptionModel.GENERALIZED.value, OptionKind.CALL.value)
def _generalized_call_price() -> FormulaGraph:
    return generalized_call


@option_price_router.case(OptionModel.GENERALIZED.value, OptionKind.PUT.value)
def _generalized_put_price() -> FormulaGraph:
    return generalized_put


@option_price_router.case(OptionModel.MERTON.value, OptionKind.CALL.value)
def _merton_call_price() -> FormulaGraph:
    return merton_call


@option_price_router.case(OptionModel.MERTON.value, OptionKind.PUT.value)
def _merton_put_price() -> FormulaGraph:
    return merton_put


@option_risk_router.case(OptionModel.GENERALIZED.value, OptionKind.CALL.value)
def _generalized_call_risk() -> FormulaGraph:
    return generalized_call_risk


@option_risk_router.case(OptionModel.GENERALIZED.value, OptionKind.PUT.value)
def _generalized_put_risk() -> FormulaGraph:
    return generalized_put_risk


@option_risk_router.case(OptionModel.MERTON.value, OptionKind.CALL.value)
def _merton_call_risk() -> FormulaGraph:
    return merton_call_risk


@option_risk_router.case(OptionModel.MERTON.value, OptionKind.PUT.value)
def _merton_put_risk() -> FormulaGraph:
    return merton_put_risk


option_router = option_price_router

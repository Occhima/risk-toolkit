"""Option model graphs and routers.

Volatility is declared as graph market data. Price-only leaves compose only the
pricing core; closed-form Greek leaves compose the Greek graph on top and are
registered in a separate router used only when Greeks are requested.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.columns import cols
from schenberg.core.graph import ExprGraph
from schenberg.core.router import Router
from schenberg.domain.enums import OptionKind, OptionModel
from schenberg.domain.schemas.option import OptionGreeks, OptionPricedState, OptionTrade
from schenberg.market_data.requirements import require
from schenberg.market_data.volatility import VolSurfaceSpec
from schenberg.pricing.instruments.option.core import generalized_bsm_core
from schenberg.risk.greeks import bsm_greeks_graph

OPT = cols(OptionTrade)
STATE = cols(OptionPricedState)
VOL = VolSurfaceSpec("vol_surface")
_GREEK_OUTPUTS = {name: name for name in OptionGreeks.to_schema().columns}

# --- market requirements -----------------------------------------------------
_rate = require(
    "curves",
    (OPT.id_indexador.name, "id_indexador"),
    (OPT.payment_days.name, "tenor_days"),
    outputs={"zero_rate": STATE.rate.name},
)
_carry = require(
    "carry_curve",
    (OPT.id_indexador.name, "id_indexador"),
    (OPT.payment_days.name, "tenor_days"),
    outputs={"cost_of_carry": STATE.cost_of_carry.name},
)
_dividend = require(
    "dividend_curve",
    (OPT.id_indexador.name, "id_indexador"),
    (OPT.payment_days.name, "tenor_days"),
    outputs={"div_yield": "div_yield"},
)
_vol = VOL.implied_vol(
    indexer_col=OPT.id_indexador.name,
    tenor_col=OPT.payment_days.name,
    strike_col=OPT.strike.name,
    output=STATE.vol.name,
)


def _price_leaf(name: str, *graphs: ExprGraph, price_node: str) -> ExprGraph:
    leaf = ExprGraph.compose(name, generalized_bsm_core, *graphs)
    leaf.with_outputs("price", price=price_node)
    leaf.with_outputs(
        "pricing",  # backward-compatible profile name
        d1="d1",
        d2="d2",
        price=price_node,
        cost_of_carry=STATE.cost_of_carry.name,
    )
    leaf.with_outputs(
        "priced_state",
        year_fraction=STATE.year_fraction.name,
        d1=STATE.d1.name,
        d2=STATE.d2.name,
        price=price_node,
        rate=STATE.rate.name,
        cost_of_carry=STATE.cost_of_carry.name,
        vol=STATE.vol.name,
    )
    return leaf


def _greeks_leaf(name: str, price_graph: ExprGraph, *, price_node: str) -> ExprGraph:
    leaf = ExprGraph.compose(name, price_graph, bsm_greeks_graph)
    pricing = {
        "year_fraction": STATE.year_fraction.name,
        "d1": STATE.d1.name,
        "d2": STATE.d2.name,
        "price": price_node,
        "rate": STATE.rate.name,
        "cost_of_carry": STATE.cost_of_carry.name,
        "vol": STATE.vol.name,
    }
    leaf.with_outputs("price_with_greeks", **pricing, **_GREEK_OUTPUTS)
    leaf.with_outputs("greeks", **pricing, **_GREEK_OUTPUTS)  # compatibility
    return leaf


# --- GENERALIZED: cost of carry is a joined market column --------------------
generalized_call = _price_leaf("generalized_call", price_node="call_price").with_market(
    _rate, _carry, _vol
)
generalized_put = _price_leaf("generalized_put", price_node="put_price").with_market(
    _rate, _carry, _vol
)


# --- MERTON: cost of carry derived as b = r - q ------------------------------
merton_carry = ExprGraph("merton_carry")


@merton_carry.node(
    tags=("merton",),
    symbol="b",
    formula=r"r - q",
    description="Merton cost of carry b = r - q.",
)
def cost_of_carry(rate: pl.Expr, div_yield: pl.Expr) -> pl.Expr:
    return rate - div_yield


merton_call = _price_leaf("merton_call", merton_carry, price_node="call_price").with_market(
    _rate, _dividend, _vol
)
merton_put = _price_leaf("merton_put", merton_carry, price_node="put_price").with_market(
    _rate, _dividend, _vol
)

generalized_call_with_greeks = _greeks_leaf(
    "generalized_call_with_greeks", generalized_call, price_node="call_price"
)
generalized_put_with_greeks = _greeks_leaf(
    "generalized_put_with_greeks", generalized_put, price_node="put_price"
)
merton_call_with_greeks = _greeks_leaf(
    "merton_call_with_greeks", merton_call, price_node="call_price"
)
merton_put_with_greeks = _greeks_leaf(
    "merton_put_with_greeks", merton_put, price_node="put_price"
)

option_price_router = Router.by(OPT.option_model, OPT.option_kind)
option_price_router.register(
    OPT.option_model == OptionModel.GENERALIZED, OPT.option_kind == OptionKind.CALL
)(lambda: generalized_call)
option_price_router.register(
    OPT.option_model == OptionModel.GENERALIZED, OPT.option_kind == OptionKind.PUT
)(lambda: generalized_put)
option_price_router.register(
    OPT.option_model == OptionModel.MERTON, OPT.option_kind == OptionKind.CALL
)(lambda: merton_call)
option_price_router.register(
    OPT.option_model == OptionModel.MERTON, OPT.option_kind == OptionKind.PUT
)(lambda: merton_put)

option_greeks_router = Router.by(OPT.option_model, OPT.option_kind)
option_greeks_router.register(
    OPT.option_model == OptionModel.GENERALIZED, OPT.option_kind == OptionKind.CALL
)(lambda: generalized_call_with_greeks)
option_greeks_router.register(
    OPT.option_model == OptionModel.GENERALIZED, OPT.option_kind == OptionKind.PUT
)(lambda: generalized_put_with_greeks)
option_greeks_router.register(
    OPT.option_model == OptionModel.MERTON, OPT.option_kind == OptionKind.CALL
)(lambda: merton_call_with_greeks)
option_greeks_router.register(
    OPT.option_model == OptionModel.MERTON, OPT.option_kind == OptionKind.PUT
)(lambda: merton_put_with_greeks)

option_router = option_price_router

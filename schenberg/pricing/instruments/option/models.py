"""The two option models, both built from :data:`generalized_bsm_core`.

* **GENERALIZED** — ``cost_of_carry`` (``b``) is joined straight from a carry
  curve. This is the knob; pick ``b`` and you get any BSM variant.
* **MERTON** — ``cost_of_carry`` is *derived* in-graph as ``b = r - q`` from a
  dividend curve, the classic continuous-dividend stock option.

Each model fans out into a call graph and a put graph via ``compose`` (a single
graph composed = a fresh clone with its own outputs), and the four leaves are
dispatched by :data:`option_router`.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.columns import cols
from schenberg.core.graph import ExprGraph
from schenberg.core.router import Router
from schenberg.domain.enums import OptionKind, OptionModel
from schenberg.domain.schemas.option import OptionTrade
from schenberg.market_data.requirements import require
from schenberg.pricing.instruments.option.core import generalized_bsm_core

OPT = cols(OptionTrade)

# --- market requirements (join on underlying + tenor) ------------------------
_rate = require(
    "curves",
    ("id_indexador", "id_indexador"),
    ("payment_days", "tenor_days"),
    outputs={"zero_rate": "rate"},
)
_carry = require(
    "carry_curve",
    ("id_indexador", "id_indexador"),
    ("payment_days", "tenor_days"),
    outputs={"cost_of_carry": "cost_of_carry"},
)
_dividend = require(
    "dividend_curve",
    ("id_indexador", "id_indexador"),
    ("payment_days", "tenor_days"),
    outputs={"div_yield": "div_yield"},
)


def _leaf(name: str, *graphs: ExprGraph, price_node: str) -> ExprGraph:
    """A priced clone of the core exposing price + the terms the Greeks reuse."""
    return ExprGraph.compose(name, generalized_bsm_core, *graphs).with_outputs(
        "pricing", d1="d1", d2="d2", price=price_node, cost_of_carry="cost_of_carry"
    )


# --- GENERALIZED: cost of carry is a joined market column --------------------
generalized_call = _leaf("generalized_call", price_node="call_price").with_market(_rate, _carry)
generalized_put = _leaf("generalized_put", price_node="put_price").with_market(_rate, _carry)


# --- MERTON: cost of carry derived as b = r - q ------------------------------
merton_carry = ExprGraph("merton_carry")


@merton_carry.node(tags=("merton",), description="Merton cost of carry b = r - q.")
def cost_of_carry(rate: pl.Expr, div_yield: pl.Expr) -> pl.Expr:
    return rate - div_yield


merton_call = _leaf("merton_call", merton_carry, price_node="call_price").with_market(
    _rate, _dividend
)
merton_put = _leaf("merton_put", merton_carry, price_node="put_price").with_market(_rate, _dividend)


option_router = Router.by(OPT.option_model, OPT.option_kind)
option_router.register(
    OPT.option_model == OptionModel.GENERALIZED, OPT.option_kind == OptionKind.CALL
)(lambda: generalized_call)
option_router.register(
    OPT.option_model == OptionModel.GENERALIZED, OPT.option_kind == OptionKind.PUT
)(lambda: generalized_put)
option_router.register(OPT.option_model == OptionModel.MERTON, OPT.option_kind == OptionKind.CALL)(
    lambda: merton_call
)
option_router.register(OPT.option_model == OptionModel.MERTON, OPT.option_kind == OptionKind.PUT)(
    lambda: merton_put
)

"""Shared swap-leg valuation primitives.

A swap leg is the shared
:data:`~schenberg.pricing.discounting.discounted_cashflow_graph` backbone plus a
direction (pay/receive): its *future value* is just its signed cashflow, which the
backbone then discounts — the exact same machine a forward uses, with no FX step.
A leg differs only in the payoff that produces ``cashflow_amount`` (see
:mod:`.legs`). The ``pay_receive`` sign is the leg's signed quantity, the same
role ``side`` plays for a structured-product leg.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import FormulaGraph
from schenberg.domain.enums import PayReceive
from schenberg.domain.schemas import LegPricing
from schenberg.market_data.curves import CurveSpec
from schenberg.pricing.discounting import discounted_cashflow_graph

_leg_payoff = FormulaGraph("swap_leg_payoff")


@_leg_payoff.formula(dtype=pl.Float64, tags=("direction",))
def pay_receive_sign(pay_receive: pl.Expr) -> pl.Expr:
    return pl.when(pay_receive == PayReceive.RECEIVE.value).then(1.0).otherwise(-1.0)


# A leg's future value is its signed cashflow; the shared backbone discounts it
# into present_value. Exposed under the LegPricing.signed_cashflow column name.
@_leg_payoff.formula(dtype=pl.Float64, tags=("cashflow",))
def future_value(cashflow_amount: pl.Expr, pay_receive_sign: pl.Expr) -> pl.Expr:
    return cashflow_amount * pay_receive_sign


# pv is the shared present_value; signed_cashflow is the leg's future_value.
swap_leg_valuation_graph = FormulaGraph.compose(
    "swap_leg_valuation", discounted_cashflow_graph, _leg_payoff
).returns("pricing", LegPricing, signed_cashflow="future_value", pv="present_value")

# The "pricing" view carries through compose, so base_swap_leg only adds its
# market — no need to re-declare the view.
base_swap_leg_graph = FormulaGraph.assemble(
    "base_swap_leg",
    swap_leg_valuation_graph,
    market={"zero_rate": CurveSpec("curves").value("zero_rate")},
)

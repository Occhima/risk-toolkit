"""Shared swap-leg valuation primitives.

A swap leg is the shared :data:`~schenberg.pricing.discounting.discount_graph`
backbone plus a direction (pay/receive) and a discounted, signed cashflow — the
same "discount a cashflow" recipe a forward uses, differing only in the payoff a
leg feeds in (see :mod:`.legs`).
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import ExprGraph
from schenberg.core.market import curve
from schenberg.domain.enums import PayReceive
from schenberg.domain.schemas import LegPricing
from schenberg.pricing.discounting import discount_graph

_leg_payoff = ExprGraph("swap_leg_payoff")


@_leg_payoff.node(dtype=pl.Float64, tags=("direction",))
def pay_receive_sign(pay_receive: pl.Expr) -> pl.Expr:
    return pl.when(pay_receive == PayReceive.RECEIVE.value).then(1.0).otherwise(-1.0)


@_leg_payoff.node(dtype=pl.Float64, tags=("cashflow",))
def signed_cashflow(cashflow_amount: pl.Expr, pay_receive_sign: pl.Expr) -> pl.Expr:
    return cashflow_amount * pay_receive_sign


@_leg_payoff.node(dtype=pl.Float64, tags=("pricing",))
def pv(signed_cashflow: pl.Expr, discount_factor: pl.Expr) -> pl.Expr:
    return signed_cashflow * discount_factor


swap_leg_valuation_graph = ExprGraph.compose(
    "swap_leg_valuation", discount_graph, _leg_payoff
).with_outputs("pricing", LegPricing)

base_swap_leg_graph = (
    ExprGraph.compose("base_swap_leg", swap_leg_valuation_graph)
    .with_market(curve("zero_rate"))
    .with_outputs("pricing", LegPricing)
)

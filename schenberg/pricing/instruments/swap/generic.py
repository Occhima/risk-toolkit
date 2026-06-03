"""Shared swap-leg valuation primitives.

A swap leg is the shared :data:`~schenberg.pricing.discounting.discount_graph`
backbone plus a direction (pay/receive) and a discounted, signed cashflow — the
same "discount a cashflow" recipe a forward uses, differing only in the payoff a
leg feeds in (see :mod:`.legs`).
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import FormulaGraph
from schenberg.core.market import curve
from schenberg.domain.enums import PayReceive
from schenberg.domain.schemas import LegPricing
from schenberg.pricing.discounting import discount_graph

_leg_payoff = FormulaGraph("swap_leg_payoff")


@_leg_payoff.formula(dtype=pl.Float64, tags=("direction",))
def pay_receive_sign(pay_receive: pl.Expr) -> pl.Expr:
    return pl.when(pay_receive == PayReceive.RECEIVE.value).then(1.0).otherwise(-1.0)


@_leg_payoff.formula(dtype=pl.Float64, tags=("cashflow",))
def signed_cashflow(cashflow_amount: pl.Expr, pay_receive_sign: pl.Expr) -> pl.Expr:
    return cashflow_amount * pay_receive_sign


@_leg_payoff.formula(dtype=pl.Float64, tags=("pricing",))
def pv(signed_cashflow: pl.Expr, discount_factor: pl.Expr) -> pl.Expr:
    return signed_cashflow * discount_factor


swap_leg_valuation_graph = FormulaGraph.compose(
    "swap_leg_valuation", discount_graph, _leg_payoff
).returns("pricing", LegPricing)

base_swap_leg_graph = (
    FormulaGraph.compose("base_swap_leg", swap_leg_valuation_graph)
    .uses_market(curve("zero_rate"))
    .returns("pricing", LegPricing)
)

"""Swap-leg valuation: shared :class:`Term` builders and the default leg.

A swap leg is the shared discount-a-signed-cashflow backbone plus a direction
(pay/receive): its *future value* is its signed cashflow, which the backbone then
discounts -- the same machine a forward uses, minus the FX step. A leg differs
only in the payoff that produces ``cashflow_amount`` (see :mod:`.legs`); the
``pay_receive`` sign is the leg's signed quantity.

:func:`assemble_leg` wires the common terms onto a graph given its
``cashflow_amount`` term and publishes the ``LegPricing`` view. :data:`base_swap_leg_graph`
is the router default: it takes ``cashflow_amount`` straight from the input column.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from schenberg.core.graph import FormulaGraph, Term, uses
from schenberg.domain.enums import PayReceive
from schenberg.domain.schemas import LegPricing, SwapLegInput
from schenberg.market_data.curves import CurveSpec
from schenberg.pricing.discounting import (
    discount_factor_term,
    present_value_term,
    year_fraction_term,
)

CURVES = CurveSpec("curves")


def discount_curve(g: FormulaGraph, t: Any) -> Any:
    """Declare the zero-rate discount curve as a market term on ``g``."""
    return g.market(
        zero_rate=CURVES.value("zero_rate", indexer=t.id_indexador, tenor=t.payment_days)
    )


def pay_receive_sign_term(g: FormulaGraph, *, pay_receive: Term[str]) -> Term[float]:
    @g.formula(tags=("direction",), description="Leg direction sign: +1 receive, -1 pay.")
    def pay_receive_sign(pr: pl.Expr = uses(pay_receive)) -> pl.Expr:
        return pl.when(pr == PayReceive.RECEIVE.value).then(1.0).otherwise(-1.0)

    return pay_receive_sign


def assemble_leg(
    g: FormulaGraph,
    *,
    t: Any,
    zero_rate: Term[float],
    cashflow_amount: Term[float],
    year_fraction: Term[float],
) -> FormulaGraph:
    """Discount the leg's signed cashflow and publish the ``LegPricing`` view."""
    discount_factor = discount_factor_term(g, zero_rate=zero_rate, year_fraction=year_fraction)
    sign = pay_receive_sign_term(g, pay_receive=t.pay_receive)

    @g.formula(tags=("cashflow",), description="Signed cashflow = cashflow_amount * direction.")
    def signed_cashflow(cf: pl.Expr = uses(cashflow_amount), s: pl.Expr = uses(sign)) -> pl.Expr:
        return cf * s

    pv = present_value_term(g, future_value=signed_cashflow, discount_factor=discount_factor)
    g.returns(
        "pricing",
        LegPricing,
        year_fraction=year_fraction,
        discount_factor=discount_factor,
        cashflow_amount=cashflow_amount,
        signed_cashflow=signed_cashflow,
        pv=pv,
    )
    return g


def base_swap_leg() -> FormulaGraph:
    """The default leg: ``cashflow_amount`` is supplied as an input column."""
    g = FormulaGraph("base_swap_leg", input=SwapLegInput)
    t = g.input
    m = discount_curve(g, t)
    year_fraction = year_fraction_term(g, payment_days=t.payment_days)
    return assemble_leg(
        g,
        t=t,
        zero_rate=m.zero_rate,
        cashflow_amount=t.cashflow_amount,
        year_fraction=year_fraction,
    )


base_swap_leg_graph = base_swap_leg()

"""Swap-leg valuation: shared :class:`Term` builders and the default leg.

A swap leg is **pure component pricing**: it discounts a ``cashflow_amount`` into a
present value and knows nothing about pay/receive, ativo/passivo, or any position
direction. ``pv = cashflow_amount * discount_factor`` — no sign. The direction
(``leg_weight``) and the ativo/passivo split live one layer up, in the swap
:class:`~schenberg.core.structure.Structure` (see :mod:`.structure`).

A leg differs from another only in the payoff that produces ``cashflow_amount``
(see :mod:`.legs`). :func:`assemble_leg` wires the common discount terms onto a
graph given that ``cashflow_amount`` term and publishes the ``LegPricing`` view.
:data:`base_swap_leg_graph` is the router default: it takes ``cashflow_amount``
straight from the input column.
"""

from __future__ import annotations

from typing import Any

from schenberg.core.graph import FormulaGraph, Term
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


def assemble_leg(
    g: FormulaGraph,
    *,
    zero_rate: Term[float],
    cashflow_amount: Term[float],
    year_fraction: Term[float],
) -> FormulaGraph:
    """Discount the leg's cashflow and publish the pure ``LegPricing`` view.

    No direction, no sign: ``pv = cashflow_amount * discount_factor``.
    """
    discount_factor = discount_factor_term(g, zero_rate=zero_rate, year_fraction=year_fraction)
    pv = present_value_term(g, future_value=cashflow_amount, discount_factor=discount_factor)
    g.returns(
        "pricing",
        LegPricing,
        year_fraction=year_fraction,
        discount_factor=discount_factor,
        cashflow_amount=cashflow_amount,
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
        zero_rate=m.zero_rate,
        cashflow_amount=t.cashflow_amount,
        year_fraction=year_fraction,
    )


base_swap_leg_graph = base_swap_leg()

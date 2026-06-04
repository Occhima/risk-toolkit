"""Fixed-rate swap leg: cashflow = notional * fixed_rate * accrual."""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import FormulaGraph, uses
from schenberg.domain.enums import SwapLegKind
from schenberg.domain.schemas import SwapLegInput
from schenberg.pricing.discounting import year_fraction_term
from schenberg.pricing.instruments.swap.generic import assemble_leg, discount_curve
from schenberg.pricing.instruments.swap.legs.registry import register


def _build() -> FormulaGraph:
    g = FormulaGraph("fixed_swap_leg", input=SwapLegInput)
    t = g.input
    m = discount_curve(g, t)
    year_fraction = year_fraction_term(g, payment_days=t.payment_days)

    @g.formula(tags=("fixed", "cashflow"))
    def cashflow_amount(
        notional: pl.Expr = uses(t.notional),
        fixed_rate: pl.Expr = uses(t.fixed_rate),
        accrual: pl.Expr = uses(t.accrual),
    ) -> pl.Expr:
        return notional * fixed_rate * accrual

    return assemble_leg(
        g, zero_rate=m.zero_rate, cashflow_amount=cashflow_amount, year_fraction=year_fraction
    )


fixed_swap_leg_graph = register(SwapLegKind.FIXED.value, graph=_build())

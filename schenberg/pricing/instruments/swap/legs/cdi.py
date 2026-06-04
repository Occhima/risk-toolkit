"""CDI swap leg: cashflow = notional * forward_rate * accrual.

CDI projects off the ``forward_rate`` carried on the curve, then takes the same
notional * rate * accrual shape as a fixed leg.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import FormulaGraph, uses
from schenberg.domain.enums import SwapLegKind
from schenberg.domain.schemas import SwapLegInput
from schenberg.pricing.discounting import year_fraction_term
from schenberg.pricing.instruments.swap.generic import assemble_leg
from schenberg.pricing.instruments.swap.legs.registry import CURVES, register


def _build() -> FormulaGraph:
    g = FormulaGraph("cdi_swap_leg", input=SwapLegInput)
    t = g.input
    m = g.market(
        zero_rate=CURVES.value("zero_rate", indexer=t.id_indexador, tenor=t.payment_days),
        forward_rate=CURVES.value("forward_rate", indexer=t.id_indexador, tenor=t.payment_days),
    )
    year_fraction = year_fraction_term(g, payment_days=t.payment_days)

    @g.formula(tags=("cashflow", "cdi"))
    def cashflow_amount(
        notional: pl.Expr = uses(t.notional),
        forward_rate: pl.Expr = uses(m.forward_rate),
        accrual: pl.Expr = uses(t.accrual),
    ) -> pl.Expr:
        return notional * forward_rate * accrual

    return assemble_leg(
        g, zero_rate=m.zero_rate, cashflow_amount=cashflow_amount, year_fraction=year_fraction
    )


cdi_swap_leg_graph = register(SwapLegKind.CDI.value, graph=_build())

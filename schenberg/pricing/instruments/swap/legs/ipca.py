"""IPCA / CPI inflation-linked swap leg.

cashflow = notional * (projected/base) * (1 + real_coupon * T) - notional.
IPCA and CPI share the same payoff and market; only the routed kind differs.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import FormulaGraph, uses
from schenberg.domain.enums import SwapLegKind
from schenberg.domain.schemas import SwapLegInput
from schenberg.pricing.discounting import year_fraction_term
from schenberg.pricing.instruments.swap.generic import assemble_leg
from schenberg.pricing.instruments.swap.legs.registry import CURVES, FIXINGS, PROJECTED, register


def _build(name: str) -> FormulaGraph:
    g = FormulaGraph(name, input=SwapLegInput)
    t = g.input
    m = g.market(
        zero_rate=CURVES.value("zero_rate", indexer=t.id_indexador, tenor=t.payment_days),
        base_index=FIXINGS.fixing(indexer=t.id_indexador, date=t.base_date),
        projected_index=PROJECTED.value(
            "projected_index", indexer=t.id_indexador, tenor=t.payment_days
        ),
    )
    year_fraction = year_fraction_term(g, payment_days=t.payment_days)

    @g.formula(tags=("inflation",))
    def inflation_factor(
        base_index: pl.Expr = uses(m.base_index),
        projected_index: pl.Expr = uses(m.projected_index),
    ) -> pl.Expr:
        return projected_index / base_index

    @g.formula(tags=("coupon",))
    def real_coupon_factor(
        real_coupon: pl.Expr = uses(t.real_coupon), T: pl.Expr = uses(year_fraction)
    ) -> pl.Expr:
        return 1.0 + real_coupon * T

    @g.formula(tags=("cashflow",))
    def cashflow_amount(
        notional: pl.Expr = uses(t.notional),
        inflation_factor: pl.Expr = uses(inflation_factor),
        real_coupon_factor: pl.Expr = uses(real_coupon_factor),
    ) -> pl.Expr:
        return notional * inflation_factor * real_coupon_factor - notional

    return assemble_leg(
        g, zero_rate=m.zero_rate, cashflow_amount=cashflow_amount, year_fraction=year_fraction
    )


ipca_swap_leg_graph = register(SwapLegKind.IPCA.value, graph=_build("ipca_swap_leg"))
cpi_swap_leg_graph = register("CPI", graph=_build("cpi_swap_leg"))

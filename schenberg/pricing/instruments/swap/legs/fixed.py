"""Fixed-rate swap leg formulas."""

from __future__ import annotations

import polars as pl

from schenberg.core.columns import cols
from schenberg.core.graph import ExprGraph
from schenberg.core.market import curve
from schenberg.domain.enums import SwapLegKind
from schenberg.domain.schemas import LegPricing, SwapLegInput
from schenberg.pricing.instruments.swap.generic import swap_leg_valuation_graph
from schenberg.pricing.instruments.swap.router import swap_leg_router

L = cols(SwapLegInput)

fixed_leg_cashflow_graph = ExprGraph("fixed_swap_leg_cashflow")


@fixed_leg_cashflow_graph.node(dtype=pl.Float64, tags=("fixed", "cashflow"))
def cashflow_amount(
    notional: pl.Expr,
    fixed_rate: pl.Expr,
    accrual: pl.Expr,
) -> pl.Expr:
    return notional * fixed_rate * accrual


@swap_leg_router.register(L.leg_kind == SwapLegKind.FIXED.value)
def fixed_swap_leg_graph() -> ExprGraph:
    return (
        ExprGraph.compose(
            "fixed_swap_leg",
            swap_leg_valuation_graph,
            fixed_leg_cashflow_graph,
        )
        .with_market(curve("zero_rate"))
        .with_outputs("pricing", LegPricing)
    )

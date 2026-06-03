"""CDI swap leg formulas."""

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

cdi_cashflow_graph = ExprGraph("cdi_cashflow")


@cdi_cashflow_graph.node(tags=("projection", "cdi"))
def projected_rate(forward_rate: pl.Expr) -> pl.Expr:
    return forward_rate


@cdi_cashflow_graph.node(tags=("cashflow", "cdi"))
def cashflow_amount(notional: pl.Expr, projected_rate: pl.Expr, accrual: pl.Expr) -> pl.Expr:
    return notional * projected_rate * accrual


@swap_leg_router.register(L.leg_kind == SwapLegKind.CDI.value)
def cdi_swap_leg_graph() -> ExprGraph:
    return (
        ExprGraph.compose("cdi_swap_leg", swap_leg_valuation_graph, cdi_cashflow_graph)
        .with_market(curve("zero_rate", "forward_rate"))
        .with_outputs("pricing", LegPricing)
    )

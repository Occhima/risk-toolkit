"""Fixed-rate swap leg formulas."""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import ExprGraph
from schenberg.domain.enums import SwapLegKind
from schenberg.pricing.instruments.swap.legs.registry import register_leg

fixed_leg_cashflow_graph = ExprGraph("fixed_swap_leg_cashflow")


@fixed_leg_cashflow_graph.node(dtype=pl.Float64, tags=("fixed", "cashflow"))
def cashflow_amount(
    notional: pl.Expr,
    fixed_rate: pl.Expr,
    accrual: pl.Expr,
) -> pl.Expr:
    return notional * fixed_rate * accrual


# Fixed legs read only the discount curve (the registry default).
fixed_swap_leg_graph = register_leg(
    SwapLegKind.FIXED.value,
    name="fixed_swap_leg",
    cashflow=fixed_leg_cashflow_graph,
)

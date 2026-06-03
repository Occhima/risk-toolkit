"""CDI swap leg formulas."""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import ExprGraph
from schenberg.core.market import curve
from schenberg.domain.enums import SwapLegKind
from schenberg.pricing.instruments.swap.legs.registry import register_leg

cdi_cashflow_graph = ExprGraph("cdi_cashflow")


@cdi_cashflow_graph.node(tags=("projection", "cdi"))
def projected_rate(forward_rate: pl.Expr) -> pl.Expr:
    return forward_rate


@cdi_cashflow_graph.node(tags=("cashflow", "cdi"))
def cashflow_amount(notional: pl.Expr, projected_rate: pl.Expr, accrual: pl.Expr) -> pl.Expr:
    return notional * projected_rate * accrual


# CDI projects off the forward_rate column carried on the discount curve.
cdi_swap_leg_graph = register_leg(
    SwapLegKind.CDI.value,
    name="cdi_swap_leg",
    cashflow=cdi_cashflow_graph,
    market=[curve("zero_rate", "forward_rate")],
)

"""IPCA inflation-linked swap leg formulas."""

from __future__ import annotations

import polars as pl

from schenberg.core.columns import cols
from schenberg.core.graph import ExprGraph
from schenberg.core.market import curve, fixing, projected_index
from schenberg.domain.enums import SwapLegKind
from schenberg.domain.schemas import LegPricing, SwapLegInput
from schenberg.pricing.instruments.swap.generic import swap_leg_valuation_graph
from schenberg.pricing.instruments.swap.router import swap_leg_router

L = cols(SwapLegInput)

ipca_cashflow_graph = ExprGraph("ipca_cashflow")


@ipca_cashflow_graph.node(tags=("inflation",))
def inflation_factor(base_index: pl.Expr, projected_index: pl.Expr) -> pl.Expr:
    return projected_index / base_index


@ipca_cashflow_graph.node(tags=("coupon",))
def real_coupon_factor(real_coupon: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return 1.0 + real_coupon * year_fraction


@ipca_cashflow_graph.node(tags=("cashflow",))
def cashflow_amount(
    notional: pl.Expr, inflation_factor: pl.Expr, real_coupon_factor: pl.Expr
) -> pl.Expr:
    return notional * inflation_factor * real_coupon_factor - notional


@swap_leg_router.register(L.leg_kind == SwapLegKind.IPCA.value)
def ipca_swap_leg_graph() -> ExprGraph:
    return (
        ExprGraph.compose("ipca_swap_leg", swap_leg_valuation_graph, ipca_cashflow_graph)
        .with_market(curve("zero_rate"), fixing(), projected_index())
        .with_outputs("pricing", LegPricing)
    )


@swap_leg_router.register(L.leg_kind == "CPI")
def cpi_swap_leg_graph() -> ExprGraph:
    return (
        ExprGraph.compose("cpi_swap_leg", swap_leg_valuation_graph, ipca_cashflow_graph)
        .with_market(curve("zero_rate"), fixing(), projected_index())
        .with_outputs("pricing", LegPricing)
    )

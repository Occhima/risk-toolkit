"""IPCA / CPI inflation-linked swap leg formulas."""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import ExprGraph
from schenberg.core.market import curve, fixing, projected_index
from schenberg.domain.enums import SwapLegKind
from schenberg.pricing.instruments.swap.legs.registry import register_leg

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


# IPCA and CPI share the same inflation payoff and market; only the kind differs.
_inflation_market = [curve("zero_rate"), fixing(), projected_index()]

ipca_swap_leg_graph = register_leg(
    SwapLegKind.IPCA.value,
    name="ipca_swap_leg",
    cashflow=ipca_cashflow_graph,
    market=_inflation_market,
)
cpi_swap_leg_graph = register_leg(
    "CPI",
    name="cpi_swap_leg",
    cashflow=ipca_cashflow_graph,
    market=_inflation_market,
)

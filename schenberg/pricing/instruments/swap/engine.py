"""Declarative assembly: formulas, market requirements, routing, public API.

This file contains no engine logic -- only declarations.
"""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.graph import ExprGraph, Router
from schenberg.core.market import MarketSnapshot, curve, fixing, projected_index
from schenberg.domain.schemas import LegPricing, SwapInput, SwapOutput
from schenberg.pricing.instruments.swap.transforms import aggregate_swap_pv, swap_to_legs

# ---------------------------------------------------------------------------
# Shared discounting graph
# ---------------------------------------------------------------------------
discounting_graph = ExprGraph("discounting")


@discounting_graph.node(
    dtype=pl.Float64, tags=("time",), description="Business-day year fraction, 252 basis."
)
def year_fraction(payment_days: pl.Expr) -> pl.Expr:
    return payment_days / 252.0


@discounting_graph.node(
    dtype=pl.Float64,
    tags=("discounting",),
    description="Discount factor from a continuously compounded zero rate.",
)
def discount_factor(zero_rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return (-zero_rate * year_fraction).exp()


@discounting_graph.node(
    dtype=pl.Float64, tags=("pricing",), description="Present value of a signed cashflow."
)
def leg_pv(signed_cashflow: pl.Expr, discount_factor: pl.Expr) -> pl.Expr:
    return signed_cashflow * discount_factor


# ---------------------------------------------------------------------------
# CDI cashflow graph  (generic terminal node names)
# ---------------------------------------------------------------------------
cdi_cashflow_graph = ExprGraph("cdi_cashflow")


@cdi_cashflow_graph.node(tags=("projection", "cdi"))
def projected_rate(forward_rate: pl.Expr) -> pl.Expr:
    return forward_rate


@cdi_cashflow_graph.node(tags=("cashflow", "cdi"), name="cashflow_amount")
def cdi_cashflow_amount(notional: pl.Expr, projected_rate: pl.Expr, accrual: pl.Expr) -> pl.Expr:
    return notional * projected_rate * accrual


@cdi_cashflow_graph.node(tags=("cashflow",), name="signed_cashflow")
def cdi_signed_cashflow(pay_receive: pl.Expr, cashflow_amount: pl.Expr) -> pl.Expr:
    return pay_receive * cashflow_amount


# ---------------------------------------------------------------------------
# Inflation cashflow graph  (shared by IPCA and CPI). Market inputs use GENERIC
# names (base_index / projected_index), so the market bindings are pure identity
# and no with_inputs / per-family renames are needed.
# ---------------------------------------------------------------------------
inflation_cashflow_graph = ExprGraph("inflation_cashflow")


@inflation_cashflow_graph.node(tags=("inflation",))
def inflation_factor(base_index: pl.Expr, projected_index: pl.Expr) -> pl.Expr:
    return projected_index / base_index


@inflation_cashflow_graph.node(tags=("coupon",))
def real_coupon_factor(real_coupon: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return 1.0 + real_coupon * year_fraction


@inflation_cashflow_graph.node(tags=("cashflow",), name="cashflow_amount")
def inflation_cashflow_amount(
    notional: pl.Expr, inflation_factor: pl.Expr, real_coupon_factor: pl.Expr
) -> pl.Expr:
    return notional * inflation_factor * real_coupon_factor - notional


@inflation_cashflow_graph.node(tags=("cashflow",), name="signed_cashflow")
def inflation_signed_cashflow(pay_receive: pl.Expr, cashflow_amount: pl.Expr) -> pl.Expr:
    return pay_receive * cashflow_amount


# ---------------------------------------------------------------------------
# Leg graphs: registered on the router via @case. Generic node names mean NO
# with_inputs; LegPricing drives the output profile; terse market constructors.
# ---------------------------------------------------------------------------
swap_router = Router("indexador_kind")


@swap_router.case("CDI")
def cdi_leg_graph():
    return (
        ExprGraph.compose("cdi_leg", discounting_graph, cdi_cashflow_graph)
        .with_market(curve("zero_rate", "forward_rate"))
        .with_outputs("pricing", LegPricing, pv="leg_pv")
    )


@swap_router.case("IPCA")
def ipca_leg_graph():
    return (
        ExprGraph.compose("ipca_leg", discounting_graph, inflation_cashflow_graph)
        .with_market(curve("zero_rate"), fixing(), projected_index())
        .with_outputs("pricing", LegPricing, pv="leg_pv")
    )


# CPI is structurally identical to IPCA (same inflation graph + market block).
# Kept as a distinct case so the two can diverge later; collapse to one shared
# leg if you confirm the conventions match.
@swap_router.case("CPI")
def cpi_leg_graph():
    return (
        ExprGraph.compose("cpi_leg", discounting_graph, inflation_cashflow_graph)
        .with_market(curve("zero_rate"), fixing(), projected_index())
        .with_outputs("pricing", LegPricing, pv="leg_pv")
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@pa.check_types(lazy=True)
def price_swap(
    swaps: LazyFrame[SwapInput], market: MarketSnapshot, *, output_profile: str = "pricing"
) -> LazyFrame[SwapOutput]:
    """Public API. Lazy in, lazy out. Pandera validates the boundary contracts."""
    legs = swap_to_legs(swaps)
    priced = swap_router.compute_for(legs, market=market, output_profile=output_profile)
    return cast(LazyFrame[SwapOutput], aggregate_swap_pv(priced))

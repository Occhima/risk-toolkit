"""Shared swap-leg valuation primitives."""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import ExprGraph
from schenberg.core.market import curve
from schenberg.domain.enums import PayReceive
from schenberg.domain.schemas import LegPricing

swap_leg_valuation_graph = ExprGraph("swap_leg_valuation")


@swap_leg_valuation_graph.node(dtype=pl.Float64, tags=("time",))
def year_fraction(payment_days: pl.Expr) -> pl.Expr:
    return payment_days / 252.0


@swap_leg_valuation_graph.node(dtype=pl.Float64, tags=("discounting",))
def discount_factor(zero_rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return (-zero_rate * year_fraction).exp()


@swap_leg_valuation_graph.node(dtype=pl.Float64, tags=("direction",))
def pay_receive_sign(pay_receive: pl.Expr) -> pl.Expr:
    return pl.when(pay_receive == PayReceive.RECEIVE.value).then(1.0).otherwise(-1.0)


@swap_leg_valuation_graph.node(dtype=pl.Float64, tags=("cashflow",))
def signed_cashflow(cashflow_amount: pl.Expr, pay_receive_sign: pl.Expr) -> pl.Expr:
    return cashflow_amount * pay_receive_sign


@swap_leg_valuation_graph.node(dtype=pl.Float64, tags=("pricing",))
def pv(signed_cashflow: pl.Expr, discount_factor: pl.Expr) -> pl.Expr:
    return signed_cashflow * discount_factor


swap_leg_valuation_graph.with_outputs("pricing", LegPricing)

base_swap_leg_graph = (
    ExprGraph.compose("base_swap_leg", swap_leg_valuation_graph)
    .with_market(curve("zero_rate"))
    .with_outputs("pricing", LegPricing)
)

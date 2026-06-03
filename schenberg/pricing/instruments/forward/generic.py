"""Generic forward valuation graph.

The graph stays boring and instrument-agnostic:

``forward_price - strike -> future_value -> present_value -> value``
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import ExprGraph
from schenberg.domain.schemas.forward import ForwardPricing
from schenberg.market_data.curves.di import DiCurveSpec
from schenberg.market_data.fx import FxRatesSpec
from schenberg.math.expressions import (
    continuous_discount_factor_expr,
    year_fraction_252_expr,
)

DI = DiCurveSpec("di_curve")
FX = FxRatesSpec("fx_rates")


forward_valuation_graph = ExprGraph("forward_valuation")


@forward_valuation_graph.node(
    dtype=pl.Float64,
    tags=("time",),
    description="252-day year fraction.",
)
def year_fraction(payment_days: pl.Expr) -> pl.Expr:
    return year_fraction_252_expr(payment_days)


@forward_valuation_graph.node(
    dtype=pl.Float64,
    tags=("discounting",),
    description="Continuously compounded discount factor.",
)
def discount_factor(zero_rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return continuous_discount_factor_expr(zero_rate, year_fraction)


@forward_valuation_graph.node(
    dtype=pl.Float64,
    tags=("cashflow",),
    description="Generic forward unit payoff.",
)
def future_value(forward_price: pl.Expr, strike: pl.Expr) -> pl.Expr:
    return forward_price - strike


@forward_valuation_graph.node(
    dtype=pl.Float64,
    tags=("pricing",),
    description="Discount future value into local present value.",
)
def present_value(future_value: pl.Expr, discount_factor: pl.Expr) -> pl.Expr:
    return future_value * discount_factor


@forward_valuation_graph.node(
    dtype=pl.Float64,
    tags=("pricing", "fx"),
    description="Translate local present value into reporting currency.",
)
def value(present_value: pl.Expr, fx_rate: pl.Expr) -> pl.Expr:
    return present_value * fx_rate


forward_valuation_graph.with_outputs("pricing", ForwardPricing)

base_forward_graph = (
    ExprGraph.compose("base_forward", forward_valuation_graph)
    .with_market(
        DI.zero_rate(),
        FX.fx_rate(),
    )
    .with_outputs("pricing", ForwardPricing)
)

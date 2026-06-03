"""Generic forward valuation graph.

The graph keeps instrument-specific cash-flow construction separate from the
reusable valuation backbone:

``future_value -> present_value -> value``

Instrument pricers compose this graph and provide a ``future_value`` formula or
column, plus market inputs for ``zero_rate`` and ``fx_rate``.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import ExprGraph

forward_valuation_graph = ExprGraph("forward_valuation")


@forward_valuation_graph.node(
    dtype=pl.Float64,
    tags=("time",),
    description="252-day year fraction.",
)
def year_fraction(payment_days: pl.Expr) -> pl.Expr:
    return payment_days / 252.0


@forward_valuation_graph.node(
    dtype=pl.Float64,
    tags=("discounting",),
    description="Continuously compounded discount factor.",
)
def discount_factor(zero_rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return (-zero_rate * year_fraction).exp()


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


forward_valuation_graph.with_outputs(
    "pricing",
    future_value="future_value",
    present_value="present_value",
    value="value",
)

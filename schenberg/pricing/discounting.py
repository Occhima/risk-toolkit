"""Shared time-and-discount backbone.

Every discounted instrument — a forward, a swap leg — needs the same three facts:
the year fraction to maturity, the discount factor off the zero curve, and the
present value of a future cashflow. They are declared once here and composed into
each instrument's valuation graph, so "discount a future cashflow" is one
canonical statement instead of a per-instrument copy.

A forward and a swap leg are the *same machine*: both end in
``future_value * discount_factor``. They differ only in the payoff that produces
``future_value`` (``forward_price - strike`` for a forward; the signed cashflow
for a swap leg) and whether an FX step follows. So each instrument composes
:data:`discounted_cashflow_graph` and supplies a payoff that defines
``future_value``; nothing re-states the discounting itself.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import FormulaGraph
from schenberg.math.expressions import (
    continuous_discount_factor_expr,
    year_fraction_252_expr,
)

discount_graph = FormulaGraph("discounting")


@discount_graph.formula(
    dtype=pl.Float64,
    tags=("time",),
    symbol="T",
    latex=r"\frac{d}{252}",
    description="252-day year fraction (time to maturity).",
)
def year_fraction(payment_days: pl.Expr) -> pl.Expr:
    return year_fraction_252_expr(payment_days)


@discount_graph.formula(
    dtype=pl.Float64,
    tags=("discounting",),
    symbol="DF",
    latex=r"e^{-rT}",
    description="Continuously compounded discount factor off the zero curve.",
)
def discount_factor(zero_rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return continuous_discount_factor_expr(zero_rate, year_fraction)


_present_value = FormulaGraph("present_value")


@_present_value.formula(
    dtype=pl.Float64,
    tags=("pricing",),
    symbol="PV",
    latex=r"V \cdot DF",
    description="Discount a future cashflow into local present value.",
)
def present_value(future_value: pl.Expr, discount_factor: pl.Expr) -> pl.Expr:
    return future_value * discount_factor


# The shared discounted-cashflow backbone: time + discounting + the one PV step.
# Instruments compose this and define ``future_value`` (their payoff).
discounted_cashflow_graph = FormulaGraph.compose(
    "discounted_cashflow", discount_graph, _present_value
)

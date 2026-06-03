"""Shared time-and-discount backbone.

Every discounted instrument — a forward, a swap leg — needs the same two facts:
the year fraction to maturity and the discount factor off the zero curve. They
are declared once here and composed into each instrument's valuation graph, so
"discount a cashflow" is one canonical statement instead of a per-instrument
copy. New discounted instruments compose this graph and add only their payoff.
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

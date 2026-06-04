"""Shared time-and-discount term builders.

Every discounted instrument -- a forward, a swap leg -- needs the same three
facts: the year fraction to maturity, the discount factor off the zero curve, and
the present value of a future cashflow. They are stated once here as :class:`Term`
builders that register their formulas onto any graph given its boundary terms, so
"discount a future cashflow" is one canonical statement instead of a
per-instrument copy.

A forward and a swap leg are the *same machine*: both end in
``future_value * discount_factor``. They differ only in the payoff that produces
``future_value`` (``forward_price - strike`` for a forward; the signed cashflow
for a swap leg) and whether an FX step follows.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import FormulaGraph, Term, uses
from schenberg.math.expressions import (
    continuous_discount_factor_expr,
    year_fraction_252_expr,
)


def year_fraction_term(g: FormulaGraph, *, payment_days: Term[int]) -> Term[float]:
    @g.formula(
        tags=("time",),
        symbol="T",
        latex=r"\frac{d}{252}",
        description="252-day year fraction (time to maturity).",
    )
    def year_fraction(d: pl.Expr = uses(payment_days)) -> pl.Expr:
        return year_fraction_252_expr(d)

    return year_fraction


def discount_factor_term(
    g: FormulaGraph, *, zero_rate: Term[float], year_fraction: Term[float]
) -> Term[float]:
    @g.formula(
        tags=("discounting",),
        symbol="DF",
        latex=r"e^{-rT}",
        description="Continuously compounded discount factor off the zero curve.",
    )
    def discount_factor(r: pl.Expr = uses(zero_rate), T: pl.Expr = uses(year_fraction)) -> pl.Expr:
        return continuous_discount_factor_expr(r, T)

    return discount_factor


def present_value_term(
    g: FormulaGraph, *, future_value: Term[float], discount_factor: Term[float], name: str = "pv"
) -> Term[float]:
    @g.formula(
        name=name,
        tags=("pricing",),
        symbol="PV",
        latex=r"V \cdot DF",
        description="Discount a future cashflow into local present value.",
    )
    def present_value(
        fv: pl.Expr = uses(future_value), df: pl.Expr = uses(discount_factor)
    ) -> pl.Expr:
        return fv * df

    return present_value

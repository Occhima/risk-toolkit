"""Generalized Black-Scholes-Merton core, as a FormulaGraph.

One vectorized valuation parameterized by a cost of carry ``b``:

``d1, d2 -> S·e^{(b-r)T}·N(±d1) - K·e^{-rT}·N(±d2)``

Every textbook variant is just a choice of ``b``: ``b = r`` (Black-Scholes
1973), ``b = r - q`` (Merton, dividend yield), ``b = 0`` (Black-76 futures),
``b = r - r_f`` (Garman-Kohlhagen FX). The two models we expose differ only in
how ``cost_of_carry`` is supplied — see :mod:`.models`.

``vol`` arrives as a column already interpolated off the volatility surface;
``rate`` and ``cost_of_carry`` arrive via market joins.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import FormulaGraph
from schenberg.domain.schemas.option import OptionPricing
from schenberg.math.expressions import norm_cdf_expr, year_fraction_252_expr

generalized_bsm_core = FormulaGraph(
    "generalized_bsm_core",
    returns=OptionPricing,
    view="pricing",
)


@generalized_bsm_core.formula(
    tags=("time",),
    symbol="T",
    latex=r"\frac{d}{252}",
    description="252-day year fraction (time to maturity).",
)
def year_fraction(payment_days: pl.Expr) -> pl.Expr:
    return year_fraction_252_expr(payment_days)


@generalized_bsm_core.formula(
    tags=("bsm",),
    symbol="d_1",
    latex=r"\frac{\ln(S/K) + (b + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}",
    description="Standardized log-moneyness drift term d1.",
)
def d1(
    spot: pl.Expr, strike: pl.Expr, cost_of_carry: pl.Expr, vol: pl.Expr, year_fraction: pl.Expr
):
    return ((spot / strike).log() + (cost_of_carry + 0.5 * vol**2) * year_fraction) / (
        vol * year_fraction.sqrt()
    )


@generalized_bsm_core.formula(
    tags=("bsm",),
    symbol="d_2",
    latex=r"d_1 - \sigma\sqrt{T}",
    description="d2 = d1 - sigma*sqrt(T).",
)
def d2(d1: pl.Expr, vol: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return d1 - vol * year_fraction.sqrt()


@generalized_bsm_core.formula(
    tags=("bsm",), description="Carry-grown, discounted spot: S*e^{(b-r)T}."
)
def carry_spot(
    spot: pl.Expr, cost_of_carry: pl.Expr, rate: pl.Expr, year_fraction: pl.Expr
) -> pl.Expr:
    return spot * ((cost_of_carry - rate) * year_fraction).exp()


@generalized_bsm_core.formula(tags=("bsm",), description="Discounted strike: K*e^{-rT}.")
def disc_strike(strike: pl.Expr, rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return strike * (-rate * year_fraction).exp()


@generalized_bsm_core.formula(
    tags=("bsm", "call"),
    symbol="C",
    latex=r"S e^{(b-r)T}N(d_1) - K e^{-rT}N(d_2)",
    description="Generalized BSM call price.",
)
def call_price(carry_spot: pl.Expr, d1: pl.Expr, disc_strike: pl.Expr, d2: pl.Expr) -> pl.Expr:
    return carry_spot * norm_cdf_expr(d1) - disc_strike * norm_cdf_expr(d2)


@generalized_bsm_core.formula(
    tags=("bsm", "put"),
    symbol="P",
    latex=r"K e^{-rT}N(-d_2) - S e^{(b-r)T}N(-d_1)",
    description="Generalized BSM put price.",
)
def put_price(carry_spot: pl.Expr, d1: pl.Expr, disc_strike: pl.Expr, d2: pl.Expr) -> pl.Expr:
    return disc_strike * norm_cdf_expr(-d2) - carry_spot * norm_cdf_expr(-d1)

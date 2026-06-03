"""Generalized Black-Scholes-Merton core, as an ExprGraph.

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

import math

import polars as pl

from schenberg.core.graph import ExprGraph
from schenberg.domain.schemas.option import OptionPricing
from schenberg.math.expressions import year_fraction_252_expr


def norm_cdf(x: pl.Expr) -> pl.Expr:
    """Standard normal CDF, Abramowitz & Stegun 26.2.17 — vectorized, pure
    Polars, no map_elements (~7.5e-8 abs error). N(x) = 1 - N(-x)."""
    ax = x.abs()
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = t * (
        0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    pdf = (-(ax * ax) / 2.0).exp() / math.sqrt(2.0 * math.pi)
    cdf_pos = 1.0 - pdf * poly
    return pl.when(x >= 0).then(cdf_pos).otherwise(1.0 - cdf_pos)


generalized_bsm_core = ExprGraph("generalized_bsm_core")


@generalized_bsm_core.node(tags=("time",), description="252-day year fraction (time to maturity).")
def year_fraction(payment_days: pl.Expr) -> pl.Expr:
    return year_fraction_252_expr(payment_days)


@generalized_bsm_core.node(tags=("bsm",), description="Standardized log-moneyness drift term d1.")
def d1(
    spot: pl.Expr, strike: pl.Expr, cost_of_carry: pl.Expr, vol: pl.Expr, year_fraction: pl.Expr
):
    return ((spot / strike).log() + (cost_of_carry + 0.5 * vol**2) * year_fraction) / (
        vol * year_fraction.sqrt()
    )


@generalized_bsm_core.node(tags=("bsm",), description="d2 = d1 - sigma*sqrt(T).")
def d2(d1: pl.Expr, vol: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return d1 - vol * year_fraction.sqrt()


@generalized_bsm_core.node(tags=("bsm",), description="Carry-grown, discounted spot: S*e^{(b-r)T}.")
def carry_spot(
    spot: pl.Expr, cost_of_carry: pl.Expr, rate: pl.Expr, year_fraction: pl.Expr
) -> pl.Expr:
    return spot * ((cost_of_carry - rate) * year_fraction).exp()


@generalized_bsm_core.node(tags=("bsm",), description="Discounted strike: K*e^{-rT}.")
def disc_strike(strike: pl.Expr, rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return strike * (-rate * year_fraction).exp()


@generalized_bsm_core.node(tags=("bsm", "call"), description="Generalized BSM call price.")
def call_price(carry_spot: pl.Expr, d1: pl.Expr, disc_strike: pl.Expr, d2: pl.Expr) -> pl.Expr:
    return carry_spot * norm_cdf(d1) - disc_strike * norm_cdf(d2)


@generalized_bsm_core.node(tags=("bsm", "put"), description="Generalized BSM put price.")
def put_price(carry_spot: pl.Expr, d1: pl.Expr, disc_strike: pl.Expr, d2: pl.Expr) -> pl.Expr:
    return disc_strike * norm_cdf(-d2) - carry_spot * norm_cdf(-d1)


generalized_bsm_core.with_outputs("pricing", OptionPricing)

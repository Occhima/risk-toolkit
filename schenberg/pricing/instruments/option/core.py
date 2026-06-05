"""Generalized Black-Scholes-Merton, expressed as :class:`Term` builders.

One vectorized valuation parameterized by a cost of carry ``b``:

``d1, d2 -> S·e^{(b-r)T}·N(±d1) - K·e^{-rT}·N(±d2)``

Every textbook variant is just a choice of ``b``: ``b = r`` (Black-Scholes
1973), ``b = r - q`` (Merton), ``b = 0`` (Black-76 futures), ``b = r - r_f``
(Garman-Kohlhagen FX). These builders register the BSM formula *terms* onto a
graph given its boundary terms — spot/strike from the input, rate/vol/carry from
the market — so each option model graph (see :mod:`.models`) wires the same core
with its own choice of ``b``.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from schenberg.core.graph import PricingGraph, Term, uses
from schenberg.domain.enums import OptionKind
from schenberg.math.expressions import norm_cdf_expr, year_fraction_252_expr


@dataclass(frozen=True, slots=True)
class BsmTerms:
    """The terms a BSM valuation exposes, for views and Greeks to reference."""

    year_fraction: Term[float]
    d1: Term[float]
    d2: Term[float]
    carry_spot: Term[float]
    disc_strike: Term[float]
    price: Term[float]


def year_fraction_term(g: PricingGraph, *, payment_days: Term[int]) -> Term[float]:
    @g.formula(
        tags=("time",),
        symbol="T",
        latex=r"\frac{d}{252}",
        description="252-day year fraction (time to maturity).",
    )
    def year_fraction(d: pl.Expr = uses(payment_days)) -> pl.Expr:
        return year_fraction_252_expr(d)

    return year_fraction


def bsm_terms(
    g: PricingGraph,
    *,
    kind: OptionKind,
    spot: Term[float],
    strike: Term[float],
    cost_of_carry: Term[float],
    rate: Term[float],
    vol: Term[float],
    year_fraction: Term[float],
) -> BsmTerms:
    """Register the generalized-BSM price terms onto ``g`` and return them."""
    T = year_fraction

    @g.formula(
        tags=("bsm",),
        symbol="d_1",
        latex=r"\frac{\ln(S/K) + (b + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}",
        description="Standardized log-moneyness drift term d1.",
    )
    def d1(
        S: pl.Expr = uses(spot),
        K: pl.Expr = uses(strike),
        b: pl.Expr = uses(cost_of_carry),
        sigma: pl.Expr = uses(vol),
        T_: pl.Expr = uses(T),
    ) -> pl.Expr:
        return ((S / K).log() + (b + 0.5 * sigma**2) * T_) / (sigma * T_.sqrt())

    @g.formula(
        tags=("bsm",),
        symbol="d_2",
        latex=r"d_1 - \sigma\sqrt{T}",
        description="d2 = d1 - sigma*sqrt(T).",
    )
    def d2(
        d1_: pl.Expr = uses(d1),
        sigma: pl.Expr = uses(vol),
        T_: pl.Expr = uses(T),
    ) -> pl.Expr:
        return d1_ - sigma * T_.sqrt()

    @g.formula(tags=("bsm",), description="Carry-grown, discounted spot: S*e^{(b-r)T}.")
    def carry_spot(
        S: pl.Expr = uses(spot),
        b: pl.Expr = uses(cost_of_carry),
        r: pl.Expr = uses(rate),
        T_: pl.Expr = uses(T),
    ) -> pl.Expr:
        return S * ((b - r) * T_).exp()

    @g.formula(tags=("bsm",), description="Discounted strike: K*e^{-rT}.")
    def disc_strike(
        K: pl.Expr = uses(strike),
        r: pl.Expr = uses(rate),
        T_: pl.Expr = uses(T),
    ) -> pl.Expr:
        return K * (-r * T_).exp()

    if kind is OptionKind.CALL:

        @g.formula(
            tags=("bsm", "call"),
            name="price",
            symbol="C",
            latex=r"S e^{(b-r)T}N(d_1) - K e^{-rT}N(d_2)",
            description="Generalized BSM call price.",
        )
        def price(
            cs: pl.Expr = uses(carry_spot),
            d1_: pl.Expr = uses(d1),
            ds: pl.Expr = uses(disc_strike),
            d2_: pl.Expr = uses(d2),
        ) -> pl.Expr:
            return cs * norm_cdf_expr(d1_) - ds * norm_cdf_expr(d2_)

    else:

        @g.formula(
            tags=("bsm", "put"),
            name="price",
            symbol="P",
            latex=r"K e^{-rT}N(-d_2) - S e^{(b-r)T}N(-d_1)",
            description="Generalized BSM put price.",
        )
        def price(
            cs: pl.Expr = uses(carry_spot),
            d1_: pl.Expr = uses(d1),
            ds: pl.Expr = uses(disc_strike),
            d2_: pl.Expr = uses(d2),
        ) -> pl.Expr:
            return ds * norm_cdf_expr(-d2_) - cs * norm_cdf_expr(-d1_)

    return BsmTerms(
        year_fraction=year_fraction,
        d1=d1,
        d2=d2,
        carry_spot=carry_spot,
        disc_strike=disc_strike,
        price=price,
    )

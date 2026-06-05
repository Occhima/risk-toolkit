"""Closed-form generalized-BSM Greeks, as :class:`Term` builders.

The Greeks are ordinary graph terms: each one names the price terms it needs —
``d1``, ``carry_spot``, ``vol`` — and is registered onto the *same* option graph
as the price (see :mod:`schenberg.pricing.instruments.option.models`), so the
sensitivities fall out as plain Polars expressions with no Python callback.

The partials are taken wrt the same independent variables as the numeric and
autodiff backends in :mod:`schenberg.math.black_scholes` (``rho = dV/dr`` at fixed
carry ``b``, ``theta = -dV/dT``), so the three reconcile. ``eta`` is +1 for a
call, -1 for a put, derived from ``option_kind`` so one builder serves both.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from schenberg.core.graph import PricingGraph, Term, uses
from schenberg.domain.enums import OptionKind
from schenberg.math.expressions import norm_cdf_expr, norm_pdf_expr


@dataclass(frozen=True, slots=True)
class GreekTerms:
    delta: Term[float]
    gamma: Term[float]
    vega: Term[float]
    theta: Term[float]
    rho: Term[float]


def bsm_greeks_terms(
    g: PricingGraph,
    *,
    option_kind: Term[str],
    spot: Term[float],
    strike: Term[float],
    rate: Term[float],
    cost_of_carry: Term[float],
    vol: Term[float],
    year_fraction: Term[float],
    d1: Term[float],
    d2: Term[float],
    carry_spot: Term[float],
    disc_strike: Term[float],
) -> GreekTerms:
    """Register the five closed-form Greek terms onto ``g`` and return them."""

    @g.formula(tags=("greeks",), description="Call/put sign: +1 call, -1 put.")
    def eta(kind: pl.Expr = uses(option_kind)) -> pl.Expr:
        return pl.when(kind == OptionKind.CALL.value).then(1.0).otherwise(-1.0)

    @g.formula(tags=("greeks",), description="Carry discount e^{(b-r)T} = carry_spot / S.")
    def carry_discount(cs: pl.Expr = uses(carry_spot), S: pl.Expr = uses(spot)) -> pl.Expr:
        return cs / S

    @g.formula(tags=("greeks",), description="Rate discount e^{-rT} = disc_strike / K.")
    def rate_discount(ds: pl.Expr = uses(disc_strike), K: pl.Expr = uses(strike)) -> pl.Expr:
        return ds / K

    @g.formula(
        tags=("greeks",),
        symbol=r"\Delta",
        latex=r"\eta e^{(b-r)T}N(\eta d_1)",
        description="dV/dS = eta * e^{(b-r)T} * N(eta*d1).",
    )
    def delta(
        e: pl.Expr = uses(eta),
        cd: pl.Expr = uses(carry_discount),
        d1_: pl.Expr = uses(d1),
    ) -> pl.Expr:
        return e * cd * norm_cdf_expr(e * d1_)

    @g.formula(
        tags=("greeks",),
        symbol=r"\Gamma",
        latex=r"\frac{e^{(b-r)T}N'(d_1)}{S\sigma\sqrt{T}}",
        description="d2V/dS2 = e^{(b-r)T} N'(d1) / (S sigma sqrt(T)).",
    )
    def gamma(
        cd: pl.Expr = uses(carry_discount),
        d1_: pl.Expr = uses(d1),
        S: pl.Expr = uses(spot),
        sigma: pl.Expr = uses(vol),
        T_: pl.Expr = uses(year_fraction),
    ) -> pl.Expr:
        return cd * norm_pdf_expr(d1_) / (S * sigma * T_.sqrt())

    @g.formula(tags=("greeks",), description="dV/dsigma = S e^{(b-r)T} N'(d1) sqrt(T).")
    def vega(
        S: pl.Expr = uses(spot),
        cd: pl.Expr = uses(carry_discount),
        d1_: pl.Expr = uses(d1),
        T_: pl.Expr = uses(year_fraction),
    ) -> pl.Expr:
        return S * cd * norm_pdf_expr(d1_) * T_.sqrt()

    @g.formula(tags=("greeks",), description="theta = dV/dt = -dV/dT.")
    def theta(
        S: pl.Expr = uses(spot),
        K: pl.Expr = uses(strike),
        r: pl.Expr = uses(rate),
        b: pl.Expr = uses(cost_of_carry),
        sigma: pl.Expr = uses(vol),
        T_: pl.Expr = uses(year_fraction),
        e: pl.Expr = uses(eta),
        cd: pl.Expr = uses(carry_discount),
        rd: pl.Expr = uses(rate_discount),
        d1_: pl.Expr = uses(d1),
        d2_: pl.Expr = uses(d2),
    ) -> pl.Expr:
        decay = -(S * cd * norm_pdf_expr(d1_) * sigma) / (2.0 * T_.sqrt())
        carry_term = e * (b - r) * S * cd * norm_cdf_expr(e * d1_)
        rate_term = e * r * K * rd * norm_cdf_expr(e * d2_)
        return decay - carry_term - rate_term

    @g.formula(tags=("greeks",), description="dV/dr at fixed carry b.")
    def rho(
        e: pl.Expr = uses(eta),
        T_: pl.Expr = uses(year_fraction),
        K: pl.Expr = uses(strike),
        rd: pl.Expr = uses(rate_discount),
        d2_: pl.Expr = uses(d2),
        S: pl.Expr = uses(spot),
        cd: pl.Expr = uses(carry_discount),
        d1_: pl.Expr = uses(d1),
    ) -> pl.Expr:
        return e * T_ * (K * rd * norm_cdf_expr(e * d2_) - S * cd * norm_cdf_expr(e * d1_))

    return GreekTerms(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

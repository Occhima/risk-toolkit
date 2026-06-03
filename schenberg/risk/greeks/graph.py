"""Closed-form generalized-BSM Greeks, as a composable :class:`FormulaGraph`.

The Greeks are their own graph. Every node names the term it needs — ``d1``,
``carry_spot``, ``vol`` — and :meth:`FormulaGraph.compose` wires those names to the
producing nodes of :data:`generalized_bsm_core` when the two graphs are merged
(see :mod:`schenberg.pricing.instruments.option.models`). So this module never
imports the option core: it only declares the sensitivities and lets composition
supply the pricing terms.

The partials are taken wrt the *same* independent variables as the numeric and
autodiff backends in :mod:`schenberg.math.black_scholes` (``rho = dV/dr`` at
fixed carry ``b``, ``theta = -dV/dT``), so the three reconcile. ``eta`` is +1 for
a call, -1 for a put, derived from ``option_kind`` so one graph serves both.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import FormulaGraph
from schenberg.domain.enums import OptionKind
from schenberg.domain.schemas.option import OptionGreeks
from schenberg.math.expressions import norm_cdf_expr, norm_pdf_expr

bsm_greeks_graph = FormulaGraph("bsm_greeks")


@bsm_greeks_graph.formula(tags=("greeks",), description="Call/put sign: +1 call, -1 put.")
def eta(option_kind: pl.Expr) -> pl.Expr:
    return pl.when(option_kind == OptionKind.CALL.value).then(1.0).otherwise(-1.0)


@bsm_greeks_graph.formula(
    tags=("greeks",), description="Carry discount e^{(b-r)T} = carry_spot / S."
)
def carry_discount(carry_spot: pl.Expr, spot: pl.Expr) -> pl.Expr:
    return carry_spot / spot


@bsm_greeks_graph.formula(tags=("greeks",), description="Rate discount e^{-rT} = disc_strike / K.")
def rate_discount(disc_strike: pl.Expr, strike: pl.Expr) -> pl.Expr:
    return disc_strike / strike


@bsm_greeks_graph.formula(
    tags=("greeks",),
    symbol=r"\Delta",
    latex=r"\eta e^{(b-r)T}N(\eta d_1)",
    description="dV/dS = eta * e^{(b-r)T} * N(eta*d1).",
)
def delta(eta: pl.Expr, carry_discount: pl.Expr, d1: pl.Expr) -> pl.Expr:
    return eta * carry_discount * norm_cdf_expr(eta * d1)


@bsm_greeks_graph.formula(
    tags=("greeks",),
    symbol=r"\Gamma",
    latex=r"\frac{e^{(b-r)T}N'(d_1)}{S\sigma\sqrt{T}}",
    description="d2V/dS2 = e^{(b-r)T} N'(d1) / (S sigma sqrt(T)).",
)
def gamma(
    carry_discount: pl.Expr, d1: pl.Expr, spot: pl.Expr, vol: pl.Expr, year_fraction: pl.Expr
) -> pl.Expr:
    return carry_discount * norm_pdf_expr(d1) / (spot * vol * year_fraction.sqrt())


@bsm_greeks_graph.formula(tags=("greeks",), description="dV/dsigma = S e^{(b-r)T} N'(d1) sqrt(T).")
def vega(spot: pl.Expr, carry_discount: pl.Expr, d1: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
    return spot * carry_discount * norm_pdf_expr(d1) * year_fraction.sqrt()


@bsm_greeks_graph.formula(tags=("greeks",), description="theta = dV/dt = -dV/dT.")
def theta(
    spot: pl.Expr,
    strike: pl.Expr,
    rate: pl.Expr,
    cost_of_carry: pl.Expr,
    vol: pl.Expr,
    year_fraction: pl.Expr,
    eta: pl.Expr,
    carry_discount: pl.Expr,
    rate_discount: pl.Expr,
    d1: pl.Expr,
    d2: pl.Expr,
) -> pl.Expr:
    decay = -(spot * carry_discount * norm_pdf_expr(d1) * vol) / (2.0 * year_fraction.sqrt())
    carry_term = eta * (cost_of_carry - rate) * spot * carry_discount * norm_cdf_expr(eta * d1)
    rate_term = eta * rate * strike * rate_discount * norm_cdf_expr(eta * d2)
    return decay - carry_term - rate_term


@bsm_greeks_graph.formula(tags=("greeks",), description="dV/dr at fixed carry b.")
def rho(
    eta: pl.Expr,
    year_fraction: pl.Expr,
    strike: pl.Expr,
    rate_discount: pl.Expr,
    d2: pl.Expr,
    spot: pl.Expr,
    carry_discount: pl.Expr,
    d1: pl.Expr,
) -> pl.Expr:
    return (
        eta
        * year_fraction
        * (
            strike * rate_discount * norm_cdf_expr(eta * d2)
            - spot * carry_discount * norm_cdf_expr(eta * d1)
        )
    )


bsm_greeks_graph.returns("greeks", OptionGreeks)

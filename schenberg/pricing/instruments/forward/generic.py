"""Generic forward valuation graph.

The graph stays boring and instrument-agnostic: it adds a generic payoff on top
of the shared :data:`~schenberg.pricing.discounting.discount_graph` and an FX
translation, so

``forward_price - strike -> future_value -> present_value -> value``
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import FormulaGraph
from schenberg.domain.schemas.forward import ForwardPricing
from schenberg.market_data.curves.di import DiCurveSpec
from schenberg.market_data.fx import FxRatesSpec
from schenberg.pricing.discounting import discount_graph

DI = DiCurveSpec("di_curve")
FX = FxRatesSpec("fx_rates")


_forward_payoff = FormulaGraph("forward_payoff")


@_forward_payoff.formula(
    dtype=pl.Float64,
    tags=("cashflow",),
    description="Generic forward unit payoff.",
)
def future_value(forward_price: pl.Expr, strike: pl.Expr) -> pl.Expr:
    return forward_price - strike


@_forward_payoff.formula(
    dtype=pl.Float64,
    tags=("pricing",),
    description="Discount future value into local present value.",
)
def present_value(future_value: pl.Expr, discount_factor: pl.Expr) -> pl.Expr:
    return future_value * discount_factor


@_forward_payoff.formula(
    dtype=pl.Float64,
    tags=("pricing", "fx"),
    description="Translate local present value into reporting currency.",
)
def value(present_value: pl.Expr, fx_rate: pl.Expr) -> pl.Expr:
    return present_value * fx_rate


# A forward = the shared discounting backbone + a forward payoff.
forward_valuation_graph = FormulaGraph.compose(
    "forward_valuation", discount_graph, _forward_payoff
).returns("pricing", ForwardPricing)

base_forward_graph = (
    FormulaGraph.compose("base_forward", forward_valuation_graph)
    .uses_market(
        DI.zero_rate(),
        FX.fx_rate(),
    )
    .returns("pricing", ForwardPricing)
)

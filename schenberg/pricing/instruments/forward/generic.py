"""Generic forward valuation graph.

The graph stays boring and instrument-agnostic: on top of the shared
:data:`~schenberg.pricing.discounting.discounted_cashflow_graph` (which owns the
``future_value -> present_value`` discount step) it adds only the forward's
payoff and an FX translation, so

``forward_price - strike -> future_value -> present_value -> value``
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import FormulaGraph
from schenberg.domain.schemas.forward import ForwardPricing
from schenberg.market_data.curves.di import DiCurveSpec
from schenberg.market_data.fx import FxRatesSpec
from schenberg.pricing.discounting import discounted_cashflow_graph

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
    tags=("pricing", "fx"),
    description="Translate local present value into reporting currency.",
)
def value(present_value: pl.Expr, fx_rate: pl.Expr) -> pl.Expr:
    return present_value * fx_rate


# A forward = the shared discounted-cashflow backbone + a forward payoff (which
# defines future_value) + an FX step.
forward_valuation_graph = FormulaGraph.compose(
    "forward_valuation", discounted_cashflow_graph, _forward_payoff
).returns("pricing", ForwardPricing)

# The "pricing" view carries through compose, so base_forward only adds its market.
base_forward_graph = FormulaGraph.assemble(
    "base_forward",
    forward_valuation_graph,
    fixed_market=(DI.zero_rate(), FX.fx_rate()),
)

"""Generic forward valuation: shared :class:`Term` builders and the default graph.

On top of the shared discount backbone (``future_value -> present_value``) a
forward adds only its payoff and an FX translation::

    forward_price - strike -> future_value -> present_value -> value
"""

from __future__ import annotations

from typing import Any

import polars as pl

from schenberg.core.graph import FormulaGraph, Term, uses
from schenberg.domain.schemas.forward import ForwardPricing, GenericForwardLeg
from schenberg.market_data.curves.di import DiCurveSpec
from schenberg.market_data.fx import FxRatesSpec
from schenberg.pricing.discounting import (
    discount_factor_term,
    present_value_term,
    year_fraction_term,
)

DI = DiCurveSpec("di_curve")
FX = FxRatesSpec("fx_rates")


def forward_payoff_term(
    g: FormulaGraph, *, forward_price: Term[float], strike: Term[float]
) -> Term[float]:
    @g.formula(name="future_value", tags=("cashflow",), description="Generic forward unit payoff.")
    def future_value(fwd: pl.Expr = uses(forward_price), k: pl.Expr = uses(strike)) -> pl.Expr:
        return fwd - k

    return future_value


def assemble_forward(
    g: FormulaGraph,
    *,
    future_value: Term[float],
    zero_rate: Term[float],
    payment_days: Term[int],
    fx_rate: Term[float],
) -> FormulaGraph:
    """Discount the payoff, translate to reporting currency, publish ``pricing``."""
    year_fraction = year_fraction_term(g, payment_days=payment_days)
    discount_factor = discount_factor_term(g, zero_rate=zero_rate, year_fraction=year_fraction)
    present_value = present_value_term(
        g, future_value=future_value, discount_factor=discount_factor, name="present_value"
    )

    @g.formula(tags=("pricing", "fx"), description="Translate local PV to reporting currency.")
    def value(pv: pl.Expr = uses(present_value), fx: pl.Expr = uses(fx_rate)) -> pl.Expr:
        return pv * fx

    g.returns(
        "pricing",
        ForwardPricing,
        future_value=future_value,
        present_value=present_value,
        value=value,
    )
    return g


def discount_and_fx(g: FormulaGraph, t: Any) -> Any:
    """Declare the di zero curve and the FX rate as market terms on ``g``."""
    return g.market(
        zero_rate=DI.zero_rate(indexer=t.id_indexador, tenor=t.payment_days),
        fx_rate=FX.fx_rate(currency=t.currency),
    )


def base_forward() -> FormulaGraph:
    g = FormulaGraph("base_forward", input=GenericForwardLeg)
    t = g.input
    m = discount_and_fx(g, t)
    future_value = forward_payoff_term(g, forward_price=t.forward_price, strike=t.strike)
    return assemble_forward(
        g,
        future_value=future_value,
        zero_rate=m.zero_rate,
        payment_days=t.payment_days,
        fx_rate=m.fx_rate,
    )


base_forward_graph = base_forward()

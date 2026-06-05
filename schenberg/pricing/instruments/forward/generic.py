"""Generic forward valuation on the typed :class:`PricingGraph`.

On top of the shared discount backbone (``future_value -> present_value``) a
forward adds only its payoff and an FX translation::

    forward_price - strike -> future_value -> present_value -> value

The graph is pure: it reads ``g.contract`` and ``g.market`` terms declared by a
:class:`MarketRequirements` schema and never joins. Specialized forwards (energy)
reuse :func:`forward_payoff_term` and :func:`assemble_forward`, supplying only
where ``forward_price`` comes from.
"""

from __future__ import annotations

import polars as pl

from schenberg.core.graph import PricingGraph, Term, uses
from schenberg.domain.schemas.forward import ForwardPricing, GenericForwardLeg
from schenberg.market_data.requirements import MarketRequirements, requires
from schenberg.pricing.discounting import (
    discount_factor_term,
    present_value_term,
    year_fraction_term,
)
from schenberg.pricing.market import DI, FX


class ForwardDiscountRequirements(MarketRequirements[GenericForwardLeg]):
    """Discount + FX market every forward needs: the zero rate and the reporting
    FX. Energy forwards extend this with their own price read."""

    zero_rate: Term[float] = requires(DI.zero_rate())
    fx_rate: Term[float] = requires(FX.rate())


def forward_payoff_term(
    g: PricingGraph, *, forward_price: Term[float], strike: Term[float]
) -> Term[float]:
    @g.formula(name="future_value", tags=("cashflow",), description="Generic forward unit payoff.")
    def future_value(fwd: pl.Expr = uses(forward_price), k: pl.Expr = uses(strike)) -> pl.Expr:
        return fwd - k

    return future_value


def assemble_forward(
    g: PricingGraph,
    *,
    future_value: Term[float],
    zero_rate: Term[float],
    payment_days: Term[int],
    fx_rate: Term[float],
) -> PricingGraph:
    """Discount the payoff, translate to reporting currency, publish ``output``."""
    year_fraction = year_fraction_term(g, payment_days=payment_days)
    discount_factor = discount_factor_term(g, zero_rate=zero_rate, year_fraction=year_fraction)
    present_value = present_value_term(
        g, future_value=future_value, discount_factor=discount_factor, name="present_value"
    )

    @g.formula(tags=("pricing", "fx"), description="Translate local PV to reporting currency.")
    def value(pv: pl.Expr = uses(present_value), fx: pl.Expr = uses(fx_rate)) -> pl.Expr:
        return pv * fx

    return g.returns(ForwardPricing)


def base_forward() -> PricingGraph:
    """The default forward: ``forward_price`` is supplied as an input column."""
    g = PricingGraph[GenericForwardLeg, ForwardDiscountRequirements, ForwardPricing]("base_forward")
    c, m = g.contract, g.market
    future_value = forward_payoff_term(g, forward_price=c.forward_price, strike=c.strike)
    return assemble_forward(
        g,
        future_value=future_value,
        zero_rate=m.zero_rate,
        payment_days=c.payment_days,
        fx_rate=m.fx_rate,
    )


base_forward_graph = base_forward()

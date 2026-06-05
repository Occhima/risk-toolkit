"""Swap-leg valuation on the typed :class:`PricingGraph`.

A swap leg is **pure component pricing**: it discounts a ``cashflow_amount`` into a
present value and knows nothing about pay/receive or position direction.
``pv = cashflow_amount * discount_factor`` -- no sign. The direction
(``leg_weight``) lives one layer up, in the swap
:class:`~schenberg.core.structure.Structure`.

A leg differs from another only in the payoff that produces ``cashflow_amount``
(see :mod:`.legs`). :func:`assemble_leg` wires the common discount terms onto a
graph given that ``cashflow_amount`` term and publishes the ``LegPricing`` output.
:data:`base_swap_leg_graph` is the router default: it takes ``cashflow_amount``
straight from the input column.
"""

from __future__ import annotations

from schenberg.core.graph import PricingGraph, Term
from schenberg.domain.schemas import LegPricing, SwapLegInput
from schenberg.market_data.requirements import MarketRequirements, requires
from schenberg.pricing.discounting import (
    discount_factor_term,
    present_value_term,
    year_fraction_term,
)
from schenberg.pricing.market import CURVES


class DiscountRequirements(MarketRequirements[SwapLegInput]):
    """The zero-rate discount curve every leg reads."""

    zero_rate: Term[float] = requires(CURVES.zero_rate())


def assemble_leg(
    g: PricingGraph,
    *,
    zero_rate: Term[float],
    cashflow_amount: Term[float],
    year_fraction: Term[float],
) -> PricingGraph:
    """Discount the leg's cashflow and publish the pure ``LegPricing`` output.

    No direction, no sign: ``pv = cashflow_amount * discount_factor``.
    """
    discount_factor = discount_factor_term(g, zero_rate=zero_rate, year_fraction=year_fraction)
    present_value_term(g, future_value=cashflow_amount, discount_factor=discount_factor)
    return g.returns(LegPricing)


def base_swap_leg() -> PricingGraph:
    """The default leg: ``cashflow_amount`` is supplied as an input column."""
    g = PricingGraph[SwapLegInput, DiscountRequirements, LegPricing]("base_swap_leg")
    c, m = g.contract, g.market
    year_fraction = year_fraction_term(g, payment_days=c.payment_days)
    return assemble_leg(
        g,
        zero_rate=m.zero_rate,
        cashflow_amount=c.cashflow_amount,
        year_fraction=year_fraction,
    )


base_swap_leg_graph = base_swap_leg()

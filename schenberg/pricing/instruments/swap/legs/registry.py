"""Leg registration: route each leg kind to its valuation graph.

Every swap leg builds a :class:`PricingGraph` over ``SwapLegInput`` with
:func:`schenberg.pricing.instruments.swap.generic.assemble_leg`; this module just
routes the kinds to it. One payoff graph can back several kinds (e.g. IPCA and CPI).
"""

from __future__ import annotations

from schenberg.core.columns import cols
from schenberg.core.graph import PricingGraph
from schenberg.domain.schemas import SwapLegInput
from schenberg.pricing.instruments.swap.router import swap_leg_router

_L = cols(SwapLegInput)


def register(*kinds: str, graph: PricingGraph) -> PricingGraph:
    """Route every ``kind`` to ``graph`` on the swap-leg router."""
    for kind in kinds:
        swap_leg_router.when(_L.leg_kind == kind)(lambda bound=graph: bound)
    return graph

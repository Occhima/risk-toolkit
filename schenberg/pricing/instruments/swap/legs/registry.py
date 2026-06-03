"""Declarative leg registration.

Every swap leg is the same recipe: discount a signed cashflow on the shared
:data:`swap_leg_valuation_graph` backbone. The only things that vary per leg are
its *payoff* graph and the *market* it reads — so a leg is declared by those two,
not by re-spelling the compose/uses_market/returns incantation each time.
One payoff can back several leg kinds (e.g. IPCA and CPI).
"""

from __future__ import annotations

from collections.abc import Sequence

from schenberg.core.columns import cols
from schenberg.core.graph import FormulaGraph
from schenberg.core.market import MarketRequirement, curve
from schenberg.domain.schemas import LegPricing, SwapLegInput
from schenberg.pricing.instruments.swap.generic import swap_leg_valuation_graph
from schenberg.pricing.instruments.swap.router import swap_leg_router

_L = cols(SwapLegInput)


def register_leg(
    *kinds: str,
    name: str,
    cashflow: FormulaGraph,
    market: Sequence[MarketRequirement] = (curve("zero_rate"),),
) -> FormulaGraph:
    """Build a priced leg graph from a payoff + market, and route every ``kind`` to it."""
    graph = (
        FormulaGraph.compose(name, swap_leg_valuation_graph, cashflow)
        .uses_market(*market)
        .returns("pricing", LegPricing)
    )
    for kind in kinds:
        swap_leg_router.when(_L.leg_kind == kind)(lambda bound=graph: bound)
    return graph

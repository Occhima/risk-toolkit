"""Declarative leg registration.

Every swap leg is the same recipe: discount a signed cashflow on the shared
:data:`swap_leg_valuation_graph` backbone. The only things that vary per leg are
its *payoff* graph and the *market* it reads — so a leg is declared by those two,
not by re-spelling the compose/with_market/with_outputs incantation each time.
One payoff can back several leg kinds (e.g. IPCA and CPI).
"""

from __future__ import annotations

from collections.abc import Sequence

from schenberg.core.columns import cols
from schenberg.core.graph import ExprGraph
from schenberg.core.market import MarketRequirement, curve
from schenberg.domain.schemas import LegPricing, SwapLegInput
from schenberg.market_data.fx import FxRatesSpec
from schenberg.pricing.instruments.swap.generic import swap_leg_valuation_graph
from schenberg.pricing.instruments.swap.router import swap_leg_router

_L = cols(SwapLegInput)
_FX = FxRatesSpec("fx_rates")


def register_leg(
    *kinds: str,
    name: str,
    cashflow: ExprGraph,
    market: Sequence[MarketRequirement] = (curve("zero_rate"),),
) -> ExprGraph:
    """Build a priced leg graph from a payoff + market, and route every ``kind`` to it.

    FX is appended as optional market data, so any leg can be priced in a foreign
    currency just by declaring an ``fx_rates`` source and carrying a ``currency``;
    without them the leg stays local (fx rate 1.0)."""
    graph = (
        ExprGraph.compose(name, swap_leg_valuation_graph, cashflow)
        .with_market(*market, _FX.fx_rate(optional=True))
        .with_outputs("pricing", LegPricing)
    )
    for kind in kinds:
        swap_leg_router.register(_L.leg_kind == kind)(lambda bound=graph: bound)
    return graph

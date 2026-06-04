"""Declarative leg registration.

Every swap leg is the same recipe: discount a signed cashflow on the shared
:data:`swap_leg_valuation_graph` backbone. The only things that vary per leg are
its *payoff* graph and the *market* it reads — so a leg is declared by those two,
not by re-spelling the compose/for_market/returns incantation each time.
One payoff can back several leg kinds (e.g. IPCA and CPI).
"""

from __future__ import annotations

from collections.abc import Mapping

from schenberg.core.columns import cols
from schenberg.core.graph import FormulaGraph
from schenberg.core.market import MarketDependency
from schenberg.domain.schemas import SwapLegInput
from schenberg.market_data.curves import CurveSpec
from schenberg.market_data.fixings import FixingsSpec
from schenberg.pricing.instruments.swap.generic import swap_leg_valuation_graph
from schenberg.pricing.instruments.swap.router import swap_leg_router

_L = cols(SwapLegInput)

# Shared specs for the standard swap market tables.
CURVES = CurveSpec("curves")
FIXINGS = FixingsSpec("fixings")
PROJECTED = CurveSpec("projected_indexes")


def register_leg(
    *kinds: str,
    name: str,
    cashflow: FormulaGraph,
    market: Mapping[str, MarketDependency] | None = None,
) -> FormulaGraph:
    """Build a priced leg graph from a payoff + market reads, routing every ``kind`` to it.

    ``market`` maps each output column to its read; it defaults to the discount
    curve alone (the only market a fixed leg needs).
    """
    reads = market if market is not None else {"zero_rate": CURVES.value("zero_rate")}
    graph = FormulaGraph.assemble(name, swap_leg_valuation_graph, cashflow, market=reads)
    for kind in kinds:
        swap_leg_router.when(_L.leg_kind == kind)(lambda bound=graph: bound)
    return graph

"""Swap-leg routing: a choice among leg-valuation graphs keyed by ``leg_kind``.

Unknown kinds fall to the default leg, which discounts the ``cashflow_amount``
column the caller supplies. Known kinds (FIXED/CDI/IPCA/CPI) register their own
graph in :mod:`.legs`, each computing ``cashflow_amount`` from its payoff.
"""

from __future__ import annotations

from schenberg.core.columns import cols
from schenberg.core.router import Router
from schenberg.domain.schemas import SwapLegInput
from schenberg.pricing.instruments.swap.generic import base_swap_leg_graph

L = cols(SwapLegInput)

swap_leg_router = Router.on(L.leg_kind, name="swap_leg_router").default(base_swap_leg_graph)

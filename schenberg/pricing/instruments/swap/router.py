"""Swap-leg routing."""

from __future__ import annotations

from schenberg.core.columns import cols
from schenberg.core.router import Router
from schenberg.domain.schemas import SwapLegInput
from schenberg.pricing.instruments.swap.generic import base_swap_leg_graph

L = cols(SwapLegInput)

swap_leg_router = Router.by(L.leg_kind).default(base_swap_leg_graph)

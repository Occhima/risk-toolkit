"""Forward instrument routing."""

from __future__ import annotations

from schenberg.core.columns import cols
from schenberg.core.router import Router
from schenberg.domain.schemas.forward import ForwardTrade
from schenberg.pricing.instruments.forward.generic import base_forward_graph

F = cols(ForwardTrade)

forward_router = Router.on(
    F.instrument_type,
    F.forward_family,
    F.settlement_type,
).default(base_forward_graph)

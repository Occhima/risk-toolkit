from __future__ import annotations

import polars as pl

from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.forward import forward_router
from schenberg.pricing.instruments.forward.energy import price_energy_forward
from schenberg.pricing.instruments.swap.pricing import price_swap, price_swaps

__all__ = ["price_energy_forward", "price_forwards", "price_swap", "price_swaps"]


def price_forwards(forwards: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    return forward_router.compute_for(forwards, market=market, output_profile="pricing")

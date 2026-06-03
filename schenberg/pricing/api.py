from __future__ import annotations

from typing import cast

import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.domain.schemas.forward import ForwardTrade
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.forward import forward_router
from schenberg.pricing.instruments.forward.energy import price_energy_forward
from schenberg.pricing.instruments.forward.prices import price_forward_instruments
from schenberg.pricing.instruments.option import price_options, price_options_with_greeks
from schenberg.pricing.instruments.swap.pricing import price_swap, price_swaps

__all__ = [
    "compute_forward_pricing_rows",
    "price_energy_forward",
    "price_forwards",
    "price_options",
    "price_options_with_greeks",
    "price_swap",
    "price_swaps",
]


def price_forwards(forwards: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    return price_forward_instruments(cast(LazyFrame[ForwardTrade], forwards), market)


def compute_forward_pricing_rows(forwards: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    return forward_router.compute(forwards, market=market, view="pricing")

"""Public swap pricing orchestration.

A swap *is* its legs. Callers pass normalized leg rows (``SwapLegInput``) directly
— there is no wide one-row-per-swap contract. Pricing is delegated entirely to
:data:`swap_structure`: pure leg pricing, then exposure (``leg_weight``), then a
fold by ``swap_id``. The position direction is never in the pricing graph.
"""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.domain.schemas import SwapLegInput, SwapOutput
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.swap.structure import swap_structure


@pa.check_types(lazy=True)
def price_swaps(
    legs: LazyFrame[SwapLegInput],
    market: MarketSnapshot,
) -> LazyFrame[SwapOutput]:
    """Price a book of swap legs into one NPV row per ``swap_id``."""
    return cast(LazyFrame[SwapOutput], swap_structure.compute(legs, market=market))


@pa.check_types(lazy=True)
def price_swap(
    legs: LazyFrame[SwapLegInput],
    market: MarketSnapshot,
) -> LazyFrame[SwapOutput]:
    """Alias for :func:`price_swaps` (reads well for a single swap)."""
    return price_swaps(legs, market)


def stage_swaps(legs: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    """Debug helper: per-leg pure pricing *plus* exposure (``weighted_pv``),
    before the fold. Stays lazy."""
    return swap_structure.stage(legs, market=market)


def swap_components(legs: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    """Debug helper: pure per-leg component pricing, before exposure or fold.
    Stays lazy."""
    return swap_structure.components_frame(legs, market=market)

"""Public swap pricing orchestration and aggregation.

A swap *is* its legs. Callers pass normalized leg rows (``SwapLegInput``)
directly — there is no wide one-row-per-swap contract and no reshaping step:
each leg is priced by its kind and the signed PVs are summed per ``swap_id``.
"""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.domain.schemas import LegPricing, SwapLegInput, SwapOutput
from schenberg.market_data.snapshot import MarketSnapshot

# Import side-effect registrations explicitly.
from schenberg.pricing.instruments.swap import legs as _legs  # noqa: F401
from schenberg.pricing.instruments.swap.router import swap_leg_router

L = cols(SwapLegInput)
P = cols(LegPricing)


def aggregate_swap_pv(priced_legs: pl.LazyFrame) -> pl.LazyFrame:
    """Leg-level PVs -> swap-level NPV.

    ``pv`` already carries pay/receive sign, so NPV is a plain sum and per-leg
    columns are signed contributions.
    """
    return priced_legs.group_by(L.swap_id.name).agg(
        npv=P.pv.expr().sum(),
        ativo_pv=P.pv.expr().filter(pl.col(L.leg_id.name) == "ativo").sum(),
        passivo_pv=P.pv.expr().filter(pl.col(L.leg_id.name) == "passivo").sum(),
    )


@pa.check_types(lazy=True)
def price_swaps(
    legs: LazyFrame[SwapLegInput], market: MarketSnapshot, *, output_profile: str = "pricing"
) -> LazyFrame[SwapOutput]:
    priced = swap_leg_router.compute_for(legs, market=market, output_profile=output_profile)
    return cast(LazyFrame[SwapOutput], aggregate_swap_pv(priced))


@pa.check_types(lazy=True)
def price_swap(
    legs: LazyFrame[SwapLegInput], market: MarketSnapshot, *, output_profile: str = "pricing"
) -> LazyFrame[SwapOutput]:
    return price_swaps(legs, market, output_profile=output_profile)

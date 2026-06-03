"""Public swap pricing orchestration and aggregation."""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.domain.enums import PayReceive
from schenberg.domain.schemas import LegPricing, SwapInput, SwapLegInput, SwapOutput
from schenberg.market_data.snapshot import MarketSnapshot

# Import side-effect registrations explicitly.
from schenberg.pricing.instruments.swap import legs as _legs  # noqa: F401
from schenberg.pricing.instruments.swap.router import swap_leg_router

L = cols(SwapLegInput)
P = cols(LegPricing)

_COMMON = ["swap_id", "notional", "payment_days", "accrual", "base_date"]


def swap_to_legs(swaps: pl.LazyFrame) -> pl.LazyFrame:
    """Wide swap rows -> normalized long leg rows."""
    names = set(swaps.collect_schema().names())

    def currency(side: str) -> pl.Expr:
        column = f"currency_{side}"
        source = pl.col(column) if column in names else pl.lit(None, dtype=pl.Utf8)
        return source.alias(L.currency.name)

    def leg(side: str, pay_receive: PayReceive) -> pl.LazyFrame:
        return swaps.select(
            *_COMMON,
            pl.lit(side).alias(L.leg_id.name),
            pl.lit(pay_receive.value).alias(L.pay_receive.name),
            pl.col(f"id_indexador_{side}").alias(L.id_indexador.name),
            pl.col(f"indexador_kind_{side}").alias(L.leg_kind.name),
            pl.col(f"fixed_rate_{side}").alias(L.fixed_rate.name),
            pl.col(f"real_coupon_{side}").alias(L.real_coupon.name),
            pl.lit(None, dtype=pl.Float64).alias(L.cashflow_amount.name),
            currency(side),
        )

    return pl.concat(
        [leg("ativo", PayReceive.RECEIVE), leg("passivo", PayReceive.PAY)],
        how="vertical",
    )


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
    swaps: LazyFrame[SwapInput], market: MarketSnapshot, *, output_profile: str = "pricing"
) -> LazyFrame[SwapOutput]:
    legs_lf = swap_to_legs(swaps)
    priced = swap_leg_router.compute_for(legs_lf, market=market, output_profile=output_profile)
    return cast(LazyFrame[SwapOutput], aggregate_swap_pv(priced))


@pa.check_types(lazy=True)
def price_swap(
    swaps: LazyFrame[SwapInput], market: MarketSnapshot, *, output_profile: str = "pricing"
) -> LazyFrame[SwapOutput]:
    return price_swaps(swaps, market, output_profile=output_profile)

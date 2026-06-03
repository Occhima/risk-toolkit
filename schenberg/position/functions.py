from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.domain.enums import InstrumentType
from schenberg.domain.schemas.forward import ForwardPricing, ForwardTrade
from schenberg.domain.schemas.position import InstrumentValue, Position, PositionValue
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.forward import forward_router

F = cols(ForwardTrade)
P = cols(ForwardPricing)
POS = cols(Position)


@pa.check_types(lazy=True)
def mtm_forward(
    forwards: LazyFrame[ForwardTrade],
    market: MarketSnapshot,
) -> LazyFrame[InstrumentValue]:
    priced = forward_router.compute_for(
        forwards,
        market=market,
        output_profile="pricing",
    )

    result = (
        priced.group_by(F.instrument_id.name)
        .agg(value=P.value.expr().sum())
        .with_columns(instrument_type=pl.lit(InstrumentType.FORWARD.value))
        .select(
            "instrument_type",
            F.instrument_id.name,
            "value",
        )
    )

    return cast(LazyFrame[InstrumentValue], result)


@pa.check_types(lazy=True)
def value_positions(
    positions: LazyFrame[Position],
    instrument_values: LazyFrame[InstrumentValue],
) -> LazyFrame[PositionValue]:
    result = (
        positions.join(
            instrument_values,
            on=[POS.instrument_type.name, POS.instrument_id.name],
            how="left",
        )
        .rename({"value": "unit_value"})
        .with_columns(
            market_value=POS.quantity.expr() * pl.col("unit_value"),
        )
        .select(
            POS.position_id.name,
            POS.book.name,
            POS.instrument_type.name,
            POS.instrument_id.name,
            POS.quantity.name,
            "unit_value",
            "market_value",
        )
    )

    return cast(LazyFrame[PositionValue], result)


@pa.check_types(lazy=True)
def pnl_from_position_values(
    today: LazyFrame[PositionValue],
    previous: LazyFrame[PositionValue],
) -> pl.LazyFrame:
    return (
        today.rename({"market_value": "mv_today"})
        .join(
            previous.rename({"market_value": "mv_previous"}),
            on=[
                "position_id",
                "book",
                "instrument_type",
                "instrument_id",
            ],
        )
        .with_columns(
            pnl=pl.col("mv_today") - pl.col("mv_previous"),
        )
    )

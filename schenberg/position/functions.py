from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.domain.enums import InstrumentType
from schenberg.domain.schemas.forward import ForwardPricing, ForwardTrade
from schenberg.domain.schemas.position import InstrumentPrice, Position, PricedPosition
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.forward import forward_router

F = cols(ForwardTrade)
P = cols(ForwardPricing)
POS = cols(Position)
PX = cols(InstrumentPrice)


@pa.check_types(lazy=True)
def price_forward_instruments(
    forwards: LazyFrame[ForwardTrade],
    market: MarketSnapshot,
) -> LazyFrame[InstrumentPrice]:
    priced = forward_router.compute_for(
        forwards,
        market=market,
        output_profile="pricing",
    )

    result = (
        priced.group_by(F.instrument_id.name)
        .agg(price=P.value.expr().sum())
        .with_columns(instrument_type=pl.lit(InstrumentType.FORWARD.value))
        .select(
            PX.instrument_type.name,
            PX.instrument_id.name,
            PX.price.name,
        )
    )

    return cast(LazyFrame[InstrumentPrice], result)


@pa.check_types(lazy=True)
def with_prices(
    positions: LazyFrame[Position],
    prices: LazyFrame[InstrumentPrice],
) -> LazyFrame[PricedPosition]:
    result = (
        positions.join(
            prices,
            on=[POS.instrument_type.name, POS.instrument_id.name],
            how="left",
        )
        .with_columns(
            mtm=POS.quantity.expr() * PX.price.expr(),
        )
        .select(
            POS.position_id.name,
            POS.book.name,
            POS.instrument_type.name,
            POS.instrument_id.name,
            POS.quantity.name,
            PX.price.name,
            "mtm",
        )
    )

    return cast(LazyFrame[PricedPosition], result)


@pa.check_types(lazy=True)
def pnl_from_priced_positions(
    today: LazyFrame[PricedPosition],
    previous: LazyFrame[PricedPosition],
) -> pl.LazyFrame:
    result = (
        today.rename({"price": "price_today", "mtm": "mtm_today"})
        .join(
            previous.rename({"price": "price_previous", "mtm": "mtm_previous"}),
            on=[
                POS.position_id.name,
                POS.book.name,
                POS.instrument_type.name,
                POS.instrument_id.name,
                POS.quantity.name,
            ],
            how="inner",
        )
        .with_columns(
            price_pnl=POS.quantity.expr() * (pl.col("price_today") - pl.col("price_previous")),
            mtm_pnl=pl.col("mtm_today") - pl.col("mtm_previous"),
        )
    )

    return result

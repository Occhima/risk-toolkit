from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.domain.schemas.position import InstrumentPrice, Position, PricedPosition
from schenberg.pricing.instruments.forward.prices import price_forward_instruments

POS = cols(Position)
PX = cols(InstrumentPrice)

__all__ = ["pnl_from_priced_positions", "price_forward_instruments", "with_prices"]


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
            mtm=POS.side.expr() * POS.quantity.expr() * PX.price.expr(),
        )
        .select(
            POS.position_id.name,
            POS.book.name,
            POS.instrument_type.name,
            POS.instrument_id.name,
            POS.quantity.name,
            POS.side.name,
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
    today_view = today.rename({"price": "price_today", "mtm": "mtm_today"})
    previous_view = previous.select(
        POS.position_id.name,
        POS.book.name,
        POS.instrument_type.name,
        POS.instrument_id.name,
        pl.col("price").alias("price_previous"),
        pl.col("mtm").alias("mtm_previous"),
    )

    return today_view.join(
        previous_view,
        on=[
            POS.position_id.name,
            POS.book.name,
            POS.instrument_type.name,
            POS.instrument_id.name,
        ],
        how="inner",
    ).with_columns(
        price_pnl=pl.col("side")
        * pl.col("quantity")
        * (pl.col("price_today") - pl.col("price_previous")),
        mtm_pnl=pl.col("mtm_today") - pl.col("mtm_previous"),
    )

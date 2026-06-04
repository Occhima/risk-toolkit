from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.domain.enums import InstrumentType
from schenberg.domain.schemas.forward import ForwardPricing, ForwardTrade
from schenberg.domain.schemas.position import InstrumentPrice
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.forward.router import forward_router

F = cols(ForwardTrade)
P = cols(ForwardPricing)
PX = cols(InstrumentPrice)


def aggregate_forward_prices(priced: pl.LazyFrame, *, id_col: str) -> LazyFrame[InstrumentPrice]:
    """Sum priced forward legs (``value``) per instrument into ``InstrumentPrice``.

    Shared by every forward facade — the generic backbone and the energy forward —
    so the group-by/aggregate/select shape is stated once.
    """
    result = (
        priced.group_by(id_col)
        .agg(price=P.value.expr().sum())
        .with_columns(instrument_type=pl.lit(InstrumentType.FORWARD.value))
        .select(PX.instrument_type.name, PX.instrument_id.name, PX.price.name)
    )
    return cast(LazyFrame[InstrumentPrice], result)


@pa.check_types(lazy=True)
def price_forward_instruments(
    forwards: LazyFrame[ForwardTrade],
    market: MarketSnapshot,
) -> LazyFrame[InstrumentPrice]:
    priced = forward_router.compute(forwards, market=market, view="pricing")
    return aggregate_forward_prices(priced, id_col=F.instrument_id.name)

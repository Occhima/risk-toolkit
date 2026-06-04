from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.core.fold import Fold, lit_, sum_
from schenberg.domain.enums import InstrumentType
from schenberg.domain.schemas.forward import ForwardPricing, ForwardTrade
from schenberg.domain.schemas.position import InstrumentPrice
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.forward.router import forward_router

F = cols(ForwardTrade)
P = cols(ForwardPricing)
PX = cols(InstrumentPrice)

# Forward legs roll up to one InstrumentPrice row per instrument: sum the leg
# ``value`` and tag the instrument type. The same Fold is shared by every forward
# facade (generic and energy) — the aggregation semantics live in one place.
forward_price_fold = (
    Fold("forward_price", input_schema=ForwardPricing)
    .by(F.instrument_id)
    .returns(
        InstrumentPrice,
        instrument_type=lit_(InstrumentType.FORWARD.value),
        price=sum_(P.value),
    )
)


def aggregate_forward_prices(priced: pl.LazyFrame) -> LazyFrame[InstrumentPrice]:
    """Sum priced forward legs (``value``) per instrument into ``InstrumentPrice``."""
    return cast(LazyFrame[InstrumentPrice], forward_price_fold.compute(priced))


@pa.check_types(lazy=True)
def price_forward_instruments(
    forwards: LazyFrame[ForwardTrade],
    market: MarketSnapshot,
) -> LazyFrame[InstrumentPrice]:
    priced = forward_router.compute(forwards, market=market, view="pricing")
    return aggregate_forward_prices(priced)

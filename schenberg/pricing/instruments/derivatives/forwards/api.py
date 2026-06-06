from __future__ import annotations

from typing import cast

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import LazyFrame

from schenberg.core.graph import Formula
from schenberg.domain.schemas.position import InstrumentValue
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.derivatives.forwards.contracts import (
    ForwardContractPricing,
    ForwardPricing,
)
from schenberg.pricing.instruments.derivatives.forwards.formulas import (
    forward_formula,
)


@pa.check_types(lazy=True)
def price_forward(
    trades: LazyFrame[ForwardContractPricing],
    market: MarketSnapshot,
) -> LazyFrame[ForwardPricing]:
    """Price generic forward contracts lazily."""
    return cast(
        "LazyFrame[ForwardPricing]", forward_formula.compute(trades, market=market, view="output")
    )


def forward_value_frame(
    formula: Formula,
    trades: pl.LazyFrame,
    market: MarketSnapshot,
    *,
    instrument_type: str,
) -> pl.LazyFrame:
    """Project a priced forward into an :class:`InstrumentValue` frame.

    The position layer wants a **pure, own-currency** value: ``present_value``
    (the forward value discounted), tagged with the contract's own currency and a
    stable ``instrument_type``. Reporting-currency conversion is the position
    layer's concern (``ReportingFx``). Stays lazy.
    """
    priced = formula.compute(trades, market=market, view="output")
    return priced.select(
        instrument_type=pl.lit(instrument_type),
        instrument_id=pl.col("instrument_id"),
        value=pl.col("present_value"),
        currency=pl.col("currency"),
    )


@pa.check_types(lazy=True)
def forward_instrument_value(
    trades: LazyFrame[ForwardContractPricing],
    market: MarketSnapshot,
    *,
    instrument_type: str = "FORWARD",
) -> LazyFrame[InstrumentValue]:
    """Price generic forwards and emit pure :class:`InstrumentValue` rows, ready to
    feed ``position_value`` directly. Stays lazy."""
    return cast(
        "LazyFrame[InstrumentValue]",
        forward_value_frame(forward_formula, trades, market, instrument_type=instrument_type),
    )

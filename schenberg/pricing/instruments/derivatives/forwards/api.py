from __future__ import annotations

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

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
    return forward_formula.compute(trades, market=market, view="output")

from __future__ import annotations

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.derivatives.forwards.contracts import (
    ForwardPricing,
)
from schenberg.pricing.instruments.derivatives.forwards.energy.contracts import (
    EnergyForwardPricing,
)
from schenberg.pricing.instruments.derivatives.forwards.energy.market import (
    EnergyForwardMarket,
)
from schenberg.pricing.instruments.derivatives.forwards.formulas import (
    build_forward_formula,
)

energy_forward_formula = build_forward_formula(
    name="energy_forward",
    contract=EnergyForwardPricing,
    market=EnergyForwardMarket,
)


@pa.check_types(lazy=True)
def price_energy_forward(
    trades: LazyFrame[EnergyForwardPricing],
    market: MarketSnapshot,
) -> LazyFrame[ForwardPricing]:
    """Price energy forward contracts lazily."""
    return energy_forward_formula.compute(trades, market=market, view="output")

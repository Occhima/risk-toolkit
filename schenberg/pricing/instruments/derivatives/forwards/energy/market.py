from __future__ import annotations

from schenberg.core import Term
from schenberg.market_data.requirements import MarketRequirements, contract, requires
from schenberg.pricing.instruments.derivatives.forwards.energy.contracts import (
    EnergyForwardPricing,
)
from schenberg.pricing.market import CURVES, ENERGY_FWD
from schenberg.pricing.market import FX as CURRENCY


class EnergyForwardMarket(MarketRequirements[EnergyForwardPricing]):
    """Market reads for energy forward pricing.

    Keep graph term names aligned with the reusable forward formula:

        m.forward_price
        m.risk_free
        m.currency
    """

    forward_price: Term[float] = requires(ENERGY_FWD.price())
    risk_free: Term[float] = requires(CURVES.risk_free_rate().by(indexer=contract.indexer))
    currency: Term[float] = requires(CURRENCY.rate())

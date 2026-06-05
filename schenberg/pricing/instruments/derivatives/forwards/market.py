from __future__ import annotations

from schenberg.core import Term
from schenberg.market_data.requirements import MarketRequirements, contract, requires
from schenberg.pricing.instruments.derivatives.forwards.contracts import (
    ForwardContractPricing,
)
from schenberg.pricing.market import CURVES
from schenberg.pricing.market import FX as CURRENCY


class ForwardMarket(MarketRequirements[ForwardContractPricing]):
    """Market reads for generic forward pricing.

    These attribute names define the typed formula market namespace:

        m.forward_price
        m.risk_free
        m.currency

    Keep these names stable because formulas and explain/mermaid output should
    read naturally.
    """

    forward_price: Term[float] = requires(CURVES.forward_rate().by(indexer=contract.indexer))
    risk_free: Term[float] = requires(CURVES.risk_free_rate().by(indexer=contract.indexer))
    currency: Term[float] = requires(CURRENCY.rate())

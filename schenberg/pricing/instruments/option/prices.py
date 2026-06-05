"""Public option-pricing facade.

The facade orchestrates typed contracts and routers only. Volatility, rates and
carry are declared by each option graph as market terms; this module never pulls
``vol_surface`` from the snapshot directly.
"""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.domain.enums import GreeksBackend
from schenberg.domain.schemas.option import (
    OptionPrice,
    OptionPricedState,
    OptionPriceWithGreeks,
    OptionTrade,
)
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.option.models import option_price_router, option_risk_router
from schenberg.risk.greeks import GreeksEngine


@pa.check_types(lazy=True)
def price_options(
    options: LazyFrame[OptionTrade],
    market: MarketSnapshot,
) -> LazyFrame[OptionPrice]:
    """Price a book of options and return the public price contract."""
    priced = option_price_router.compute(options, market=market, view="price")
    return cast(LazyFrame[OptionPrice], priced)


def _price_option_state(
    options: LazyFrame[OptionTrade],
    market: MarketSnapshot,
) -> LazyFrame[OptionPricedState]:
    """The internal priced ``state`` view consumed by numeric/autodiff Greeks.

    Every branch of the price router also publishes ``state``; computing that view
    reuses the same case split without re-declaring it.
    """
    state = option_price_router.compute(options, market=market, view="state")
    fields = OptionPricedState.to_schema().columns.keys()
    return cast(LazyFrame[OptionPricedState], state.select(fields))


@pa.check_types(lazy=True)
def price_options_with_greeks(
    options: LazyFrame[OptionTrade],
    market: MarketSnapshot,
    *,
    backend: GreeksBackend | str = GreeksBackend.CLOSED_FORM,
) -> LazyFrame[OptionPriceWithGreeks]:
    """Price a book of options and attach delta, gamma, vega, theta and rho."""
    backend = GreeksBackend(backend)
    if backend is GreeksBackend.CLOSED_FORM:
        priced = option_risk_router.compute(options, market=market, view="output")
    else:
        state = _price_option_state(options, market)
        priced = GreeksEngine(backend).attach(state)
    fields = OptionPriceWithGreeks.to_schema().columns.keys()
    return cast(LazyFrame[OptionPriceWithGreeks], priced.select(fields))

"""Public option-pricing facade.

The facade orchestrates typed contracts and routers only. Volatility is declared
by the option graph as market data; this module never pulls ``vol_surface`` from
the snapshot directly.
"""

from __future__ import annotations

from typing import cast

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import cols
from schenberg.domain.enums import GreeksBackend
from schenberg.domain.schemas.option import (
    OptionPrice,
    OptionPricedState,
    OptionPriceWithGreeks,
    OptionTrade,
)
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.pricing.instruments.option.models import option_greeks_router, option_price_router
from schenberg.risk.greeks import GreeksEngine

STATE = cols(OptionPricedState)
PRICE = cols(OptionPrice)
PRICE_GREEKS = cols(OptionPriceWithGreeks)


@pa.check_types(lazy=True)
def price_options(
    options: LazyFrame[OptionTrade],
    market: MarketSnapshot,
) -> LazyFrame[OptionPrice]:
    """Price a book of options and return the public price contract."""
    priced = option_price_router.compute_for(options, market=market, output_profile="price")
    result = priced.select(
        PRICE.option_id.name,
        PRICE.instrument_type.name,
        PRICE.price.name,
    )
    return cast(LazyFrame[OptionPrice], result)


def _price_option_state(
    options: LazyFrame[OptionTrade],
    market: MarketSnapshot,
) -> LazyFrame[OptionPricedState]:
    priced = option_price_router.compute_for(
        options,
        market=market,
        output_profile="priced_state",
    )
    result = priced.select(
        STATE.option_id.name,
        STATE.instrument_type.name,
        STATE.option_model.name,
        STATE.option_kind.name,
        STATE.id_indexador.name,
        STATE.spot.name,
        STATE.strike.name,
        STATE.payment_days.name,
        STATE.vol.name,
        STATE.rate.name,
        STATE.cost_of_carry.name,
        STATE.year_fraction.name,
        STATE.d1.name,
        STATE.d2.name,
        STATE.price.name,
    )
    return cast(LazyFrame[OptionPricedState], result)


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
        priced = option_greeks_router.compute_for(
            options,
            market=market,
            output_profile="price_with_greeks",
        )
    else:
        state = _price_option_state(options, market)
        priced = GreeksEngine(backend).attach(state)

    result = priced.select(
        PRICE_GREEKS.option_id.name,
        PRICE_GREEKS.instrument_type.name,
        PRICE_GREEKS.price.name,
        PRICE_GREEKS.delta.name,
        PRICE_GREEKS.gamma.name,
        PRICE_GREEKS.vega.name,
        PRICE_GREEKS.theta.name,
        PRICE_GREEKS.rho.name,
    )
    return cast(LazyFrame[OptionPriceWithGreeks], result)

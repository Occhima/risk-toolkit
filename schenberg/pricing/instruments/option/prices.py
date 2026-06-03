"""Public option-pricing facade.

``price_options`` enriches the trades with an interpolated implied vol off the
surface, then routes them through the generalized BSM graph.
``price_options_with_greeks`` chains the Greek engine on top.
"""

from __future__ import annotations

import polars as pl

from schenberg.domain.enums import GreekMethod
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.volatility import VolSurface
from schenberg.pricing.instruments.option.models import option_router
from schenberg.risk.greeks import attach_greeks

_DAYS_PER_YEAR = 252.0


def _enrich(options: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    """Add time-to-maturity and the surface-interpolated implied vol."""
    surface = VolSurface.from_quotes(market.source("vol_surface").data)
    lf = options.with_columns((pl.col("payment_days") / _DAYS_PER_YEAR).alias("ttm"))
    return surface.attach(lf, ttm_col="ttm", strike_col="strike", output="vol")


def price_options(options: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    """Price a book of options. One row per option with price, d1, d2."""
    enriched = _enrich(options, market)
    return option_router.compute_for(enriched, market=market, output_profile="pricing")


def price_options_with_greeks(
    options: pl.LazyFrame,
    market: MarketSnapshot,
    *,
    method: GreekMethod | str = GreekMethod.ANALYTIC,
) -> pl.LazyFrame:
    """Price a book of options and attach delta, gamma, vega, theta, rho."""
    return attach_greeks(price_options(options, market), method=method)

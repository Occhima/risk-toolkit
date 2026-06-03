from __future__ import annotations

import polars as pl

from schenberg.core.pipeline import Workflow
from schenberg.position.functions import with_prices
from schenberg.pricing.instruments.forward.prices import price_forward_instruments

valuation_pipe = Workflow("valuation")


@valuation_pipe.stage
def forward_prices(forwards, market):
    return price_forward_instruments(forwards, market)


@valuation_pipe.stage
def priced_positions(positions, forward_prices):
    return positions.pipe(with_prices, forward_prices)


@valuation_pipe.stage
def book_mtm(priced_positions):
    return priced_positions.group_by("book").agg(
        mtm=pl.col("mtm").sum(),
    )

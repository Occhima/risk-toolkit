from __future__ import annotations

import polars as pl

from schenberg.core.pipeline import Pipe
from schenberg.position.functions import mtm_forward, value_positions

valuation_pipe = Pipe("valuation")


@valuation_pipe.stage
def forward_values(forwards, market):
    return mtm_forward(forwards, market)


@valuation_pipe.stage
def instrument_values(forward_values):
    return forward_values


@valuation_pipe.stage
def position_values(positions, instrument_values):
    return value_positions(positions, instrument_values)


@valuation_pipe.stage
def book_values(position_values):
    return position_values.group_by("book").agg(
        market_value=pl.col("market_value").sum(),
    )

from __future__ import annotations

import polars as pl

from schenberg.core.pipeline import Workflow
from schenberg.position.functions import with_prices

valuation_pipe = Workflow("valuation")


@valuation_pipe.stage
def priced_positions(positions, prices):
    return positions.pipe(with_prices, prices)


@valuation_pipe.stage
def book_mtm(priced_positions):
    return priced_positions.group_by("book").agg(
        mtm=pl.col("mtm").sum(),
    )

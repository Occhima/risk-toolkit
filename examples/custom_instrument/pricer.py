"""Public entry point for the inflation-linked energy forward.

Mirrors the shape of the built-in pricers (e.g. ``price_energy_forward``):
a normalization step, then the graph, then aggregation to instrument level.
Stays lazy end-to-end.
"""

from __future__ import annotations

import polars as pl
from schenberg.market_data.snapshot import MarketSnapshot

from .conventions import add_reference_date
from .graph import inflation_energy_graph


def price_inflation_energy(legs: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
    """Price inflation-linked energy forward legs and aggregate to instrument PV.

    ``legs`` carries one row per delivery period with: instrument_id, id_indexador,
    tenor_date, payment_days, forward_price, strike, currency.
    """
    prepared = add_reference_date(legs)
    priced = inflation_energy_graph.compute(prepared, market=market, view="output")
    return priced.group_by("instrument_id").agg(price=pl.col("value").sum())

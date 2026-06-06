"""Price a generic forward contract.

Run with:  uv run python docs/examples/01_price_a_forward.py

A forward pays ``forward_rate - strike`` at tenor, discounted back to today at
the risk-free rate in the instrument's own currency. The contract schema is
``ForwardContractPricing``; the market is a ``MarketSnapshot`` built from plain
DataFrames. Everything stays lazy until the final ``.collect()``.

``bind(trades, market, ForwardContractPricing)`` glues the market data to the
trade frame: it discovers the ``With[ForwardRate]`` and ``With[RiskFreeRate]``
mixins declared on the schema and left-joins the curve source keyed by
``(id_indexador, tenor_days)`` → trade ``(indexer, payment_days)``.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.api import price_forward

# ---------------------------------------------------------------------------
# Market: one curve keyed by (id_indexador, tenor_days)
# ---------------------------------------------------------------------------
market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 5),
    sources=[
        MarketSource(
            "curves",
            pl.DataFrame(
                {
                    # forward_rate: the projected forward price (not a yield).
                    # risk_free_rate: the continuous discount rate.
                    "id_indexador": ["DI", "DI"],
                    "tenor_days": [252, 504],
                    "forward_rate": [112.0, 115.0],
                    "risk_free_rate": [0.10, 0.10],
                }
            ).lazy(),
            unique_by=("id_indexador", "tenor_days"),
        ),
    ],
)

# ---------------------------------------------------------------------------
# Trades: two forwards, same indexer, different tenors
# ---------------------------------------------------------------------------
trades = pl.DataFrame(
    {
        "instrument_id": ["FWD-1", "FWD-2"],
        "indexer": ["DI", "DI"],
        "currency": ["BRL", "BRL"],
        "strike": [100.0, 100.0],
        "payment_days": [252, 504],
    }
).lazy()

# ---------------------------------------------------------------------------
# Price: lazy until .collect()
# ---------------------------------------------------------------------------
result = price_forward(trades, market)
print("Schema:", result.collect_schema())
print()
print(result.collect().select("instrument_id", "future_value", "present_value", "value"))

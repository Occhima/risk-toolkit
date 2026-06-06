"""Price a generic forward contract.

Run with:  uv run python docs/examples/01_price_a_forward.py

This is the simplest entry point. A forward pays ``forward_price - strike`` at
tenor, discounted back at the risk-free rate in its own currency. The contract
is typed (``ForwardContractPricing``); the market is a ``MarketSnapshot`` built
from plain DataFrames. Everything stays lazy until
the final ``.collect()``.

Two contract-rule derived dates are filled automatically from the contract terms
before validation runs — you never call ``.resolve()`` yourself:

- ``index_fixing_date``: the date the index is read. Defaults to tenor;
  ``"CPI"`` shifts it +5 calendar days.
- ``currency_fixing_date``: the date associated with the currency convention.
  Defaults to tenor; ``"EUR"`` uses the previous business day.
"""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
from pandera.typing.polars import LazyFrame
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.derivatives.forwards import ForwardContractPricing, price_forward

# ---------------------------------------------------------------------------
# Market: one curve keyed by (indexer, payment_days)
# ---------------------------------------------------------------------------
market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 5),
    sources=[
        MarketSource(
            "curves",
            pl.DataFrame(
                {
                    # forward_rate is the projected forward price (not a yield).
                    # risk_free_rate is the continuous discount rate.
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
# index_fixing_date and currency_fixing_date are omitted — the contract rules
# fill them from tenor automatically (same_day default for both "DI" / "BRL").
trades = cast(
    LazyFrame[ForwardContractPricing],
    pl.DataFrame(
        {
            "instrument_id": ["FWD-1", "FWD-2"],
            "tenor": [date(2027, 6, 5), date(2028, 6, 5)],
            "indexer": ["DI", "DI"],
            "currency": ["BRL", "BRL"],
            "strike": [100.0, 100.0],  # strike in the same units as forward_rate
            "payment_days": [252, 504],
        }
    ).lazy(),
)

# ---------------------------------------------------------------------------
# Price: lazy until .collect()
# ---------------------------------------------------------------------------
result = price_forward(trades, market)
print("Schema:", result.collect_schema())
print()
print(
    cast(pl.DataFrame, result.collect()).select(
        "instrument_id", "future_value", "present_value", "value"
    )
)

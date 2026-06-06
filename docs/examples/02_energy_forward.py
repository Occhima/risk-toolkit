"""Price an energy forward sourced from a forward curve.

Run with:  uv run python docs/examples/02_energy_forward.py

The energy pricer reuses the *generic* forward formula
(``forward_rate - strike → future_value → present_value → value``) and only
adds *where the numbers come from*:

- ``forward_rate`` (renamed from ``forward_price``) is read from an
  ``energy_forward_curve`` source keyed by ``(submarket, delivery_period)``.
- ``risk_free_rate`` is read from the ``curves`` source keyed by
  ``(id_indexador, tenor_days)`` exactly as for a generic forward.

The formula graph is identical to the generic forward — the specialisation
lives entirely in the market roles declared on ``EnergyForwardPricing``.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.api import price_energy_forward

# ---------------------------------------------------------------------------
# Market: energy curve (submarket/delivery_period) + discount curve
# ---------------------------------------------------------------------------
market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 5),
    sources=[
        # Energy forward prices keyed by (submarket, delivery_period)
        MarketSource(
            "energy_forward_curve",
            pl.DataFrame(
                {
                    "submarket": ["SE", "SE", "NE"],
                    "delivery_period": ["2026-07", "2026-08", "2026-07"],
                    "forward_price": [270.0, 260.0, 245.0],
                }
            ).lazy(),
            unique_by=("submarket", "delivery_period"),
        ),
        # Discount curve keyed by (id_indexador, tenor_days)
        MarketSource(
            "curves",
            pl.DataFrame(
                {
                    "id_indexador": ["PLD", "PLD"],
                    "tenor_days": [252, 504],
                    "risk_free_rate": [0.10, 0.10],
                }
            ).lazy(),
            unique_by=("id_indexador", "tenor_days"),
        ),
    ],
)

# ---------------------------------------------------------------------------
# Trades: two SE contracts, one NE
# ---------------------------------------------------------------------------
trades = pl.DataFrame(
    {
        "instrument_id": ["ENG-1", "ENG-2", "ENG-3"],
        "indexer": ["PLD", "PLD", "PLD"],
        "currency": ["BRL", "BRL", "BRL"],
        "strike": [250.0, 250.0, 230.0],
        "payment_days": [252, 252, 252],
        "submarket": ["SE", "SE", "NE"],
        "delivery_period": ["2026-07", "2026-08", "2026-07"],
    }
).lazy()

result = price_energy_forward(trades, market)
print(
    result.collect().select(
        "instrument_id",
        "submarket",
        "delivery_period",
        "future_value",
        "present_value",
        "value",
    )
)

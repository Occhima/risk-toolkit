"""Price an energy forward sourced from a forward curve.

Run with:  uv run python docs/examples/02_energy_forward.py

The energy pricer reuses the *generic* forward formula
(``forward_price - strike → future_value → present_value → value``) and only
adds *where the numbers come from*:

- ``forward_price`` is looked up on the energy curve by ``(submarket, delivery_period)``.
- ``risk_free`` is discounted from the DI curve keyed by ``(indexer, payment_days)``.
The formula graph never mentions "energy" — the specialisation lives entirely in
``EnergyForwardMarket`` (the market requirements) and in the ``EnergyForwardPricing``
contract which overrides the PLD fixing-date rule.

Contract rules in action: PLD's ``index_fixing_date`` is set to the **6th business
day of the month after delivery** automatically from the tenor column. You can
override it by supplying the column explicitly.
"""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
from pandera.typing.polars import LazyFrame
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.derivatives.forwards.energy import (
    EnergyForwardPricing,
    price_energy_forward,
)

# ---------------------------------------------------------------------------
# Market: energy curve + discount curve
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
# Trades: two SE contracts, one NE — different delivery periods
#
# index_fixing_date is intentionally omitted. The PLD rule fills it as the
# 6th business day of the month following tenor (e.g. 2026-07-xx).
# ---------------------------------------------------------------------------
trades = cast(
    LazyFrame[EnergyForwardPricing],
    pl.DataFrame(
        {
            "instrument_id": ["ENG-1", "ENG-1", "ENG-2"],
            "tenor": [date(2026, 7, 1), date(2026, 8, 1), date(2026, 7, 1)],
            "indexer": ["PLD", "PLD", "PLD"],
            "currency": ["BRL", "BRL", "BRL"],
            "strike": [250.0, 250.0, 230.0],
            "payment_days": [252, 252, 252],
            "submarket": ["SE", "SE", "NE"],
            "incentive": ["I0", "I0", "I0"],
            "delivery_period": ["2026-07", "2026-08", "2026-07"],
        }
    ).lazy(),
)

result = price_energy_forward(trades, market)
print(
    cast(pl.DataFrame, result.collect()).select(
        "instrument_id",
        "submarket",
        "delivery_period",
        "future_value",
        "present_value",
        "value",
        "index_fixing_date",
    )
)

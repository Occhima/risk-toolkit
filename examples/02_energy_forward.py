"""Price an energy forward sourced from a forward curve.

Run with:  uv run python examples/02_energy_forward.py

The energy pricer reuses the generic forward backbone
(``forward_price - strike -> future_value -> present_value -> value``) and only
adds *where the numbers come from*: it looks up ``forward_price`` on the energy
curve by (submarket, delivery_period), and derives the discount tenor from the
delivery period's ANBIMA settlement date (the 6th business day after month-end)
versus ``as_of`` — no ``payment_days`` column on the input. It then discounts on
the DI curve and converts via FX. The math graph never mentions "energy".
"""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
from pandera.typing.polars import LazyFrame
from schenberg.domain.schemas import EnergyForwardLeg
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.forward.energy import price_energy_forward

market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 3),
    sources=[
        MarketSource(
            "energy_forward_curve",
            pl.DataFrame(
                {
                    "submarket": ["SE", "SE"],
                    "delivery_period": ["2026-07", "2026-08"],
                    "forward_price": [120.0, 130.0],
                }
            ).lazy(),
        ),
        MarketSource(
            "di_curve",
            pl.DataFrame(
                {
                    "curve_name": ["DI"] * 366,
                    "id_indexador": [1] * 366,
                    "tenor_days": list(range(366)),
                    "zero_rate": [0.10] * 366,
                }
            ).lazy(),
        ),
        MarketSource("fx_rates", pl.DataFrame({"currency": ["BRL"], "fx_rate": [1.0]}).lazy()),
    ],
)

# One instrument (ENG-1) delivering across two monthly periods -> two legs.
legs = cast(
    LazyFrame[EnergyForwardLeg],
    pl.DataFrame(
        {
            "instrument_id": ["ENG-1", "ENG-1"],
            "instrument_type": ["FORWARD", "FORWARD"],
            "forward_family": ["ENERGY", "ENERGY"],
            "settlement_type": ["PHYSICAL", "PHYSICAL"],
            "submarket": ["SE", "SE"],
            "delivery_period": ["2026-07", "2026-08"],
            "id_indexador": [1, 1],
            "strike": [100.0, 100.0],
            "currency": ["BRL", "BRL"],
        }
    ).lazy(),
)

print(price_energy_forward(legs, market).collect())

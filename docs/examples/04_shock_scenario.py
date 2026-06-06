"""Reprice a forward under a shocked market.

Run with:  uv run python docs/examples/04_shock_scenario.py

A ``Shock`` is an **endomorphism** ``MarketSnapshot → MarketSnapshot``: it
returns a *new* snapshot with one or more sources transformed, never mutating
the original. Shocks compose associatively — you can build a stress scenario by
chaining them.

The repricing is just ``price_forward(trades, market.apply(shock))``.
The original market is untouched; you can price both base and stressed without
keeping two separate copies of the data.

Two ways to build the same +100bp parallel shift are shown here:
``MarketPath`` (lens-lite path into a source/column) and
``curve_parallel_shift`` (the canned helper). Both produce the same result.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from schenberg.market_data.path import MarketPath
from schenberg.market_data.shocks import Shock, curve_parallel_shift
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.api import price_forward

# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------
market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 5),
    sources=[
        MarketSource(
            "curves",
            pl.DataFrame(
                {
                    "id_indexador": ["DI"],
                    "tenor_days": [252],
                    "forward_rate": [112.0],
                    "risk_free_rate": [0.10],
                }
            ).lazy(),
            unique_by=("id_indexador", "tenor_days"),
        ),
    ],
)

trades = pl.DataFrame(
    {
        "instrument_id": ["FWD-1"],
        "indexer": ["DI"],
        "currency": ["BRL"],
        "strike": [100.0],
        "payment_days": [252],
    }
).lazy()

# ---------------------------------------------------------------------------
# Build a +100bp shock two equivalent ways
# ---------------------------------------------------------------------------
bump_via_path = MarketPath("curves").column("risk_free_rate").modify(lambda r: r + 0.01)
bump_canned = curve_parallel_shift(source="curves", column="risk_free_rate", shift=0.01)

print("Shock (path  ):", bump_via_path.explain())
print("Shock (canned):", bump_canned.explain())

scenario = Shock.compose(bump_via_path)

# ---------------------------------------------------------------------------
# Base and stressed prices
# ---------------------------------------------------------------------------
base_price = price_forward(trades, market).collect()
stressed_price = price_forward(trades, market.apply(scenario)).collect()

print("\n=== Base value ===")
print(base_price.select("instrument_id", "future_value", "present_value", "value"))

print("\n=== Stressed value (+100bp on risk_free_rate) ===")
print(stressed_price.select("instrument_id", "future_value", "present_value", "value"))

print("\n=== Value change (dv01-style) ===")
delta = (stressed_price.select("value") - base_price.select("value")).rename(
    {"value": "dv01_approx"}
)
print(delta)

# The original market is untouched — shocks never mutate.
original_rate = market.source("curves").data.select("risk_free_rate").collect().item()
print(f"\nOriginal risk_free_rate still {original_rate} (no mutation)")

"""Reprice a swap under a shocked market, and inspect the swap Structure.

Run with:  uv run python examples/04_shock_scenario.py

A ``Shock`` is an endomorphism ``MarketSnapshot -> MarketSnapshot``: it returns a
*new* market with sources transformed, never mutating the original, and shocks
compose. A ``MarketPath`` focuses one source/column to build one. Repricing under
a stress is just ``price_swaps(legs, market.apply(shock))``.

The swap itself is a ``Structure``: pure leg pricing, then exposure
(``weighted_pv = pv * leg_weight``), then a fold by ``swap_id``. Position
direction never touches the pricing graph — ``structure.explain()`` shows the
whole pipeline.
"""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
from pandera.typing.polars import LazyFrame
from schenberg.domain.schemas import SwapLegInput
from schenberg.market_data.path import MarketPath
from schenberg.market_data.shocks import Shock, curve_parallel_shift
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.api import price_swaps
from schenberg.pricing.instruments.swap.structure import swap_structure

market = MarketSnapshot.from_sources(
    as_of=date(2026, 6, 3),
    sources=[
        MarketSource(
            "curves",
            pl.DataFrame(
                {
                    "id_indexador": [1, 2],
                    "tenor_days": [252, 252],
                    "zero_rate": [0.10, 0.05],
                    "forward_rate": [0.12, None],
                }
            ).lazy(),
        ),
        MarketSource(
            "fixings",
            pl.DataFrame(
                {"id_indexador": [2], "fixing_date": [date(2026, 6, 3)], "fixing_value": [100.0]}
            ).lazy(),
        ),
        MarketSource(
            "projected_indexes",
            pl.DataFrame(
                {"id_indexador": [2], "tenor_days": [252], "projected_index": [106.0]}
            ).lazy(),
        ),
    ],
)

_common = {
    "notional": 1_000_000.0,
    "payment_days": 252,
    "accrual": 1.0,
    "base_date": date(2026, 6, 3),
    "fixed_rate": None,
    "cashflow_amount": None,
}
legs = cast(
    LazyFrame[SwapLegInput],
    pl.DataFrame(
        [
            {
                "swap_id": "SWP-1",
                "leg_id": "ativo",
                "leg_kind": "CDI",
                "leg_role": "ativo",
                "leg_weight": 1.0,
                "id_indexador": 1,
                "real_coupon": None,
                **_common,
            },
            {
                "swap_id": "SWP-1",
                "leg_id": "passivo",
                "leg_kind": "IPCA",
                "leg_role": "passivo",
                "leg_weight": -1.0,
                "id_indexador": 2,
                "real_coupon": 0.02,
                **_common,
            },
        ]
    ).lazy(),
)

# A +100bp parallel bump of the zero curve, built two equivalent ways.
bump = MarketPath("curves").column("zero_rate").modify(lambda r: r + 0.01)
same_bump = curve_parallel_shift(source="curves", shift=0.01)
scenario = Shock.compose(bump)  # shocks compose; this one is just the single bump

print("=== swap Structure ===")
print(swap_structure.explain())

base = price_swaps(legs, market).collect()
stressed = price_swaps(legs, market.apply(scenario)).collect()

print("\n=== base NPV ===")
print(base)
print("\n=== stressed NPV (+100bp) ===")
print(stressed)
# The original market is untouched — shocks never mutate.
print("\noriginal curve still at base level:")
print(market.source("curves").data.select("id_indexador", "zero_rate").collect())
print(f"\nshock: {same_bump.explain()}")

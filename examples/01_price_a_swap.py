"""Price a single CDI-vs-IPCA swap.

Run with:  uv run python examples/01_price_a_swap.py

A swap is one wide row. ``price_swap`` normalizes it into long leg rows, routes
each leg to the right pricer (CDI / IPCA / fixed) by ``indexador_kind``, attaches
the market curves it declares, and aggregates the leg PVs back to a swap NPV.
Everything stays lazy until ``.collect()``.
"""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
from pandera.typing.polars import LazyFrame
from schenberg.domain.schemas import SwapInput
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.api import price_swap

# --- Market: curves keyed by id_indexador, plus an IPCA fixing + projection -----
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
                    "forward_rate": [0.12, None],  # CDI projection lives on the curve
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

# --- One swap: receive CDI (ativo), pay IPCA+coupon (passivo) --------------------
swaps = cast(
    LazyFrame[SwapInput],
    pl.DataFrame(
        {
            "swap_id": ["SWP-1"],
            "notional": [1_000_000.0],
            "id_indexador_ativo": [1],
            "id_indexador_passivo": [2],
            "indexador_kind_ativo": ["CDI"],
            "indexador_kind_passivo": ["IPCA"],
            "payment_days": [252],
            "accrual": [1.0],
            "base_date": [date(2026, 6, 3)],
            "fixed_rate_ativo": [None],
            "fixed_rate_passivo": [None],
            "real_coupon_ativo": [None],
            "real_coupon_passivo": [0.02],
        }
    ).lazy(),
)

result = price_swap(swaps, market).collect()
print(result)
# npv = ativo_pv (receive CDI) + passivo_pv (pay IPCA) — both already signed.

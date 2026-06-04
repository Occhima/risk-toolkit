"""Price a single CDI-vs-IPCA swap.

Run with:  uv run python examples/01_price_a_swap.py

A swap *is* its legs. You book the legs directly (the contract is ``SwapLegInput``,
not a wide row that needs reshaping); the swap ``Structure`` routes each leg to its
pure pricer (CDI / IPCA / fixed) by ``leg_kind``, applies the position direction
(``leg_weight``) as exposure, and folds the weighted PVs back to a swap NPV. The
pricing graph itself never knows pay/receive. Everything stays lazy until
``.collect()``.
"""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
from pandera.typing.polars import LazyFrame
from schenberg.domain.schemas import SwapLegInput
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

# --- One swap, booked as two legs: receive CDI (ativo), pay IPCA+coupon (passivo) -
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
                "leg_weight": 1.0,  # receive CDI
                "id_indexador": 1,
                "real_coupon": None,
                **_common,
            },
            {
                "swap_id": "SWP-1",
                "leg_id": "passivo",
                "leg_kind": "IPCA",
                "leg_role": "passivo",
                "leg_weight": -1.0,  # pay IPCA+coupon
                "id_indexador": 2,
                "real_coupon": 0.02,
                **_common,
            },
        ]
    ).lazy(),
)

result = price_swap(legs, market).collect()
print(result)
# npv = ativo_pv (receive CDI) + passivo_pv (pay IPCA) — both already signed.

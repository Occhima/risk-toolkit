from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.swap import swap_leg_router


def test_swap_leg_router_routes_known_legs_and_unknown_to_fallback() -> None:
    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": [1, 2, 3, 4],
                        "tenor_days": [252, 252, 252, 252],
                        "zero_rate": [0.1, 0.1, 0.05, 0.1],
                        "forward_rate": [None, 0.12, None, None],
                    }
                ).lazy(),
            ),
            MarketSource(
                "fixings",
                pl.DataFrame(
                    {
                        "id_indexador": [3],
                        "fixing_date": [date(2026, 6, 3)],
                        "fixing_value": [100.0],
                    }
                ).lazy(),
            ),
            MarketSource(
                "projected_indexes",
                pl.DataFrame(
                    {"id_indexador": [3], "tenor_days": [252], "projected_index": [106.0]}
                ).lazy(),
            ),
        ],
    )
    legs = pl.DataFrame(
        {
            "swap_id": ["S", "S", "S", "S"],
            "leg_id": ["fixed", "cdi", "ipca", "unknown"],
            "leg_kind": ["FIXED", "CDI", "IPCA", "UNKNOWN"],
            "leg_role": ["fixed", "ativo", "passivo", "other"],
            "leg_weight": [1.0, 1.0, -1.0, 1.0],
            "notional": [1_000_000.0] * 4,
            "id_indexador": [1, 2, 3, 4],
            "payment_days": [252] * 4,
            "accrual": [1.0] * 4,
            "base_date": [date(2026, 6, 3)] * 4,
            "fixed_rate": [0.08, None, None, None],
            "real_coupon": [None, None, 0.02, None],
            "cashflow_amount": [None, None, None, 42.0],
        }
    ).lazy()

    out = cast(
        pl.DataFrame,
        swap_leg_router.compute(legs, market=market, view="output").collect(),
    )

    assert set(out["leg_kind"].to_list()) == {"FIXED", "CDI", "IPCA", "UNKNOWN"}
    expected_unknown_cashflow = 42.0
    assert (
        out.filter(pl.col("leg_kind") == "UNKNOWN").select("cashflow_amount").item()
        == expected_unknown_cashflow
    )

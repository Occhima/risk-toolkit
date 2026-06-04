from __future__ import annotations

import math
from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.swap.legs.ipca import ipca_swap_leg_graph


def test_ipca_leg_pricing_preserves_formula() -> None:
    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {"id_indexador": [2], "tenor_days": [252], "zero_rate": [0.05]}
                ).lazy(),
            ),
            MarketSource(
                "fixings",
                pl.DataFrame(
                    {
                        "id_indexador": [2],
                        "fixing_date": [date(2026, 6, 3)],
                        "fixing_value": [100.0],
                    }
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
    leg = pl.DataFrame(
        {
            "swap_id": ["S"],
            "leg_id": ["ipca"],
            "leg_kind": ["IPCA"],
            "leg_role": ["passivo"],
            "leg_weight": [-1.0],
            "notional": [1_000_000.0],
            "id_indexador": [2],
            "payment_days": [252],
            "accrual": [1.0],
            "base_date": [date(2026, 6, 3)],
            "fixed_rate": [None],
            "real_coupon": [0.02],
            "cashflow_amount": [None],
        }
    ).lazy()

    out = cast(
        pl.DataFrame,
        ipca_swap_leg_graph.compute(leg, market=market, view="pricing").collect(),
    )

    cashflow = 1_000_000.0 * 1.06 * 1.02 - 1_000_000.0
    assert out.select("cashflow_amount").item() == pytest.approx(cashflow)
    # Pure leg pricing carries no direction: pv is the unsigned discounted cashflow.
    # The pay/receive sign (leg_weight) is applied at the swap Structure layer.
    assert out.select("pv").item() == pytest.approx(cashflow * math.exp(-0.05))

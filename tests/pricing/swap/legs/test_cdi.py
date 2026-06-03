from __future__ import annotations

import math
from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.swap.legs.cdi import cdi_swap_leg_graph


def test_cdi_leg_pricing_preserves_formula() -> None:
    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": [1],
                        "tenor_days": [252],
                        "zero_rate": [0.1],
                        "forward_rate": [0.12],
                    }
                ).lazy(),
            )
        ],
    )
    leg = pl.DataFrame(
        {
            "swap_id": ["S"],
            "leg_id": ["cdi"],
            "leg_kind": ["CDI"],
            "pay_receive": ["RECEIVE"],
            "notional": [1_000_000.0],
            "id_indexador": [1],
            "payment_days": [252],
            "accrual": [1.0],
            "base_date": [date(2026, 6, 3)],
            "fixed_rate": [None],
            "real_coupon": [None],
            "cashflow_amount": [None],
        }
    ).lazy()

    out = cast(
        pl.DataFrame,
        cdi_swap_leg_graph.compute_for(leg, market=market, output_profile="pricing").collect(),
    )

    assert out.select("cashflow_amount").item() == pytest.approx(120_000.0)
    assert out.select("pv").item() == pytest.approx(120_000.0 * math.exp(-0.1))

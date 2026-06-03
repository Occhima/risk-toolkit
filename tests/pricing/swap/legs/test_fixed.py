from __future__ import annotations

import math
from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.swap.legs.fixed import fixed_swap_leg_graph


def test_fixed_leg_pricing() -> None:
    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame({"id_indexador": [1], "tenor_days": [252], "zero_rate": [0.1]}).lazy(),
            )
        ],
    )
    leg = pl.DataFrame(
        {
            "swap_id": ["S"],
            "leg_id": ["fixed"],
            "leg_kind": ["FIXED"],
            "pay_receive": ["RECEIVE"],
            "notional": [1_000_000.0],
            "id_indexador": [1],
            "payment_days": [252],
            "accrual": [1.0],
            "base_date": [date(2026, 6, 3)],
            "fixed_rate": [0.08],
            "real_coupon": [None],
            "cashflow_amount": [None],
        }
    ).lazy()

    out = cast(
        pl.DataFrame,
        fixed_swap_leg_graph.compute(leg, market=market, view="pricing").collect(),
    )

    assert out.select("cashflow_amount").item() == pytest.approx(80_000.0)
    assert out.select("pv").item() == pytest.approx(80_000.0 * math.exp(-0.1))

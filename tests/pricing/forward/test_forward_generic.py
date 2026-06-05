from __future__ import annotations

import math
from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.domain.schemas.forward import ForwardPricing
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.forward.generic import base_forward_graph


def test_base_forward_graph_defines_generic_payoff() -> None:
    assert base_forward_graph.has_view("output")
    assert base_forward_graph.view_schema("output") is ForwardPricing


def test_base_forward_graph_prices_generic_forward_spread() -> None:
    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "di_curve",
                pl.DataFrame(
                    {
                        "curve_name": ["DI"],
                        "id_indexador": [1],
                        "tenor_days": [252],
                        "zero_rate": [0.1],
                    }
                ).lazy(),
            ),
            MarketSource("fx_rates", pl.DataFrame({"currency": ["USD"], "fx_rate": [5.0]}).lazy()),
        ],
    )
    forwards = pl.DataFrame(
        {
            "id_indexador": [1],
            "payment_days": [252],
            "currency": ["USD"],
            "forward_price": [120.0],
            "strike": [20.0],
        }
    ).lazy()

    out = cast(
        pl.DataFrame,
        base_forward_graph.compute(forwards, market=market, view="output").collect(),
    )

    expected_fv = 100.0
    expected_pv = expected_fv * math.exp(-0.1)
    assert out.select("future_value").item() == pytest.approx(expected_fv)
    assert out.select("present_value").item() == pytest.approx(expected_pv)
    assert out.select("value").item() == pytest.approx(expected_pv * 5.0)

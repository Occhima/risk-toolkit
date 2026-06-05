from __future__ import annotations

import math
from datetime import date

import polars as pl
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.instruments.derivatives.forwards import price_forward


def _market() -> MarketSnapshot:
    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 5),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame(
                    {
                        "id_indexador": ["IDX"],
                        "tenor_days": [252],
                        "forward_rate": [110.0],
                        "risk_free_rate": [0.10],
                    }
                ).lazy(),
            ),
            MarketSource(
                "fx_rates",
                pl.DataFrame(
                    {
                        "currency": ["BRL"],
                        "fx_rate": [1.0],
                    }
                ).lazy(),
            ),
        ],
    )


def test_price_forward_is_lazy_and_computes_expected_values() -> None:
    trades = pl.DataFrame(
        {
            "instrument_id": ["FWD-1"],
            "tenor": [date(2027, 6, 5)],
            "indexer": ["IDX"],
            "currency": ["BRL"],
            "strike": [100.0],
            "payment_days": [252],
        }
    ).lazy()

    result = price_forward(trades, _market())
    assert isinstance(result, pl.LazyFrame)

    got = result.collect()

    expected_future_value = 10.0
    expected_discount_factor = math.exp(-0.10 * 1.0)
    expected_present_value = expected_future_value * expected_discount_factor

    assert got["future_value"][0] == expected_future_value
    assert abs(got["present_value"][0] - expected_present_value) < 1e-12
    assert abs(got["value"][0] - expected_present_value) < 1e-12


def test_price_forward_returns_lazy_frame() -> None:
    trades = pl.DataFrame(
        {
            "instrument_id": ["FWD-1"],
            "tenor": [date(2027, 6, 5)],
            "indexer": ["IDX"],
            "currency": ["BRL"],
            "strike": [100.0],
            "payment_days": [252],
        }
    ).lazy()

    result = price_forward(trades, _market())
    assert isinstance(result, pl.LazyFrame)

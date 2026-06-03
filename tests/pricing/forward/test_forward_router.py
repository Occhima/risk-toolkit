from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest
from schenberg.pricing.api import compute_forward_pricing_rows, price_forwards


def test_forward_router_can_compute_raw_pricing_rows(energy_inputs, energy_market) -> None:
    generic = pl.DataFrame(
        {
            "instrument_id": ["GEN-1"],
            "instrument_type": ["FORWARD"],
            "forward_family": ["GENERIC"],
            "settlement_type": ["CASH_SETTLED"],
            "currency": ["BRL"],
            "id_indexador": [1],
            "payment_days": [30],
            "forward_price": [200.0],
            "strike": [100.0],
            "submarket": [None],
            "delivery_period": [None],
        }
    ).lazy()
    rows = pl.concat([energy_inputs, generic], how="diagonal_relaxed")

    out = cast(pl.DataFrame, compute_forward_pricing_rows(rows, energy_market).collect())

    expected_rows = 3
    assert out.height == expected_rows
    assert out.select(pl.col("future_value").is_not_null().all()).item()


def test_price_forwards_returns_instrument_price_table(energy_inputs, energy_market) -> None:
    generic = pl.DataFrame(
        {
            "instrument_id": ["GEN-1"],
            "instrument_type": ["FORWARD"],
            "forward_family": ["GENERIC"],
            "settlement_type": ["CASH_SETTLED"],
            "currency": ["BRL"],
            "id_indexador": [1],
            "payment_days": [30],
            "forward_price": [200.0],
            "strike": [100.0],
            "submarket": [None],
            "delivery_period": [None],
        }
    ).lazy()
    rows = pl.concat([energy_inputs, generic], how="diagonal_relaxed")

    out = cast(pl.DataFrame, price_forwards(rows, energy_market).collect())

    assert out.columns == ["instrument_type", "instrument_id", "price"]
    expected_prices = 2
    assert out.height == expected_prices
    expected_generic = 100.0 * math.exp(-0.10 * 30.0 / 252.0)
    assert out.filter(pl.col("instrument_id") == "GEN-1").select("price").item() == pytest.approx(
        expected_generic
    )

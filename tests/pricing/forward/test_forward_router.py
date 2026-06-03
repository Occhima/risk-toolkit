from __future__ import annotations

from typing import cast

import polars as pl
from schenberg.pricing.api import price_forwards


def test_forward_router_routes_energy_and_generic_rows(energy_inputs, energy_market) -> None:
    generic = pl.DataFrame(
        {
            "trade_id": ["GEN-1"],
            "instrument_type": ["FORWARD"],
            "forward_family": ["GENERIC"],
            "settlement_type": ["CASH_SETTLED"],
            "currency": ["BRL"],
            "id_indexador": [1],
            "payment_days": [30],
            "future_value": [100.0],
            "contract_id": [None],
            "submarket": [None],
            "delivery_period": [None],
            "buy_sell": [None],
            "quantity": [None],
            "strike": [None],
        }
    ).lazy()
    rows = pl.concat([energy_inputs, generic], how="diagonal_relaxed")

    out = cast(pl.DataFrame, price_forwards(rows, energy_market).collect())

    expected_rows = 3
    assert out.height == expected_rows
    assert out.select(pl.col("future_value").is_not_null().all()).item()

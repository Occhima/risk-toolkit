from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest
from schenberg.pricing.instruments.forward import energy
from schenberg.pricing.instruments.forward.energy import energy_forward_graph, price_energy_forward


def test_price_energy_forward_prices_normalized_rows_without_quantity(
    energy_inputs, energy_market
) -> None:
    result = cast(pl.DataFrame, price_energy_forward(energy_inputs, energy_market).collect())

    # payment_days are derived from the ANBIMA settlement dates vs as_of 2026-06-03:
    #   2026-07 settles 2026-08-10 -> 47 business days
    #   2026-08 settles 2026-09-09 -> 68 business days
    expected = (120.0 - 100.0) * math.exp(-0.10 * 47.0 / 252.0)
    expected += (130.0 - 100.0) * math.exp(-0.10 * 68.0 / 252.0)

    assert result.select("instrument_id").item() == "ENG-1"
    assert result.select("instrument_type").item() == "FORWARD"
    assert result.select("price").item() == pytest.approx(expected)
    assert not set(result.columns) & {"mtm_local", "mtm", "quantity", "buy_sell"}


def test_energy_forward_graph_is_only_generic_graph_with_market_bindings() -> None:
    assert not hasattr(energy, "energy_cashflow_graph")
    assert not hasattr(energy, "pay_receive")
    assert not hasattr(energy, "future_value")
    assert energy_forward_graph.output_dtypes("pricing").keys() == {
        "future_value",
        "present_value",
        "value",
    }
    assert energy_forward_graph.dependencies_of("future_value") == {"forward_price", "strike"}

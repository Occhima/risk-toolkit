from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest
from schenberg.domain.schemas.forward import ForwardPricing
from schenberg.pricing.instruments.forward import energy
from schenberg.pricing.instruments.forward.energy import energy_forward_graph, price_energy_forward


def test_price_energy_forward_prices_normalized_rows_without_quantity(
    energy_inputs, energy_market
) -> None:
    result = cast(pl.DataFrame, price_energy_forward(energy_inputs, energy_market).collect())

    expected = (120.0 - 100.0) * math.exp(-0.10 * 30.0 / 252.0)
    expected += (130.0 - 100.0) * math.exp(-0.10 * 60.0 / 252.0)

    assert result.select("instrument_id").item() == "ENG-1"
    assert result.select("instrument_type").item() == "FORWARD"
    assert result.select("price").item() == pytest.approx(expected)
    assert not set(result.columns) & {"mtm_local", "mtm", "quantity", "buy_sell"}


def test_energy_forward_graph_is_only_generic_graph_with_market_bindings() -> None:
    assert not hasattr(energy, "energy_cashflow_graph")
    assert not hasattr(energy, "pay_receive")
    assert not hasattr(energy, "future_value")
    # The energy graph reuses the generic forward output contract; only where
    # forward_price comes from differs (a market read, not an input column).
    assert energy_forward_graph.has_view("output")
    assert energy_forward_graph.view_schema("output") is ForwardPricing
    assert "forward_price" in energy.EnergyForwardRequirements.__requirements__

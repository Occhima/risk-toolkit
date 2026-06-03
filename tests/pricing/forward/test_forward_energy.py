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

    expected = (120.0 - 100.0) * math.exp(-0.10 * 30.0 / 252.0)
    expected += (130.0 - 100.0) * math.exp(-0.10 * 60.0 / 252.0)

    assert result.select("instrument_id").item() == "ENG-1"
    assert result.select("mtm_local").item() == pytest.approx(expected)
    assert result.select("mtm").item() == pytest.approx(expected)


def test_energy_core_pricer_has_no_explode_delivery_helper_or_quantity_dependency() -> None:
    assert not hasattr(energy, "explode_delivery")
    assert "quantity" not in energy_cashflow_dependencies()
    assert energy_forward_graph.output_dtypes("pricing").keys() == {
        "future_value",
        "present_value",
        "value",
    }


def energy_cashflow_dependencies() -> set[str]:
    return energy.energy_cashflow_graph.dependencies_of("future_value")

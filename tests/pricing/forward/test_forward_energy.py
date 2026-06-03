from __future__ import annotations

import math
from typing import cast

import polars as pl
import pytest
from schenberg.pricing.instruments.forward import energy
from schenberg.pricing.instruments.forward.energy import energy_forward_graph, price_energy_forward


def test_price_energy_forward_prices_normalized_rows(energy_inputs, energy_market) -> None:
    result = cast(pl.DataFrame, price_energy_forward(energy_inputs, energy_market).collect())

    expected = 10.0 * (120.0 - 100.0) * math.exp(-0.10 * 30.0 / 252.0)
    expected += 10.0 * (130.0 - 100.0) * math.exp(-0.10 * 60.0 / 252.0)

    assert result.select("contract_id").item() == "ENG-1"
    assert result.select("mtm_local").item() == pytest.approx(expected)
    assert result.select("mtm").item() == pytest.approx(expected)


def test_energy_core_pricer_has_no_explode_delivery_helper() -> None:
    assert not hasattr(energy, "explode_delivery")
    assert energy_forward_graph.output_dtypes("pricing").keys() == {
        "future_value",
        "present_value",
        "value",
    }

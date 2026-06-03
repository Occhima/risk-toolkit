from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from schenberg.pricing.api import price_energy_forward, price_swap
from schenberg.pricing.instruments.forward.energy import energy_forward_graph


def test_price_swap_returns_aggregated_npv(swap_legs, swap_market) -> None:
    result = cast(pl.DataFrame, price_swap(swap_legs, swap_market).collect())

    assert result.select("swap_id").item() == "SWP-1"
    assert result.select("ativo_pv").item() == pytest.approx(108_580.490164, rel=1e-6)
    assert result.select("passivo_pv").item() == pytest.approx(-77_239.829269, rel=1e-6)
    assert result.select("npv").item() == pytest.approx(31_340.660895, rel=1e-6)


def test_price_energy_forward_uses_generic_forward_outputs(energy_inputs, energy_market) -> None:
    result = cast(pl.DataFrame, price_energy_forward(energy_inputs, energy_market).collect())

    assert result.select("instrument_id").item() == "ENG-1"
    assert result.select("instrument_type").item() == "FORWARD"
    assert result.select("price").item() == pytest.approx(49.057467, rel=1e-6)
    assert not set(result.columns) & {"quantity", "mtm", "mtm_local"}
    assert energy_forward_graph.output_dtypes("pricing").keys() == {
        "future_value",
        "present_value",
        "value",
    }

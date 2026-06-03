from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from schenberg.domain import schemas
from schenberg.domain.schemas.position import InstrumentPrice
from schenberg.pricing.instruments.forward.energy import price_energy_forward


def test_price_energy_forward_returns_instrument_price_not_output(
    energy_inputs, energy_market
) -> None:
    prices = price_energy_forward(energy_inputs, energy_market)

    result = cast(pl.DataFrame, InstrumentPrice.validate(prices, lazy=True).collect())

    assert result.columns == ["instrument_type", "instrument_id", "price"]
    assert result.select("instrument_id").item() == "ENG-1"
    assert result.select("instrument_type").item() == "FORWARD"
    assert result.select("price").item() == pytest.approx(49.057467, rel=1e-6)
    assert not set(result.columns) & {"quantity", "mtm", "mtm_local"}

    assert "EnergyForwardOutput" not in schemas.__all__

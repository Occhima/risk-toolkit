from __future__ import annotations

from collections.abc import Callable
from typing import cast

import polars as pl
import pytest
from pandera.typing.polars import LazyFrame
from schenberg.domain.schemas.forward import ForwardTrade
from schenberg.domain.schemas.position import InstrumentPrice, Position, PricedPosition
from schenberg.position.functions import (
    pnl_from_priced_positions,
    price_forward_instruments,
    with_prices,
)

EXPECTED_PRICED_COLUMNS = [
    "position_id",
    "book",
    "instrument_type",
    "instrument_id",
    "quantity",
    "side",
    "price",
    "mtm",
]


def test_with_prices_direct_pipe_and_schema() -> None:
    positions = Position.from_records(
        [
            {
                "position_id": "POS-1",
                "book": "Energy Desk",
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "quantity": 100.0,
                "side": 1.0,
            }
        ]
    )
    prices = InstrumentPrice.from_records(
        [{"instrument_type": "FORWARD", "instrument_id": "ENG-1", "price": 49.057467}]
    )

    direct = cast(pl.DataFrame, with_prices(positions, prices).collect())
    piped = cast(
        pl.DataFrame,
        positions.pipe(cast(Callable[..., pl.LazyFrame], with_prices), prices).collect(),
    )

    assert direct.columns == EXPECTED_PRICED_COLUMNS
    assert piped.columns == EXPECTED_PRICED_COLUMNS
    assert direct.select("mtm").item() == pytest.approx(4905.7467)
    assert piped.select("mtm").item() == pytest.approx(4905.7467)
    validated = cast(pl.DataFrame, PricedPosition.validate(direct.lazy(), lazy=True).collect())

    assert validated.height == 1


def test_pnl_from_priced_positions_direct_and_pipe() -> None:
    today = PricedPosition.from_records(
        [
            {
                "position_id": "P",
                "book": "B",
                "instrument_type": "FORWARD",
                "instrument_id": "I",
                "quantity": 2.0,
                "side": 1.0,
                "price": 6.0,
                "mtm": 12.0,
            }
        ]
    )
    previous = PricedPosition.from_records(
        [
            {
                "position_id": "P",
                "book": "B",
                "instrument_type": "FORWARD",
                "instrument_id": "I",
                "quantity": 2.0,
                "side": 1.0,
                "price": 5.0,
                "mtm": 10.0,
            }
        ]
    )

    direct = cast(pl.DataFrame, pnl_from_priced_positions(today, previous).collect())
    piped = cast(
        pl.DataFrame,
        today.pipe(
            cast(Callable[..., pl.LazyFrame], pnl_from_priced_positions), previous
        ).collect(),
    )

    assert direct.select("price_pnl").item() == pytest.approx(2.0)
    assert direct.select("mtm_pnl").item() == pytest.approx(2.0)
    assert piped.select("price_pnl").item() == pytest.approx(2.0)
    assert piped.select("mtm_pnl").item() == pytest.approx(2.0)


def test_price_forward_instruments_returns_instrument_price(energy_inputs, energy_market) -> None:
    prices = price_forward_instruments(cast(LazyFrame[ForwardTrade], energy_inputs), energy_market)

    result = cast(pl.DataFrame, InstrumentPrice.validate(prices, lazy=True).collect())

    assert result.columns == ["instrument_type", "instrument_id", "price"]
    assert result.select("instrument_type").item() == "FORWARD"
    assert result.select("instrument_id").item() == "ENG-1"
    assert result.select("price").item() == pytest.approx(48.831742, rel=1e-6)

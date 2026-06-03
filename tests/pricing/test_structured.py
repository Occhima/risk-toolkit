from __future__ import annotations

from collections.abc import Callable
from typing import cast

import polars as pl
import pytest
from schenberg.domain.schemas.position import InstrumentPrice, Position
from schenberg.domain.schemas.structure import StructureLeg
from schenberg.position.functions import with_prices
from schenberg.pricing.structured import price_structures


def _component_prices() -> pl.LazyFrame:
    return InstrumentPrice.from_records(
        [
            {"instrument_type": "FORWARD", "instrument_id": "ENG-1", "price": 10.0},
            {"instrument_type": "SWAP", "instrument_id": "SWP-1", "price": 5.0},
        ]
    )


def _structure_legs() -> pl.LazyFrame:
    return StructureLeg.from_records(
        [
            {
                "structure_id": "PREPAY-001",
                "leg_id": "L1",
                "component_instrument_type": "FORWARD",
                "component_instrument_id": "ENG-1",
                "quantity": 2.0,
                "side": 1.0,
            },
            {
                "structure_id": "PREPAY-001",
                "leg_id": "L2",
                "component_instrument_type": "SWAP",
                "component_instrument_id": "SWP-1",
                "quantity": 3.0,
                "side": -1.0,
            },
        ]
    )


def test_structure_price_is_weighted_sum() -> None:
    result = cast(pl.DataFrame, price_structures(_structure_legs(), _component_prices()).collect())

    assert result.select("price").item() == pytest.approx(2 * 10.0 + (-1) * 3 * 5.0)


def test_structure_output_columns() -> None:
    result = cast(pl.DataFrame, price_structures(_structure_legs(), _component_prices()).collect())

    assert result.columns == ["instrument_type", "instrument_id", "price"]


def test_structure_instrument_type_and_id() -> None:
    result = cast(pl.DataFrame, price_structures(_structure_legs(), _component_prices()).collect())

    assert result.select("instrument_type").item() == "STRUCTURE"
    assert result.select("instrument_id").item() == "PREPAY-001"


def test_custom_structure_type() -> None:
    result = cast(
        pl.DataFrame,
        price_structures(_structure_legs(), _component_prices(), structure_type="PREPAY").collect(),
    )

    assert result.select("instrument_type").item() == "PREPAY"


def test_result_is_lazy_before_collect() -> None:
    result_lf = price_structures(_structure_legs(), _component_prices())
    assert isinstance(result_lf, pl.LazyFrame)


def test_structure_price_passes_through_with_prices() -> None:
    structure_prices = price_structures(_structure_legs(), _component_prices())

    positions = Position.from_records(
        [
            {
                "position_id": "POS-S1",
                "book": "Desk A",
                "instrument_type": "STRUCTURE",
                "instrument_id": "PREPAY-001",
                "quantity": 100.0,
                "side": 1.0,
            }
        ]
    )

    priced = cast(
        pl.DataFrame,
        positions.pipe(cast(Callable[..., pl.LazyFrame], with_prices), structure_prices).collect(),
    )

    expected_price = 2 * 10.0 + (-1) * 3 * 5.0
    assert priced.select("price").item() == pytest.approx(expected_price)
    assert priced.select("mtm").item() == pytest.approx(100.0 * expected_price)


def test_multi_structure_pricing() -> None:
    legs = StructureLeg.from_records(
        [
            {
                "structure_id": "S1",
                "leg_id": "L1",
                "component_instrument_type": "FORWARD",
                "component_instrument_id": "ENG-1",
                "quantity": 1.0,
                "side": 1.0,
            },
            {
                "structure_id": "S2",
                "leg_id": "L2",
                "component_instrument_type": "SWAP",
                "component_instrument_id": "SWP-1",
                "quantity": 2.0,
                "side": 1.0,
            },
        ]
    )

    result = cast(pl.DataFrame, price_structures(legs, _component_prices()).collect())

    assert result.height == len(["S1", "S2"])
    prices = dict(zip(result["instrument_id"].to_list(), result["price"].to_list(), strict=True))
    assert prices["S1"] == pytest.approx(10.0)
    assert prices["S2"] == pytest.approx(10.0)


def test_concat_atomic_and_structure_prices_for_with_prices() -> None:
    """Demonstrate the full composition pattern: atomic + structure → with_prices."""
    atomic_prices = _component_prices()
    structure_prices = price_structures(_structure_legs(), atomic_prices)
    all_prices = pl.concat([atomic_prices, structure_prices], how="diagonal_relaxed")

    positions = Position.from_records(
        [
            {
                "position_id": "POS-1",
                "book": "B",
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "quantity": 1.0,
                "side": 1.0,
            },
            {
                "position_id": "POS-2",
                "book": "B",
                "instrument_type": "STRUCTURE",
                "instrument_id": "PREPAY-001",
                "quantity": 10.0,
                "side": 1.0,
            },
        ]
    )

    priced = cast(
        pl.DataFrame,
        positions.pipe(cast(Callable[..., pl.LazyFrame], with_prices), all_prices).collect(),
    )

    assert priced.height == len(["POS-1", "POS-2"])
    mtm_by_pos = dict(zip(priced["position_id"].to_list(), priced["mtm"].to_list(), strict=True))
    assert mtm_by_pos["POS-1"] == pytest.approx(10.0)
    assert mtm_by_pos["POS-2"] == pytest.approx(10.0 * 5.0)

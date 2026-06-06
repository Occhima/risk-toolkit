from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from pandera.typing.polars import LazyFrame
from schenberg.domain.schemas.position import (
    BookContract,
    InstrumentValue,
    Position,
    ReportingFx,
)
from schenberg.domain.schemas.structure import StructureLeg
from schenberg.position import position_value
from schenberg.pricing.structured import price_structures


def _book(name: str = "B") -> LazyFrame[BookContract]:
    return cast(
        LazyFrame[BookContract],
        BookContract.from_records(
            [{"book": name, "desk": "D", "legal_entity": "LE", "reporting_currency": "BRL"}]
        ),
    )


def _fx() -> LazyFrame[ReportingFx]:
    return cast(
        LazyFrame[ReportingFx],
        ReportingFx.from_records(
            [{"currency": "BRL", "reporting_currency": "BRL", "book_fx": 1.0}]
        ),
    )


def _component_values() -> LazyFrame[InstrumentValue]:
    return cast(
        LazyFrame[InstrumentValue],
        InstrumentValue.from_records(
            [
                {
                    "instrument_type": "FORWARD",
                    "instrument_id": "ENG-1",
                    "value": 10.0,
                    "currency": "BRL",
                },  # noqa: E501
                {
                    "instrument_type": "SWAP",
                    "instrument_id": "SWP-1",
                    "value": 5.0,
                    "currency": "BRL",
                },  # noqa: E501
            ]
        ),
    )


def _structure_legs() -> LazyFrame[StructureLeg]:
    return cast(
        LazyFrame[StructureLeg],
        StructureLeg.from_records(
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
        ),
    )


def test_structure_value_is_weighted_sum() -> None:
    result = cast(pl.DataFrame, price_structures(_structure_legs(), _component_values()).collect())

    assert result.select("value").item() == pytest.approx(2 * 10.0 + (-1) * 3 * 5.0)


def test_structure_output_is_an_instrument_value() -> None:
    result = cast(pl.DataFrame, price_structures(_structure_legs(), _component_values()).collect())

    assert result.columns == ["instrument_type", "instrument_id", "value", "currency"]
    assert result.select("currency").item() == "BRL"  # carried from the components


def test_structure_instrument_type_and_id() -> None:
    result = cast(pl.DataFrame, price_structures(_structure_legs(), _component_values()).collect())

    assert result.select("instrument_type").item() == "STRUCTURE"
    assert result.select("instrument_id").item() == "PREPAY-001"


def test_custom_structure_type() -> None:
    result = cast(
        pl.DataFrame,
        price_structures(_structure_legs(), _component_values(), structure_type="PREPAY").collect(),
    )

    assert result.select("instrument_type").item() == "PREPAY"


def test_result_is_lazy_before_collect() -> None:
    result_lf = price_structures(_structure_legs(), _component_values())
    assert isinstance(result_lf, pl.LazyFrame)


def test_structure_value_passes_through_position_value() -> None:
    structure_values = price_structures(_structure_legs(), _component_values())

    positions = Position.from_records(
        [
            {
                "position_id": "POS-S1",
                "book": "B",
                "instrument_type": "STRUCTURE",
                "instrument_id": "PREPAY-001",
                "quantity": 100.0,
                "side": 1.0,
                "unit_notional": None,
            }
        ]
    )

    valued = cast(
        pl.DataFrame,
        position_value(positions, value=structure_values, book=_book(), fx=_fx()).collect(),
    )

    expected_value = 2 * 10.0 + (-1) * 3 * 5.0
    assert valued.select("mtm").item() == pytest.approx(100.0 * expected_value)
    assert valued.select("reported_mtm").item() == pytest.approx(100.0 * expected_value)


def test_multi_structure_pricing() -> None:
    legs = cast(
        LazyFrame[StructureLeg],
        StructureLeg.from_records(
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
        ),
    )

    result = cast(pl.DataFrame, price_structures(legs, _component_values()).collect())

    assert result.height == len(["S1", "S2"])
    values = dict(zip(result["instrument_id"].to_list(), result["value"].to_list(), strict=True))
    assert values["S1"] == pytest.approx(10.0)
    assert values["S2"] == pytest.approx(10.0)


def test_concat_atomic_and_structure_values_for_position_value() -> None:
    """The full composition: atomic + structure InstrumentValue → position_value."""
    atomic_values: LazyFrame[InstrumentValue] = _component_values()
    structure_values = price_structures(_structure_legs(), atomic_values)
    all_values = pl.concat([atomic_values, structure_values], how="diagonal_relaxed")

    positions = Position.from_records(
        [
            {
                "position_id": "POS-1",
                "book": "B",
                "instrument_type": "FORWARD",
                "instrument_id": "ENG-1",
                "quantity": 1.0,
                "side": 1.0,
                "unit_notional": None,
            },
            {
                "position_id": "POS-2",
                "book": "B",
                "instrument_type": "STRUCTURE",
                "instrument_id": "PREPAY-001",
                "quantity": 10.0,
                "side": 1.0,
                "unit_notional": None,
            },
        ]
    )

    valued = cast(
        pl.DataFrame,
        position_value(positions, value=all_values, book=_book(), fx=_fx()).collect(),
    )

    assert valued.height == len(["POS-1", "POS-2"])
    mtm_by_pos = dict(zip(valued["position_id"].to_list(), valued["mtm"].to_list(), strict=True))
    assert mtm_by_pos["POS-1"] == pytest.approx(10.0)
    assert mtm_by_pos["POS-2"] == pytest.approx(10.0 * 5.0)

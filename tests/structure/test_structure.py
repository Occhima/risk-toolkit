from __future__ import annotations

import polars as pl
import pytest
from schenberg.structure import structure_value


def test_structure_value_sums_legs_and_negative_reduces() -> None:
    legs = pl.DataFrame(
        {
            "structure_id": ["S1", "S1"],
            "leg_id": ["L1", "L2"],
            "instrument_type": ["OPTION", "OPTION"],
            "instrument_id": ["A", "B"],
            "quantity": [1.0, -1.0],
            "weight": [1.0, 1.0],
        }
    ).lazy()
    values = pl.DataFrame(
        {
            "instrument_type": ["OPTION", "OPTION"],
            "instrument_id": ["A", "B"],
            "value": [10.0, 3.0],
            "currency": ["BRL", "BRL"],
        }
    ).lazy()
    out = structure_value(legs, values)
    assert isinstance(out, pl.LazyFrame)
    assert out.collect()["value"][0] == pytest.approx(7.0)


def test_structure_missing_value_propagates_null() -> None:
    legs = pl.DataFrame(
        {
            "structure_id": ["S1"],
            "leg_id": ["L1"],
            "instrument_type": ["OPTION"],
            "instrument_id": ["MISSING"],
            "quantity": [1.0],
            "weight": [1.0],
        }
    ).lazy()
    values = pl.DataFrame(
        {
            "instrument_type": ["OPTION"],
            "instrument_id": ["A"],
            "value": [10.0],
            "currency": ["BRL"],
        }
    ).lazy()
    assert structure_value(legs, values).collect()["value"][0] is None

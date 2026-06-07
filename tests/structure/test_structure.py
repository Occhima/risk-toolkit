from __future__ import annotations

import polars as pl
import pytest
from schenberg.structure import structure_stage, structure_value, structure_value_fold


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


def test_structure_value_matches_weighted_sum() -> None:
    legs = pl.DataFrame(
        {
            "structure_id": ["S1", "S1", "S2"],
            "leg_id": ["L1", "L2", "L3"],
            "instrument_type": ["OPTION", "OPTION", "BOND"],
            "instrument_id": ["A", "B", "C"],
            "quantity": [2.0, 1.0, 3.0],
            "weight": [0.5, -1.0, 2.0],
        }
    ).lazy()
    values = pl.DataFrame(
        {
            "instrument_type": ["OPTION", "OPTION", "BOND"],
            "instrument_id": ["A", "B", "C"],
            "value": [10.0, 3.0, 4.0],
            "currency": ["BRL", "BRL", "BRL"],
        }
    ).lazy()

    out = structure_value(legs, values).sort("structure_id").collect()

    assert out["value"].to_list() == pytest.approx([7.0, 24.0])


def test_structure_value_null_when_any_leg_missing_price() -> None:
    legs = pl.DataFrame(
        {
            "structure_id": ["S1", "S1"],
            "leg_id": ["L1", "L2"],
            "instrument_type": ["OPTION", "OPTION"],
            "instrument_id": ["A", "MISSING"],
            "quantity": [1.0, 1.0],
            "weight": [1.0, 1.0],
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


def test_structure_stage_exposes_leg_value_for_debug() -> None:
    legs = pl.DataFrame(
        {
            "structure_id": ["S1"],
            "leg_id": ["L1"],
            "instrument_type": ["OPTION"],
            "instrument_id": ["A"],
            "quantity": [2.0],
            "weight": [0.5],
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

    out = structure_stage(legs, values).collect()

    assert "leg_value" in out.columns
    assert out["leg_value"][0] == pytest.approx(10.0)


def test_structure_value_fold_is_inspectable() -> None:
    assert "Fold structure_value" in structure_value_fold.explain()
    assert "flowchart" in structure_value_fold.to_mermaid()

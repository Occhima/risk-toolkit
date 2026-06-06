from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from schenberg.core.columns import cols
from schenberg.core.fold import Fold, count_, lit_, sum_
from schenberg.domain.base import SchenbergDataFrameModel


class _Priced(SchenbergDataFrameModel):
    """A tiny local output schema for exercising Fold's schema handling."""

    instrument_type: str
    instrument_id: str
    price: float


def _component_rows() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["A", "A", "B"],
            "value": [1.0, 2.0, 5.0],
            "weight": [1.0, -1.0, 2.0],
            "role": ["x", "y", "x"],
        }
    ).lazy()


def test_group_by_sum_aggregation() -> None:
    fold = Fold("v").by("instrument_id").returns(None, total=sum_("value"))
    out = cast(pl.DataFrame, fold.compute(_component_rows()).collect()).sort("instrument_id")

    assert dict(zip(out["instrument_id"], out["total"], strict=True)) == {"A": 3.0, "B": 5.0}


def test_weighted_sum() -> None:
    fold = Fold("v").by("instrument_id").returns(None, w=sum_("value", weight="weight"))
    out = cast(pl.DataFrame, fold.compute(_component_rows()).collect()).sort("instrument_id")

    # A: 1*1 + 2*-1 = -1 ; B: 5*2 = 10
    assert dict(zip(out["instrument_id"], out["w"], strict=True)) == {"A": -1.0, "B": 10.0}


def test_filtered_conditional_sum() -> None:
    fold = (
        Fold("v")
        .by("instrument_id")
        .returns(None, x_only=sum_("value", where=pl.col("role") == "x"))
    )
    out = cast(pl.DataFrame, fold.compute(_component_rows()).collect()).sort("instrument_id")

    # A: only role==x -> 1.0 ; B: role==x -> 5.0
    assert dict(zip(out["instrument_id"], out["x_only"], strict=True)) == {"A": 1.0, "B": 5.0}


def test_count_and_lit() -> None:
    fold = Fold("v").by("instrument_id").returns(None, n=count_(), tag=lit_("T"))
    out = cast(pl.DataFrame, fold.compute(_component_rows()).collect()).sort("instrument_id")

    assert dict(zip(out["instrument_id"], out["n"], strict=True)) == {"A": 2, "B": 1}
    assert set(out["tag"].to_list()) == {"T"}


def test_output_schema_field_order_is_respected() -> None:
    fold = (
        Fold("forward_price")
        .by("instrument_id")
        .returns(_Priced, instrument_type=lit_("FORWARD"), price=sum_("value"))
    )
    out = cast(pl.DataFrame, fold.compute(_component_rows()).collect())

    # _Priced declares instrument_type, instrument_id, price — in that order.
    assert out.columns == ["instrument_type", "instrument_id", "price"]


def test_returns_rejects_columns_not_in_schema() -> None:
    with pytest.raises(ValueError, match="not in schema"):
        Fold("f").by("instrument_id").returns(_Priced, bogus=sum_("value"))


def test_returns_requires_aggregation_for_each_non_key_field() -> None:
    with pytest.raises(ValueError, match="missing aggregations"):
        Fold("f").by("instrument_id").returns(_Priced, price=sum_("value"))


def test_compute_without_keys_raises() -> None:
    with pytest.raises(ValueError, match="no group keys"):
        Fold("f").returns(None, total=sum_("value")).compute(_component_rows())


def test_explain_and_info_and_mermaid() -> None:
    F = cols(_Priced)
    fold = (
        Fold("forward_price")
        .by(F.instrument_id)
        .returns(_Priced, instrument_type=lit_("FORWARD"), price=sum_("value"))
    )

    text = fold.explain()
    assert "Fold forward_price" in text
    assert "instrument_id" in text
    assert "price = sum(value)" in text

    info = fold.info()
    assert info["group_keys"] == ["instrument_id"]
    assert info["schema"] == "_Priced"
    assert cast(dict, info["aggregations"])["price"] == "sum(value)"

    assert "flowchart LR" in fold.to_mermaid()


def test_raw_pl_expr_aggregation_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be an Agg"):
        Fold("f").by("instrument_id").returns(None, total=pl.col("value").sum())

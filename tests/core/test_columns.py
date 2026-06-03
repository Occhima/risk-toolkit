from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from schenberg.core.columns import RoutePredicate, cols
from schenberg.domain.schemas import ForwardTrade


def test_schema_column_ref_name_and_expr() -> None:
    F = cols(ForwardTrade)

    assert F.instrument_type.name == "instrument_type"
    out = cast(
        pl.DataFrame,
        pl.DataFrame({"instrument_type": ["FORWARD"]})
        .lazy()
        .select(F.instrument_type.expr())
        .collect(),
    )
    assert out.item() == "FORWARD"


def test_unknown_schema_column_raises_attribute_error() -> None:
    F = cols(ForwardTrade)

    with pytest.raises(AttributeError, match="not a declared column"):
        _ = F.not_a_column


def test_column_equality_builds_route_predicate() -> None:
    F = cols(ForwardTrade)

    predicate = F.forward_family == "ENERGY"

    assert isinstance(predicate, RoutePredicate)
    assert predicate.column.name == "forward_family"
    assert predicate.op == "=="
    assert predicate.value == "ENERGY"

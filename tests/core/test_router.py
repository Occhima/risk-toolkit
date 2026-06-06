from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from schenberg.core.columns import cols
from schenberg.core.router import Router
from schenberg.domain.schemas import ForwardTrade


class MarkerPricer:
    """A minimal contract-free computation used to exercise plain routing."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def plan(self, frame: pl.LazyFrame, *, view: str = "result") -> pl.LazyFrame:
        return frame.with_columns(pl.lit(self.marker).alias("priced_by"))

    def has_view(self, view: str) -> bool:
        return True

    def view_schema(self, view: str) -> object | None:
        return None


def test_fallback_receives_all_rows_when_no_cases() -> None:
    F = cols(ForwardTrade)
    router = Router.on(F.forward_family).default(MarkerPricer("fallback"))
    lf = pl.DataFrame({"forward_family": ["A", "B"]}).lazy()

    out = cast(pl.DataFrame, router.plan(lf).collect())

    assert out["priced_by"].to_list() == ["fallback", "fallback"]


def test_registered_case_and_fallback_partition_rows() -> None:
    F = cols(ForwardTrade)
    router = Router.on(F.forward_family).default(MarkerPricer("fallback"))

    @router.when(F.forward_family == "ENERGY")
    def energy() -> MarkerPricer:
        return MarkerPricer("energy")

    lf = pl.DataFrame({"forward_family": ["ENERGY", "GENERIC"]}).lazy()

    out = cast(pl.DataFrame, router.plan(lf).collect()).sort("forward_family")

    assert isinstance(energy, MarkerPricer)
    assert out["priced_by"].to_list() == ["energy", "fallback"]


def test_router_without_cases_or_fallback_raises() -> None:
    F = cols(ForwardTrade)
    router = Router.on(F.forward_family)

    with pytest.raises(ValueError, match="no registered cases"):
        router.plan(pl.DataFrame({"forward_family": ["GENERIC"]}).lazy())

from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from schenberg.core.columns import cols
from schenberg.core.router import Router
from schenberg.domain.schemas import ForwardTrade


class MarkerPricer:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def compute_for(
        self,
        lf: pl.LazyFrame,
        *,
        market=None,
        output_profile: str = "pricing",
    ) -> pl.LazyFrame:
        return lf.with_columns(pl.lit(self.marker).alias("priced_by"))


def test_fallback_receives_all_rows_when_no_cases() -> None:
    F = cols(ForwardTrade)
    router = Router.by(F.forward_family).default(MarkerPricer("fallback"))
    lf = pl.DataFrame({"forward_family": ["A", "B"]}).lazy()

    out = cast(pl.DataFrame, router.compute_for(lf).collect())

    assert out["priced_by"].to_list() == ["fallback", "fallback"]


def test_registered_case_and_fallback_partition_rows() -> None:
    F = cols(ForwardTrade)
    router = Router.by(F.forward_family).default(MarkerPricer("fallback"))

    @router.register(F.forward_family == "ENERGY")
    def energy() -> MarkerPricer:
        return MarkerPricer("energy")

    lf = pl.DataFrame({"forward_family": ["ENERGY", "GENERIC"]}).lazy()

    out = cast(pl.DataFrame, router.compute_for(lf).collect()).sort("forward_family")

    assert isinstance(energy, MarkerPricer)
    assert out["priced_by"].to_list() == ["energy", "fallback"]


def test_router_without_cases_or_fallback_raises() -> None:
    F = cols(ForwardTrade)
    router = Router.by(F.forward_family)

    with pytest.raises(ValueError, match="no registered cases"):
        router.compute_for(pl.DataFrame({"forward_family": ["GENERIC"]}).lazy())

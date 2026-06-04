"""Public-API surface: column helpers, market specs, Router sugar, Workflow."""

from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from schenberg.core.columns import ColumnLike, ColumnRef, col_name, cols
from schenberg.core.market import MarketRead, MarketRequirement
from schenberg.core.pipeline import Workflow
from schenberg.core.router import Router
from schenberg.domain.schemas.option import OptionPrice
from schenberg.market_data.curves import CurveSpec
from schenberg.market_data.volatility import VolSurfaceSpec


def test_column_like_helpers() -> None:
    assert col_name("x") == "x"
    assert col_name(ColumnRef("y")) == "y"
    accepts: ColumnLike = ColumnRef("z")
    assert col_name(accepts) == "z"


def test_curve_spec_returns_market_read_when_output_omitted() -> None:
    read = CurveSpec("curves").value("zero_rate", indexer="id_indexador", tenor="payment_days")
    # Omitting output yields a delayed MarketRead; g.market names the column.
    assert isinstance(read, MarketRead)
    assert read.as_output("rate").outputs == {"zero_rate": "rate"}
    # With an explicit output it is the concrete requirement.
    fixed = CurveSpec("curves").value("zero_rate", output="rate")
    assert isinstance(fixed, MarketRequirement)
    assert fixed.outputs == {"zero_rate": "rate"}


def test_vol_surface_spec_returns_market_read_when_output_omitted() -> None:
    OPT = cols(OptionPrice)  # any schema; just exercising ColumnLike
    read = VolSurfaceSpec("vol_surface").implied_vol(
        indexer=ColumnRef("id_indexador"), tenor="payment_days", strike=OPT.price
    )
    assert isinstance(read, MarketRead)
    assert read.as_output("vol").outputs == {"implied_vol": "vol"}


def test_router_on_case_and_when() -> None:
    SCHEMA = cols(OptionPrice)

    class Marker:
        def __init__(self, marker: str) -> None:
            self.marker = marker

        def compute(self, frame, *, market=None, view="result"):
            return frame.with_columns(pl.lit(self.marker).alias("by"))

        def has_view(self, view: str) -> bool:
            return True

        def view_schema(self, view: str) -> object | None:
            return None

    router = Router.on(SCHEMA.instrument_type)

    @router.case("A")
    def _a() -> Marker:
        return Marker("a")

    @router.when(SCHEMA.instrument_type == "B")
    def _b() -> Marker:
        return Marker("b")

    frame = pl.DataFrame(
        {"instrument_type": ["A", "B"], "option_id": ["1", "2"], "price": [1.0, 2.0]}
    ).lazy()
    out = cast(pl.DataFrame, router.compute(frame).collect()).sort("instrument_type")
    assert out["by"].to_list() == ["a", "b"]


def test_router_case_validates_value_arity() -> None:
    SCHEMA = cols(OptionPrice)
    router = Router.on(SCHEMA.instrument_type, SCHEMA.option_id)
    with pytest.raises(ValueError, match="case expects 2 value"):
        router.case("only-one")


def test_workflow_runs_stages_in_topological_order() -> None:
    workflow = Workflow("wf")

    @workflow.stage
    def doubled(base):
        return base.with_columns((pl.col("x") * 2).alias("x"))

    @workflow.stage
    def plus_one(doubled):
        return doubled.with_columns((pl.col("x") + 1).alias("x"))

    env = workflow.run(base=pl.DataFrame({"x": [5.0]}).lazy())
    assert workflow.order() == ["doubled", "plus_one"]
    expected_x = 11.0
    assert cast(pl.DataFrame, env["plus_one"].collect())["x"].item() == expected_x

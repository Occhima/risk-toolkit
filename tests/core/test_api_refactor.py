"""Coverage for the domain-oriented public API: FormulaGraph/Router/Workflow."""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.core.columns import ColumnLike, ColumnRef, col_name, cols
from schenberg.core.graph import FormulaGraph
from schenberg.core.market import MarketRead, MarketRequirement
from schenberg.core.pipeline import Workflow
from schenberg.core.router import Router
from schenberg.domain.schemas.option import OptionPrice
from schenberg.market_data.curves import CurveSpec
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.market_data.volatility import VolSurfaceSpec


def _curve_market() -> MarketSnapshot:
    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame({"id_indexador": [1], "tenor_days": [252], "zero_rate": [0.1]}).lazy(),
            )
        ],
    )


def test_constructor_returns_and_view_sugar() -> None:
    graph = FormulaGraph("sugar", returns=OptionPrice, view="public")

    # The constructor sugar declares a view mapping each schema field to a
    # node of the same name (identity).
    assert graph._views["public"] == {f: f for f in OptionPrice.to_schema().columns}


def test_column_like_helpers() -> None:
    assert col_name("x") == "x"
    assert col_name(ColumnRef("y")) == "y"
    # ColumnLike accepts both forms.
    accepts: ColumnLike = ColumnRef("z")
    assert col_name(accepts) == "z"


def test_returns_accepts_column_ref_override() -> None:
    graph = FormulaGraph("ovr")

    @graph.formula()
    def call_price(spot: pl.Expr) -> pl.Expr:
        return spot

    # A ColumnRef override value is resolved to its .name (the feeding node).
    graph.returns("price", price=ColumnRef("call_price"))
    assert graph._views["price"] == {"price": "call_price"}
    out = cast(
        pl.DataFrame,
        graph.compute(pl.DataFrame({"spot": [3.0]}).lazy(), view="price").collect(),
    )
    expected_price = 3.0
    assert out["price"].item() == expected_price


def test_for_market_finalizes_output_from_kwarg_and_view_dtypes() -> None:
    graph = FormulaGraph("mkt")

    @graph.formula(dtype=pl.Float64)
    def pv(rate: pl.Expr) -> pl.Expr:
        return rate * 2.0

    graph.for_market(rate=CurveSpec("curves").value("zero_rate")).returns("pricing", pv="pv")

    out = cast(
        pl.DataFrame,
        graph.compute(
            pl.DataFrame({"id_indexador": [1], "payment_days": [252]}).lazy(),
            market=_curve_market(),
            view="pricing",
        ).collect(),
    )
    assert out["pv"].item() == pytest.approx(0.2)
    assert graph.view_dtypes("pricing") == {"pv": pl.Float64}


def test_for_market_rejects_mismatched_fixed_output() -> None:
    graph = FormulaGraph("mkt2")
    fixed = CurveSpec("curves").value("zero_rate", output="rate")
    assert isinstance(fixed, MarketRequirement)
    with pytest.raises(ValueError, match="use uses_market"):
        graph.for_market(zero_rate=fixed)


def test_curve_spec_returns_market_read_when_output_omitted() -> None:
    read = CurveSpec("curves").value("zero_rate", indexer="id_indexador", tenor="payment_days")
    assert isinstance(read, MarketRead)
    req = read.as_output("rate")
    assert req.outputs == {"zero_rate": "rate"}


def test_vol_surface_spec_returns_market_read_when_output_omitted() -> None:
    OPT = cols(OptionPrice)  # any schema; just exercising ColumnLike
    read = VolSurfaceSpec("vol_surface").implied_vol(
        indexer=ColumnRef("id_indexador"), tenor="payment_days", strike=OPT.price
    )
    assert isinstance(read, MarketRead)
    req = read.as_output("vol")
    assert req.outputs == {"implied_vol": "vol"}


def test_compose_with_merges_formulas() -> None:
    base = FormulaGraph("base")

    @base.formula()
    def a(x: pl.Expr) -> pl.Expr:
        return x + 1

    other = FormulaGraph("other")

    @other.formula()
    def b(a: pl.Expr) -> pl.Expr:
        return a * 2

    merged = base.compose_with(other, name="merged")
    assert merged.name == "merged"
    assert set(merged.formulas()) == {"a", "b"}
    out = cast(
        pl.DataFrame,
        merged.compute(pl.DataFrame({"x": [1.0]}).lazy(), outputs={"b": "b"}).collect(),
    )
    expected_b = 4.0
    assert out["b"].item() == expected_b


def test_stage_materializes_intermediates_for_a_view() -> None:
    graph = FormulaGraph("staged")

    @graph.formula()
    def step(x: pl.Expr) -> pl.Expr:
        return x + 1

    @graph.formula()
    def out(step: pl.Expr) -> pl.Expr:
        return step * 10

    graph.returns("v", out="out")
    staged = cast(pl.DataFrame, graph.stage(pl.DataFrame({"x": [1.0]}).lazy(), view="v").collect())
    expected_step, expected_out = 2.0, 20.0
    assert staged["step"].item() == expected_step
    assert staged["out"].item() == expected_out


def test_router_on_case_and_when() -> None:
    SCHEMA = cols(OptionPrice)

    class Marker:
        def __init__(self, marker: str) -> None:
            self.marker = marker

        def compute(self, frame, *, market=None, view="result"):
            return frame.with_columns(pl.lit(self.marker).alias("by"))

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

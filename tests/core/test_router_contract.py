"""Router as ArrowChoice: contract-oriented choice among computations."""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.core.columns import cols
from schenberg.core.graph import FormulaGraph, uses
from schenberg.core.router import Router
from schenberg.domain.base import SchenbergDataFrameModel as DataFrameModel
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.market import CURVES


class Trade(DataFrameModel):
    trade_id: str
    kind: str
    spot: float


class Priced(DataFrameModel):
    trade_id: str
    price: float


class Other(DataFrameModel):
    trade_id: str
    value: float


def _graph(name: str, factor: float) -> FormulaGraph:
    g = FormulaGraph(name, input=Trade)
    t = g.input

    @g.formula()
    def price(s: pl.Expr = uses(t.spot)) -> pl.Expr:
        return s * factor

    g.returns("price", Priced, trade_id=t.trade_id, price=price)
    return g


T = cols(Trade)


def _frame() -> pl.LazyFrame:
    return pl.DataFrame({"trade_id": ["a", "b"], "kind": ["A", "B"], "spot": [10.0, 10.0]}).lazy()


def test_returns_records_contract_and_exclusive_is_default() -> None:
    router = Router.on(T.kind).returns("price", Priced)
    assert router.mode == "exclusive"
    assert router.has_view("price")
    assert router.view_schema("price") is Priced


def test_case_and_when_route_rows_under_the_contract() -> None:
    router = Router.on(T.kind).returns("price", Priced).exclusive()

    @router.case("A")
    def _a() -> FormulaGraph:
        return _graph("a", 1.0)

    @router.when(T.kind == "B")
    def _b() -> FormulaGraph:
        return _graph("b", 2.0)

    out = cast(pl.DataFrame, router.compute(_frame(), view="price").collect()).sort("trade_id")
    # contract enforced: exactly the Priced columns, in schema order
    assert out.columns == ["trade_id", "price"]
    assert out["price"].to_list() == [10.0, 20.0]


def test_duplicate_case_in_exclusive_mode_errors() -> None:
    router = Router.on(T.kind).returns("price", Priced).exclusive()

    @router.case("A")
    def _a() -> FormulaGraph:
        return _graph("a", 1.0)

    with pytest.raises(ValueError, match="duplicate case"):

        @router.case("A")
        def _a2() -> FormulaGraph:
            return _graph("a2", 9.0)


def test_branch_without_required_view_errors() -> None:
    router = Router.on(T.kind).returns("price", Priced)

    class NoView:
        def compute(self, frame, *, market=None, view="result"):
            return frame

        def has_view(self, view: str) -> bool:
            return False

        def view_schema(self, view: str):
            return None

    with pytest.raises(ValueError, match="does not provide the router view"):
        router.default(NoView())


def test_branch_with_incompatible_schema_errors() -> None:
    router = Router.on(T.kind).returns("price", Priced)

    wrong = FormulaGraph("wrong", input=Trade)
    t = wrong.input

    @wrong.formula()
    def value(s: pl.Expr = uses(t.spot)) -> pl.Expr:
        return s

    wrong.returns("price", Other, trade_id=t.trade_id, value=value)

    with pytest.raises(ValueError, match="not.*compatible"):

        @router.case("A")
        def _a() -> FormulaGraph:
            return wrong


def test_first_match_gives_priority_to_earlier_branches() -> None:
    router = Router.on(T.kind).returns("price", Priced).first_match()

    @router.when(pl.col("spot") > 0)  # matches everything
    def _all() -> FormulaGraph:
        return _graph("all", 1.0)

    @router.when(T.kind == "B")  # shadowed in first-match mode
    def _b() -> FormulaGraph:
        return _graph("b", 99.0)

    out = cast(pl.DataFrame, router.compute(_frame(), view="price").collect()).sort("trade_id")
    assert out["price"].to_list() == [10.0, 10.0]


def test_explain_lists_cases_and_contract() -> None:
    router = Router.on(T.kind).returns("price", Priced).exclusive()

    @router.case("A")
    def _a() -> FormulaGraph:
        return _graph("a", 1.0)

    text = router.explain(view="price")
    assert "Mode:" in text and "exclusive" in text
    assert "schema: Priced" in text
    assert "ArrowChoice" in text


def test_diagnose_reports_partition_counts() -> None:
    router = Router.on(T.kind).returns("price", Priced).exclusive()

    @router.case("A")
    def _a() -> FormulaGraph:
        return _graph("a", 1.0)

    report = router.diagnose(_frame(), view="price")
    rows = {r["case"]: r["matched"] for r in report.to_dicts()}
    assert rows["A"] == 1
    assert rows["<unmatched>"] == 1


def test_router_market_attachment_flows_through_branches() -> None:
    market = MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame({"id_indexador": [1], "tenor_days": [252], "zero_rate": [0.1]}).lazy(),
            )
        ],
    )

    class MarketTrade(DataFrameModel):
        trade_id: str
        kind: str
        id_indexador: int
        payment_days: int

    def discounted(name: str) -> FormulaGraph:
        g = FormulaGraph(name, input=MarketTrade)
        t = g.input
        m = g.market(rate=CURVES.zero_rate().finalize("zero_rate"))

        @g.formula()
        def price(r: pl.Expr = uses(m.rate)) -> pl.Expr:
            return r * 1000.0

        g.returns("price", Priced, trade_id=t.trade_id, price=price)
        return g

    MT = cols(MarketTrade)
    router = Router.on(MT.kind).returns("price", Priced).exclusive()

    @router.case("A")
    def _a() -> FormulaGraph:
        return discounted("a")

    frame = pl.DataFrame(
        {"trade_id": ["a"], "kind": ["A"], "id_indexador": [1], "payment_days": [252]}
    ).lazy()
    out = cast(pl.DataFrame, router.compute(frame, market=market, view="price").collect())
    assert out["price"].item() == pytest.approx(100.0)

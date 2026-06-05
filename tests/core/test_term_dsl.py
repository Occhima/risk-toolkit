"""The Term DSL: input/market/formula terms, views, composition.

These exercise the conceptual core — terms as the unified abstraction, ``uses``
for explicit dependencies, ``g.market`` as the Reader environment, ``returns`` as
typed views, and open-graph composition.
"""

from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl
import pytest
from schenberg.core.graph import FormulaGraph, Term, TermKind, uses
from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.snapshot import MarketSnapshot
from schenberg.market_data.sources import MarketSource
from schenberg.pricing.market import CURVES


class Trade(SchenbergDataFrameModel):
    trade_id: str
    spot: float
    strike: float
    payment_days: int
    id_indexador: int


class Priced(SchenbergDataFrameModel):
    trade_id: str
    price: float


def _market() -> MarketSnapshot:
    return MarketSnapshot.from_sources(
        as_of=date(2026, 6, 3),
        sources=[
            MarketSource(
                "curves",
                pl.DataFrame({"id_indexador": [1], "tenor_days": [252], "zero_rate": [0.1]}).lazy(),
            )
        ],
    )


def _frame() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "trade_id": ["t1"],
            "spot": [100.0],
            "strike": [100.0],
            "payment_days": [252],
            "id_indexador": [1],
        }
    ).lazy()


# ---- input terms --------------------------------------------------------


def test_input_namespace_yields_stable_terms() -> None:
    g = FormulaGraph("g", input=Trade)
    spot = g.input.spot
    assert isinstance(spot, Term)
    assert spot.kind is TermKind.INPUT
    assert spot.name == "spot"
    # repeated access returns the same registered term
    assert g.input.spot is spot


def test_input_namespace_rejects_undeclared_column() -> None:
    g = FormulaGraph("g", input=Trade)
    with pytest.raises(AttributeError):
        _ = g.input.not_a_column


def test_input_requires_schema() -> None:
    g = FormulaGraph("g")
    with pytest.raises(AttributeError, match="no input schema"):
        _ = g.input


# ---- uses / formula terms ----------------------------------------------


def test_formula_decorator_returns_a_term() -> None:
    g = FormulaGraph("g", input=Trade)
    t = g.input

    @g.formula(symbol="T", latex=r"\frac{d}{252}")
    def year_fraction(d: pl.Expr = uses(t.payment_days)) -> pl.Expr:
        return d / 252.0

    assert isinstance(year_fraction, Term)
    assert year_fraction.kind is TermKind.FORMULA
    assert year_fraction.deps == ("payment_days",)


def test_formula_dependencies_come_from_uses_not_param_names() -> None:
    g = FormulaGraph("g", input=Trade)
    t = g.input

    # parameter is named `whatever`, but the dependency is t.spot via uses()
    @g.formula()
    def doubled(whatever: pl.Expr = uses(t.spot)) -> pl.Expr:
        return whatever * 2

    assert doubled.deps == ("spot",)


def test_bare_term_default_is_accepted() -> None:
    g = FormulaGraph("g", input=Trade)
    t = g.input

    @g.formula()
    def shifted(s=t.spot) -> pl.Expr:
        return s + 1  # ty: ignore[unsupported-operator]  # s is a pl.Expr at runtime

    assert shifted.deps == ("spot",)


def test_strict_graph_rejects_param_without_term_dependency() -> None:
    g = FormulaGraph("g", input=Trade)

    with pytest.raises(ValueError, match="has no Term dependency"):

        @g.formula()
        def bad(spot: pl.Expr) -> pl.Expr:
            return spot


def test_later_formula_depends_on_earlier_formula_term() -> None:
    g = FormulaGraph("g", input=Trade)
    t = g.input

    @g.formula()
    def a(s: pl.Expr = uses(t.spot)) -> pl.Expr:
        return s + 1

    @g.formula()
    def b(a_: pl.Expr = uses(a)) -> pl.Expr:
        return a_ * 2

    out = cast(pl.DataFrame, g.compute(_frame(), outputs={"b": "b"}).collect())
    assert out["b"].item() == pytest.approx((100.0 + 1) * 2)


# ---- market terms -------------------------------------------------------


def test_market_creates_terms_named_by_keyword() -> None:
    g = FormulaGraph("g", input=Trade)
    m = g.market(rate=CURVES.zero_rate().finalize("zero_rate"))
    assert isinstance(m.rate, Term)
    assert m.rate.kind is TermKind.MARKET
    assert m.rate.name == "rate"


def test_market_term_compiles_to_column_after_attachment() -> None:
    g = FormulaGraph("g", input=Trade)
    t = g.input
    m = g.market(rate=CURVES.zero_rate().finalize("zero_rate"))

    @g.formula()
    def price(r: pl.Expr = uses(m.rate), s: pl.Expr = uses(t.spot)) -> pl.Expr:
        return r * s

    g.returns("price", Priced, trade_id=t.trade_id, price=price)
    out = cast(pl.DataFrame, g.compute(_frame(), market=_market(), view="price").collect())
    assert out["price"].item() == pytest.approx(0.1 * 100.0)
    assert "vol" not in out.columns


def test_market_names_a_requirement_output_from_its_keyword() -> None:
    g = FormulaGraph("g", input=Trade)
    m = g.market(rate=CURVES.zero_rate().finalize("zero_rate"))
    assert m.rate.name == "rate"
    assert g._market[0].outputs == {"zero_rate": "rate"}


# ---- views / returns ----------------------------------------------------


def test_returns_registers_view_mapping() -> None:
    g = FormulaGraph("g", input=Trade)
    t = g.input

    @g.formula()
    def price(s: pl.Expr = uses(t.spot)) -> pl.Expr:
        return s

    g.returns("price", Priced, trade_id=t.trade_id, price=price)
    assert g.has_view("price")
    assert g.view_schema("price") is Priced
    assert g._views["price"] == {"trade_id": "trade_id", "price": "price"}


def test_returns_rejects_columns_outside_schema() -> None:
    g = FormulaGraph("g", input=Trade)
    t = g.input

    @g.formula()
    def price(s: pl.Expr = uses(t.spot)) -> pl.Expr:
        return s

    with pytest.raises(ValueError, match="not in schema"):
        g.returns("price", Priced, trade_id=t.trade_id, price=price, bogus=t.spot)


def test_input_and_market_terms_can_be_returned_directly() -> None:
    g = FormulaGraph("g", input=Trade)
    t = g.input
    m = g.market(rate=CURVES.zero_rate().finalize("zero_rate"))
    g.returns("echo", trade_id=t.trade_id, rate=m.rate, spot=t.spot)
    out = cast(pl.DataFrame, g.compute(_frame(), market=_market(), view="echo").collect())
    assert out["rate"].item() == pytest.approx(0.1)
    assert out["spot"].item() == pytest.approx(100.0)


# ---- introspection ------------------------------------------------------


def _priced_graph() -> FormulaGraph:
    g = FormulaGraph("demo", input=Trade)
    t = g.input
    m = g.market(rate=CURVES.zero_rate().finalize("zero_rate"))

    @g.formula(symbol="T", latex=r"\frac{d}{252}")
    def year_fraction(d: pl.Expr = uses(t.payment_days)) -> pl.Expr:
        return d / 252.0

    @g.formula(symbol="P")
    def price(
        s: pl.Expr = uses(t.spot), r: pl.Expr = uses(m.rate), T: pl.Expr = uses(year_fraction)
    ) -> pl.Expr:
        return s * r * T

    g.returns("price", Priced, trade_id=t.trade_id, price=price)
    return g


def test_info_classifies_terms_by_kind() -> None:
    info = _priced_graph().info(view="price")
    assert "vol" not in info.market_outputs
    assert "rate" in info.market_outputs
    assert "year_fraction" in info.formula_nodes
    assert info.view_nodes == {"trade_id": "trade_id", "price": "price"}


def test_explain_reports_inputs_market_and_returns() -> None:
    text = _priced_graph().explain(view="price")
    assert "FormulaGraph demo" in text
    assert "price -> Priced" in text
    assert "rate <- curves(" in text
    assert "Returns:" in text
    assert "price <- price" in text


def test_mermaid_distinguishes_kinds() -> None:
    mer = _priced_graph().to_mermaid(show_kinds=True, view="price")
    assert "class rate market" in mer
    assert "class price formula" in mer
    assert "class price output" in mer


def test_stage_materializes_intermediates_lazily() -> None:
    g = _priced_graph()
    staged = g.stage(_frame(), market=_market(), view="price")
    assert isinstance(staged, pl.LazyFrame)
    out = cast(pl.DataFrame, staged.collect())
    assert "year_fraction" in out.columns  # intermediate is materialized
    assert out["year_fraction"].item() == pytest.approx(1.0)

"""Open-graph composition: merge, extend/compose_with, then."""

from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from schenberg.core.graph import FormulaGraph, uses
from schenberg.domain.base import SchenbergDataFrameModel as DataFrameModel


class Trade(DataFrameModel):
    x: float


class Doubled(DataFrameModel):
    doubled: float


class FutureValue(DataFrameModel):
    future_value: float


def _frame(**cols: list[float]) -> pl.LazyFrame:
    return pl.DataFrame(cols).lazy()


def _val(g: FormulaGraph, frame: pl.LazyFrame, term: str) -> float:
    out = cast(pl.DataFrame, g.compute(frame, outputs={term: term}).collect())
    return out[term].item()


def test_merge_combines_independent_graphs() -> None:
    a = FormulaGraph("a", input=Trade)

    @a.formula()
    def left(x: pl.Expr = uses(a.input.x)) -> pl.Expr:
        return x + 1

    b = FormulaGraph("b", input=Trade)

    @b.formula()
    def right(x: pl.Expr = uses(b.input.x)) -> pl.Expr:
        return x * 10

    merged = a.merge(b, name="merged")
    assert set(merged.formulas()) == {"left", "right"}
    assert _val(merged, _frame(x=[2.0]), "left") == pytest.approx(3.0)
    assert _val(merged, _frame(x=[2.0]), "right") == pytest.approx(20.0)


def test_extend_layers_a_formula_block_on_the_same_environment() -> None:
    base = FormulaGraph("base", input=Trade)

    @base.formula()
    def doubled(x: pl.Expr = uses(base.input.x)) -> pl.Expr:
        return x * 2

    # The layer reads ``doubled`` as an input port; extending wires it to base's
    # ``doubled`` formula (a boundary term satisfied by a formula of the same name).
    layer = FormulaGraph("layer", input=Doubled)

    @layer.formula()
    def quad(d: pl.Expr = uses(layer.input.doubled)) -> pl.Expr:
        return d * 2

    extended = base.extend(layer, name="extended")
    assert _val(extended, _frame(x=[3.0]), "quad") == pytest.approx(12.0)


def test_compose_with_conflicting_formula_definitions_errors() -> None:
    a = FormulaGraph("a", input=Trade)

    @a.formula()
    def f(x: pl.Expr = uses(a.input.x)) -> pl.Expr:
        return x + 1

    b = FormulaGraph("b", input=Trade)

    @b.formula()
    def f(x: pl.Expr = uses(b.input.x)) -> pl.Expr:  # noqa: F811  # same name, different body
        return x - 1

    with pytest.raises(ValueError, match="conflicting formula"):
        a.compose_with(b, name="boom")


def test_then_binds_upstream_output_to_downstream_input() -> None:
    payoff = FormulaGraph("payoff", input=Trade)

    @payoff.formula()
    def future_value(x: pl.Expr = uses(payoff.input.x)) -> pl.Expr:
        return x - 5.0

    discounting = FormulaGraph("discounting", input=FutureValue)

    @discounting.formula()
    def present_value(fv: pl.Expr = uses(discounting.input.future_value)) -> pl.Expr:
        return fv * 0.9

    priced = payoff.then(
        discounting,
        bind={discounting.input.future_value: payoff.output.future_value},
        name="priced",
    )
    assert _val(priced, _frame(x=[15.0]), "present_value") == pytest.approx((15.0 - 5.0) * 0.9)


def test_identity_is_a_unit_for_then() -> None:
    g = FormulaGraph("g", input=Trade)

    @g.formula()
    def y(x: pl.Expr = uses(g.input.x)) -> pl.Expr:
        return x + 7

    left = FormulaGraph.identity().then(g, name="l")
    right = g.then(FormulaGraph.identity(), name="r")
    frame = _frame(x=[1.0])
    assert _val(left, frame, "y") == _val(g, frame, "y") == _val(right, frame, "y")

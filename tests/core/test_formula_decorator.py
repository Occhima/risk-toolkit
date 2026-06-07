from __future__ import annotations

import ast
import math

import polars as pl
import pytest
import schenberg.core.graph as graph_mod
from schenberg.core.expr import exp
from schenberg.core.graph import Formula, FormulaGraph
from schenberg.domain.base import SchenbergDataFrameModel


class ForwardInput(SchenbergDataFrameModel):
    instrument_id: str
    currency: str
    strike: float
    payment_days: int
    forward_rate: float
    risk_free_rate: float


class ForwardOutput(SchenbergDataFrameModel):
    instrument_id: str
    value: float
    delta: float
    currency: str


def decorated_graph(name: str = "forward") -> FormulaGraph:
    g = FormulaGraph(name, input=ForwardInput)

    @g.formula(symbol="T", description="business-year fraction", tags=("time",))
    def year_fraction(payment_days):
        return payment_days / 252.0

    @g.formula(symbol="DF")
    def discount_factor(risk_free_rate, year_fraction):
        return exp(-risk_free_rate * year_fraction)

    @g.formula(symbol="FV")
    def future_value(forward_rate, strike):
        return forward_rate - strike

    @g.formula(symbol="PV")
    def present_value(future_value, discount_factor):
        return future_value * discount_factor

    @g.formula(symbol="Delta")
    def delta(discount_factor):
        return discount_factor

    g.returns(
        "output",
        instrument_id="instrument_id",
        value="present_value",
        delta="delta",
        currency="currency",
    )
    return g


def let_graph(name: str = "forward") -> FormulaGraph:
    g = FormulaGraph(name, input=ForwardInput)
    c = g.input
    year_fraction = g.let("year_fraction", c.payment_days / 252.0, symbol="T")
    discount_factor = g.let("discount_factor", exp(-c.risk_free_rate * year_fraction), symbol="DF")
    future_value = g.let("future_value", c.forward_rate - c.strike, symbol="FV")
    present_value = g.let("present_value", future_value * discount_factor, symbol="PV")
    g.let("delta", discount_factor, symbol="Delta")
    g.returns(
        "output",
        instrument_id="instrument_id",
        value=present_value,
        delta="delta",
        currency="currency",
    )
    return g


def frame() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["FWD-1"],
            "currency": ["USD"],
            "strike": [100.0],
            "payment_days": [252],
            "forward_rate": [110.0],
            "risk_free_rate": [0.10],
        }
    ).lazy()


def test_decorator_registers_formula_and_default_name() -> None:
    g = decorated_graph()
    assert "year_fraction" in g.formulas()
    assert g.formula_of("year_fraction").startswith("T =")
    assert g.info(view="output").formula_nodes == (
        "year_fraction",
        "discount_factor",
        "future_value",
        "present_value",
        "delta",
    )


def test_name_override_changes_registered_term_name() -> None:
    g = FormulaGraph("override", input=ForwardInput)

    @g.formula(name="t", symbol="T")
    def year_fraction(c):
        return c.payment_days / 252.0

    assert "t" in g.formulas()
    assert "year_fraction" not in g.formulas()


def test_formula_facade_supports_decorator() -> None:
    f = Formula[ForwardInput, ForwardOutput]("facade")

    @f.formula(symbol="V")
    def value(forward_rate, strike):
        return forward_rate - strike

    @f.formula(symbol="Delta")
    def delta():
        return 1.0

    f.returns(ForwardOutput)
    out = f.plan(frame()).collect()
    assert out.select("value").item() == pytest.approx(10.0)
    assert out.select("delta").item() == pytest.approx(1.0)


def test_headless_parameter_resolves_to_input_column() -> None:
    """A bare parameter named after an input column is injected as ``var(name)``
    — no ``c.`` indirection — and the compiled plan reads that column."""
    g = FormulaGraph("headless", input=ForwardInput)

    @g.formula(symbol="FV")
    def future_value(forward_rate, strike):
        return forward_rate - strike

    assert g.dependencies_of("future_value") == {"forward_rate", "strike"}
    out = g.plan(frame(), outputs={"fv": "future_value"}).collect()
    assert out.select("fv").item() == pytest.approx(10.0)


def test_legacy_namespace_parameter_still_supported() -> None:
    """``c``/``contract``/``input``/``inputs`` keep receiving the input namespace,
    so the lower-level ``c.<col>`` style remains backward compatible."""
    g = FormulaGraph("legacy", input=ForwardInput)

    @g.formula(symbol="FV")
    def future_value(c):
        return c.forward_rate - c.strike

    out = g.plan(frame(), outputs={"fv": "future_value"}).collect()
    assert out.select("fv").item() == pytest.approx(10.0)


def test_dependency_by_parameter_and_input_namespace() -> None:
    g = decorated_graph()
    assert g.dependencies_of("present_value") == {
        "future_value",
        "discount_factor",
        "forward_rate",
        "strike",
        "risk_free_rate",
        "payment_days",
        "year_fraction",
    }
    out = g.plan(frame(), view="output").collect()
    assert out.select("value").item() == pytest.approx(10.0 * math.exp(-0.10))
    assert out.select("delta").item() == pytest.approx(math.exp(-0.10))


def test_unknown_parameter_fails_early() -> None:
    g = FormulaGraph("bad", input=ForwardInput)

    with pytest.raises(ValueError, match="unknown formula dependency 'x' in formula 'bad_formula'"):

        @g.formula()
        def bad_formula(x):
            return x


def test_decorated_graph_equivalent_to_let_graph() -> None:
    decorated = decorated_graph("decorated").plan(frame(), view="output").collect()
    manual = let_graph("manual").plan(frame(), view="output").collect()
    assert decorated.columns == manual.columns
    assert (
        decorated.select("instrument_id", "currency").to_dicts()
        == manual.select("instrument_id", "currency").to_dicts()
    )
    for column in ("value", "delta"):
        assert decorated[column].to_list() == pytest.approx(manual[column].to_list())


def test_introspection_stage_plan_required_inputs_and_order_still_work() -> None:
    g = decorated_graph()
    assert "PV =" in g.formula_of("present_value")
    assert "present_value" in g.explain(view="output")
    assert "present_value" in g.to_mermaid()
    assert isinstance(g.plan(frame(), view="output"), pl.LazyFrame)
    assert isinstance(g.stage(frame(), view="output"), pl.LazyFrame)
    assert g.required_inputs("output") == {
        "instrument_id",
        "currency",
        "strike",
        "payment_days",
        "forward_rate",
        "risk_free_rate",
    }
    order = g.topological_order()
    assert order.index("year_fraction") < order.index("discount_factor")
    assert order.index("discount_factor") < order.index("present_value")


def test_graph_creation_and_planning_do_not_call_collect() -> None:
    with open(graph_mod.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "collect"
    ]
    assert not calls


def test_plan_materialized_matches_inlined_result() -> None:
    g = decorated_graph()

    materialized = g.plan(frame(), view="output", materialize_terms=True).collect()
    inlined = (
        g.plan(frame(), view="output", materialize_terms=False)
        .select(materialized.columns)
        .collect()
    )

    assert materialized.to_dicts() == inlined.to_dicts()


def test_plan_materialized_does_not_return_intermediate_columns() -> None:
    out = decorated_graph().plan(frame(), view="output").collect()

    assert out.columns == ["instrument_id", "value", "delta", "currency"]
    assert "year_fraction" not in out.columns
    assert "discount_factor" not in out.columns


def test_stage_still_returns_intermediate_columns() -> None:
    staged = decorated_graph().stage(frame(), view="output").collect()

    assert "year_fraction" in staged.columns
    assert "discount_factor" in staged.columns
    assert "present_value" in staged.columns


def test_term_cannot_shadow_input_column() -> None:
    g = FormulaGraph("bad", input=ForwardInput)

    with pytest.raises(ValueError, match="shadows an input column"):
        g.let("strike", 1.0)


def test_plan_remains_lazy() -> None:
    assert isinstance(decorated_graph().plan(frame(), view="output"), pl.LazyFrame)

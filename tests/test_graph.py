from __future__ import annotations

from typing import cast

import polars as pl
from schenberg.core.graph import FormulaGraph


def test_formula_graph_compiles_view_outputs() -> None:
    graph = FormulaGraph("demo")

    @graph.formula()
    def total(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return a + b

    out = cast(
        pl.DataFrame,
        graph.compute(
            pl.DataFrame({"a": [1.0], "b": [2.0]}).lazy(),
            outputs={"result": "total"},
        ).collect(),
    )

    expected_result = 3.0
    assert out["result"].item() == expected_result
    assert graph.required_inputs() == {"a", "b"}


def test_formula_metadata_and_introspection() -> None:
    graph = FormulaGraph("demo_meta")

    @graph.formula(symbol="c", latex="a + b", description="sum")
    def total(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return a + b

    @graph.formula()
    def doubled(total: pl.Expr) -> pl.Expr:
        return total * 2

    graph.returns("out", result="doubled")

    assert graph.formula_of("total") == "c = a + b"
    assert graph.formula_of("doubled") == r"doubled = \operatorname{doubled}(total)"
    assert set(graph.formulas()) == {"total", "doubled"}
    assert "total --> doubled" in graph.to_mermaid()
    assert "c = a + b" in graph.to_mermaid(math_labels=True)
    info = graph.info(view="out")
    assert info.view_nodes == {"result": "doubled"}
    assert "total" in info.intermediate_nodes
    explanation = graph.explain(view="out")
    assert "Graph: demo_meta" in explanation
    assert "c = a + b" in explanation

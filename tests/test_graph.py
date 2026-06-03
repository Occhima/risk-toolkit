from __future__ import annotations

from typing import cast

import polars as pl
from schenberg.core.graph import ExprGraph


def test_expr_graph_compiles_formula_outputs() -> None:
    graph = ExprGraph("demo")

    @graph.node()
    def total(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return a + b

    out = cast(
        pl.DataFrame,
        graph.compute_for(
            pl.DataFrame({"a": [1.0], "b": [2.0]}).lazy(),
            outputs={"result": "total"},
        ).collect(),
    )

    expected_result = 3.0
    assert out["result"].item() == expected_result
    assert graph.required_inputs() == {"a", "b"}


def test_formula_metadata_and_introspection() -> None:
    graph = ExprGraph("demo_meta")

    @graph.node(symbol="c", formula="a + b", description="sum")
    def total(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return a + b

    @graph.node()
    def doubled(total: pl.Expr) -> pl.Expr:
        return total * 2

    graph.with_outputs("out", result="doubled")

    assert graph.formula_of("total") == "c = a + b"
    assert graph.formula_of("doubled") == r"doubled = \operatorname{doubled}(total)"
    assert set(graph.formulas()) == {"total", "doubled"}
    assert "total --> doubled" in graph.to_mermaid()
    assert "c = a + b" in graph.to_mermaid(math_labels=True)
    info = graph.info(output_profile="out")
    assert info.output_nodes == {"result": "doubled"}
    assert "total" in info.intermediate_nodes
    explanation = graph.explain(output_profile="out")
    assert "Graph: demo_meta" in explanation
    assert "c = a + b" in explanation

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

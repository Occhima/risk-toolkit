from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import polars as pl
from schenberg.core.graph import FormulaGraph, uses
from schenberg.domain.base import DataFrameModel

from schenberg_distributed import (
    PricingExecutionContext,
    collect_pricing,
    compute_graph_pricing,
    register_backend,
)


def test_local_context_forwards_collect_kwargs() -> None:
    lf = pl.DataFrame({"x": [1, 2]}).lazy().with_columns((pl.col("x") + 1).alias("value"))

    result = collect_pricing(lf, context=PricingExecutionContext.local(engine="streaming"))

    assert result["value"].to_list() == [2, 3]


def test_compute_graph_pricing_collects_formula_outputs() -> None:
    class Trade(DataFrameModel):
        quantity: float
        price: float

    graph = FormulaGraph("test_pricing", input=Trade)
    t = graph.input

    @graph.formula()
    def value(quantity: pl.Expr = uses(t.quantity), price: pl.Expr = uses(t.price)) -> pl.Expr:
        return quantity * price

    trades = pl.DataFrame({"quantity": [2.0, 3.0], "price": [10.0, 7.0]}).lazy()

    result = compute_graph_pricing(
        graph,
        trades,
        context=PricingExecutionContext.local(),
        outputs={"value": "value"},
    )

    assert result["value"].to_list() == [20.0, 21.0]


def test_custom_backend_hook_receives_collect_kwargs() -> None:
    seen: dict[str, Any] = {}

    def custom_collect(lf: pl.LazyFrame, collect_kwargs: Mapping[str, Any]) -> pl.DataFrame:
        seen.update(collect_kwargs)
        return cast(pl.DataFrame, lf.collect())

    register_backend("unit-test", custom_collect)
    lf = pl.DataFrame({"value": [1]}).lazy()

    result = collect_pricing(lf, context=PricingExecutionContext.custom("unit-test", engine="auto"))

    assert result["value"].to_list() == [1]
    assert seen == {"engine": "auto"}

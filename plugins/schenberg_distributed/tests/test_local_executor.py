from __future__ import annotations

import polars as pl

from schenberg_distributed import LocalExecutor, ValuationPlan


def test_local_executor_lazy() -> None:
    plan = ValuationPlan("p").input("x", pl.DataFrame({"x": [1, 2]}).lazy())

    @plan.node("y", x="x")
    def y(x: pl.LazyFrame) -> pl.LazyFrame:
        return x.with_columns((pl.col("x") + 1).alias("y"))

    executor = LocalExecutor()
    lazy = executor.lazy(plan, target="y")
    assert isinstance(lazy, pl.LazyFrame)
    result = executor.collect(plan, target="y")
    assert result["y"].to_list() == [2, 3]


def test_concat_node_lazy() -> None:
    plan = ValuationPlan("p").input("x", pl.DataFrame({"x": [1]}).lazy())

    @plan.node("a", x="x")
    def a(x: pl.LazyFrame) -> pl.LazyFrame:
        return x.with_columns(pl.lit("a").alias("src"))

    @plan.node("b", x="x")
    def b(x: pl.LazyFrame) -> pl.LazyFrame:
        return x.with_columns(pl.lit("b").alias("src"))

    plan.concat("all", ("a", "b"))
    executor = LocalExecutor()
    lazy = executor.lazy(plan, target="all")
    assert isinstance(lazy, pl.LazyFrame)
    result = executor.collect(plan, target="all")
    assert result["src"].to_list() == ["a", "b"]

from __future__ import annotations

from datetime import date

import polars as pl
from schenberg import MarketSnapshot

from schenberg_distributed import (
    LocalExecutor,
    PartitionedPricingPlan,
    ValuationPlan,
    collect_partitioned_local,
)


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


def test_partitioned_pricing_matches_single_collect() -> None:
    trades = pl.DataFrame(
        {"instrument_id": ["A", "B", "C"], "desk": ["D1", "D1", "D2"], "price": [1.0, 2.0, 3.0]}
    )
    market = MarketSnapshot.at(date(2026, 6, 6)).build()

    def pricer(part: pl.LazyFrame, market: MarketSnapshot) -> pl.LazyFrame:
        del market
        return part.select("instrument_id", "desk", (pl.col("price") * 2.0).alias("value"))

    plan = PartitionedPricingPlan(pricer=pricer, partition_by=("desk",))

    partitioned = collect_partitioned_local(trades, market, plan).sort("instrument_id")
    single = pricer(trades.lazy(), market).collect().sort("instrument_id")

    assert partitioned.to_dicts() == single.to_dicts()

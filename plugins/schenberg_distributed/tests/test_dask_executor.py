from __future__ import annotations

import polars as pl
import pytest

from schenberg_distributed import DaskExecutor, LocalExecutor, ValuationPlan


def test_dask_executor_matches_local_executor() -> None:
    pytest.importorskip("dask")
    plan = ValuationPlan("p").input("x", pl.DataFrame({"x": [1, 2]}).lazy())

    @plan.node("y", x="x")
    def y(x: pl.DataFrame) -> pl.LazyFrame:
        return x.lazy().with_columns((pl.col("x") + 1).alias("y"))

    local = LocalExecutor().collect(plan, target="y")
    dask = DaskExecutor().collect(plan, target="y")
    assert dask.equals(local)

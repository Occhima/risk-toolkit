from __future__ import annotations

from dataclasses import FrozenInstanceError

import polars as pl
import pytest

from schenberg_distributed import LocalExecutor, ValuationPlan
from schenberg_live import (
    DependencyIndex,
    LiveValuationEngine,
    MarketEvent,
    PositionEvent,
    ValuationCache,
)


def _plan() -> ValuationPlan:
    plan = ValuationPlan("p").input("trades", pl.DataFrame({"x": [1]}).lazy()).input("market", 1)

    @plan.node("price", trades="trades", market="market", market_sources=("curves",))
    def price(trades: pl.LazyFrame, market: int) -> pl.LazyFrame:
        return trades.with_columns((pl.col("x") + market).alias("value"))

    @plan.node("rollup", price="price")
    def rollup(price: pl.LazyFrame) -> pl.LazyFrame:
        return price.select(pl.col("value").sum().alias("value"))

    return plan


def test_market_event_frozen() -> None:
    event = MarketEvent(source="curves", version="v1")
    with pytest.raises(FrozenInstanceError):
        event.version = "v2"  # type: ignore[misc]


def test_cache_get_set_invalidate() -> None:
    cache = ValuationCache()
    data = pl.DataFrame({"x": [1]})
    cache.set("t", "v1", data)
    assert cache.get("t", "v1") is data
    cache.invalidate(["t"])
    assert cache.get("t", "v1") is None


def test_dependency_index_market_source() -> None:
    index = DependencyIndex.from_plan(_plan())
    assert index.affected_by(MarketEvent(source="curves", version="v1")) == ("price", "rollup")


def test_dependency_index_position_event() -> None:
    plan = ValuationPlan("p").input("positions", pl.DataFrame({"x": [1]}).lazy())

    @plan.node("values", positions="positions")
    def values(positions: pl.LazyFrame) -> pl.LazyFrame:
        return positions

    assert DependencyIndex.from_plan(plan).affected_by(PositionEvent(version="v1")) == ("values",)


def test_live_engine_runs() -> None:
    engine = LiveValuationEngine(plan=_plan(), executor=LocalExecutor(), target="rollup")
    result = engine.on_market_event(MarketEvent(source="curves", version="v1"))
    assert result.data["value"].to_list() == [2]
    assert result.cache_hit is False


def test_live_engine_cache_hit() -> None:
    engine = LiveValuationEngine(plan=_plan(), executor=LocalExecutor(), target="rollup")
    engine.on_market_event(MarketEvent(source="curves", version="v1"))
    result = engine.on_market_event(MarketEvent(source="curves", version="v1"))
    assert result.cache_hit is True


def test_live_engine_recomputes_new_version() -> None:
    engine = LiveValuationEngine(plan=_plan(), executor=LocalExecutor(), target="rollup")
    first = engine.on_market_event(MarketEvent(source="curves", version="v1"))
    second = engine.on_market_event(MarketEvent(source="curves", version="v2"))
    assert first.cache_hit is False
    assert second.cache_hit is False

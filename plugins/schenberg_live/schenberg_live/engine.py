"""Synchronous in-memory live valuation engine."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from schenberg_distributed import ValuationPlan

from .cache import ValuationCache
from .events import MarketEvent, PositionEvent
from .index import DependencyIndex


@dataclass(frozen=True, slots=True)
class LiveResult:
    target: str
    version: str
    affected_nodes: tuple[str, ...]
    data: pl.DataFrame
    cache_hit: bool = False


@dataclass(slots=True)
class LiveValuationEngine:
    plan: ValuationPlan
    executor: Any
    target: str
    cache: ValuationCache = field(default_factory=ValuationCache)
    index: DependencyIndex | None = None

    def __post_init__(self) -> None:
        if self.index is None:
            self.index = DependencyIndex.from_plan(self.plan)

    def on_market_event(self, event: MarketEvent) -> LiveResult:
        return self.run(event.version, affected_nodes=self.index.affected_by(event))

    def on_position_event(self, event: PositionEvent) -> LiveResult:
        return self.run(event.version, affected_nodes=self.index.affected_by(event))

    def run(self, version: str, affected_nodes: Iterable[str] = ()) -> LiveResult:
        cached = self.cache.get(self.target, version)
        affected = tuple(affected_nodes)
        if cached is not None:
            return LiveResult(
                target=self.target,
                version=version,
                affected_nodes=affected,
                data=cached,
                cache_hit=True,
            )
        self.cache.invalidate(affected)
        data = self.executor.collect(self.plan, target=self.target)
        self.cache.set(self.target, version, data)
        return LiveResult(
            target=self.target,
            version=version,
            affected_nodes=affected,
            data=data,
            cache_hit=False,
        )

"""Dependency indexing for live valuation events."""

from __future__ import annotations

from dataclasses import dataclass

from schenberg_distributed import ValuationPlan

from .events import MarketEvent, PositionEvent


@dataclass(frozen=True, slots=True)
class DependencyIndex:
    plan: ValuationPlan

    @classmethod
    def from_plan(cls, plan: ValuationPlan) -> DependencyIndex:
        return cls(plan=plan)

    def affected_by(self, event: MarketEvent | PositionEvent) -> tuple[str, ...]:
        if isinstance(event, MarketEvent):
            return self.plan.affected_by_market_source(event.source)
        if not self.plan.has("positions"):
            return ()
        affected = set(self.plan.downstream_of("positions"))
        for name in list(affected):
            affected.update(self.plan.downstream_of(name))
        return tuple(name for name in self.plan.topological_order() if name in affected)

    def downstream_of(self, name: str) -> tuple[str, ...]:
        return self.plan.downstream_of(name)

    def upstream_of(self, name: str) -> tuple[str, ...]:
        return self.plan.upstream_of(name)

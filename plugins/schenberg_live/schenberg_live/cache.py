"""Tiny DataFrame cache for live valuation results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import polars as pl


@dataclass(frozen=True, slots=True)
class CacheKey:
    target: str
    version: str


@dataclass(slots=True)
class ValuationCache:
    store: dict[CacheKey, pl.DataFrame] = field(default_factory=dict)

    def get(self, target: str, version: str) -> pl.DataFrame | None:
        return self.store.get(CacheKey(target, version))

    def set(self, target: str, version: str, data: pl.DataFrame) -> None:
        self.store[CacheKey(target, version)] = data

    def invalidate(self, targets: Iterable[str]) -> None:
        target_set = set(targets)
        if not target_set:
            return
        for key in list(self.store):
            if key.target in target_set:
                del self.store[key]

    def clear(self) -> None:
        self.store.clear()

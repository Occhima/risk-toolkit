from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType

import polars as pl

from schenberg.core.market import MarketDependency
from schenberg.market_data.sources import MarketSource


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    as_of: date
    sources: Mapping[str, MarketSource]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))

    @classmethod
    def from_sources(
        cls,
        *,
        as_of: date,
        sources: Iterable[MarketSource],
    ) -> MarketSnapshot:
        return cls(as_of=as_of, sources={source.name: source for source in sources})

    def source(self, name: str) -> MarketSource:
        try:
            return self.sources[name]
        except KeyError:
            raise ValueError(f"snapshot has no market source {name!r}") from None

    def attach(self, lf: pl.LazyFrame, req: MarketDependency) -> pl.LazyFrame:
        return req.attach(lf, self)

    def with_source(self, source: MarketSource) -> MarketSnapshot:
        sources = dict(self.sources)
        sources[source.name] = source
        return MarketSnapshot(as_of=self.as_of, sources=sources)

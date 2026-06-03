from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType

import polars as pl

from schenberg.core.market import MarketRequirement
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

    def attach(self, lf: pl.LazyFrame, req: MarketRequirement) -> pl.LazyFrame:
        src = self.source(req.table).data
        right = src.select([*req.right_keys, *req.outputs.keys()]).rename(req.outputs)
        left_names = set(lf.collect_schema().names())

        # A routed/filtered subframe may lack a left join key (e.g. a derived
        # tenor that only one instrument family supplies). Materialize it as a
        # typed null so the left join is well-formed and simply finds no match.
        right_schema = right.collect_schema()
        missing_keys = [
            (left, right_key)
            for left, right_key in zip(req.left_keys, req.right_keys, strict=True)
            if left not in left_names
        ]
        if missing_keys:
            lf = lf.with_columns(
                pl.lit(None, dtype=right_schema[right_key]).alias(left)
                for left, right_key in missing_keys
            )

        output_columns = set(req.outputs.values())
        droppable = output_columns - set(req.left_keys)
        collisions = sorted(droppable & left_names)
        if collisions:
            lf = lf.drop(collisions)

        return lf.join(
            right,
            left_on=list(req.left_keys),
            right_on=list(req.right_keys),
            how="left",
        )

    def with_source(self, source: MarketSource) -> MarketSnapshot:
        sources = dict(self.sources)
        sources[source.name] = source
        return MarketSnapshot(as_of=self.as_of, sources=sources)

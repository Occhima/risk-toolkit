from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING

import polars as pl

from schenberg.core.market import MarketDependency
from schenberg.market_data.objects.errors import MissingMarketSourceError
from schenberg.market_data.sources import MarketSource

if TYPE_CHECKING:
    from schenberg.market_data.shocks import Shock


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    as_of: date
    sources: Mapping[str, MarketSource]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))

    @classmethod
    def at(cls, as_of: date) -> _SnapshotBuilder:
        """Start a fluent snapshot build: name each source, then ``.build()``::

        market = (
            MarketSnapshot.at(date(2026, 6, 6))
            .source("curves", curves_lf, unique_by=("id_indexador", "tenor_days"))
            .source("fixings", fixings_lf, unique_by=("id_indexador", "fixing_date"))
            .build()
        )
        """
        return _SnapshotBuilder(as_of=as_of, _sources=[])

    @classmethod
    def from_sources(
        cls,
        *,
        as_of: date,
        sources: Iterable[MarketSource],
        validate: bool = True,
    ) -> MarketSnapshot:
        materialized = tuple(sources)
        if validate:
            for source in materialized:
                source.validate_unique_keys()
        return cls(as_of=as_of, sources={source.name: source for source in materialized})

    def source(self, name: str) -> MarketSource:
        try:
            return self.sources[name]
        except KeyError:
            raise MissingMarketSourceError(name, tuple(self.sources)) from None

    def attach(self, lf: pl.LazyFrame, req: MarketDependency) -> pl.LazyFrame:
        return req.attach(lf, self)

    def with_source(self, source: MarketSource) -> MarketSnapshot:
        sources = dict(self.sources)
        sources[source.name] = source
        return MarketSnapshot(as_of=self.as_of, sources=sources)

    def apply(self, shock: Shock) -> MarketSnapshot:
        """Apply a :class:`Shock` (a ``MarketSnapshot -> MarketSnapshot`` endomorphism),
        returning a new snapshot. The original is never mutated."""
        return shock(self)


@dataclass
class _SnapshotBuilder:
    """Fluent accumulator for :meth:`MarketSnapshot.at`. Mutable while building;
    :meth:`build` freezes it into an immutable :class:`MarketSnapshot`."""

    as_of: date
    _sources: list[MarketSource]

    def source(
        self,
        name: str,
        data: pl.LazyFrame | pl.DataFrame,
        *,
        unique_by: tuple[str, ...] = (),
        schema: type | None = None,
    ) -> _SnapshotBuilder:
        """Register one market source by name and its quote-key uniqueness."""
        lf = data.lazy() if isinstance(data, pl.DataFrame) else data
        self._sources.append(MarketSource(name=name, data=lf, schema=schema, unique_by=unique_by))
        return self

    def add(self, schema: type, data: pl.LazyFrame | pl.DataFrame) -> _SnapshotBuilder:
        """Register a source from a schema that declares ``__source_name__`` and
        ``__unique_by__`` — no need to repeat the name or key by hand."""
        name = getattr(schema, "__source_name__", None)
        if name is None:
            raise ValueError(
                f"{schema.__name__} declares no __source_name__; use .source(name, data, ...)"
            )
        unique_by = tuple(getattr(schema, "__unique_by__", ()))
        return self.source(name, data, unique_by=unique_by, schema=schema)

    def build(self, *, validate: bool = True) -> MarketSnapshot:
        """Freeze into a :class:`MarketSnapshot`, validating quote-key uniqueness
        at this explicit market boundary unless ``validate=False``."""
        return MarketSnapshot.from_sources(
            as_of=self.as_of, sources=self._sources, validate=validate
        )

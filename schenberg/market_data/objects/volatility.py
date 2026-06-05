"""Canonical volatility-surface market object.

`VolatilitySurface` mirrors :class:`ForwardCurve`: it takes raw vendor quotes,
normalizes them lazily into a canonical schema (``VolatilityPoint``), and
exports a :class:`MarketSource` consumable by :class:`MarketSnapshot` and the
declarative ``MarketRequirements`` layer.

The canonical schema supports the common surface parameterizations used in
practice (expiry x strike, expiry x moneyness, expiry x delta, and term-tenor
variants) by leaving the non-relevant axis columns nullable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import pandera.polars as pa
import polars as pl

from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.objects.conventions import VolatilityConvention
from schenberg.market_data.sources import MarketSource


class VolatilityPoint(SchenbergDataFrameModel):
    """Canonical volatility-quote row.

    Only ``surface``, ``expiry`` and ``volatility`` are required. The remaining
    axis columns are nullable so a single schema can describe matrices keyed by
    strike, moneyness, delta, or term-tenor without proliferating subclasses.
    """

    surface: str
    expiry: date
    tenor: date = pa.Field(nullable=True)
    strike: float = pa.Field(nullable=True)
    moneyness: float = pa.Field(nullable=True)
    delta: float = pa.Field(nullable=True)
    volatility: float


@dataclass(frozen=True, slots=True)
class VolatilitySurface:
    """Reusable, lazy volatility-surface market object.

    Holds the raw quotes as a ``LazyFrame`` plus a ``VolatilityConvention``
    describing the axes and quote kind. ``normalize`` stamps the canonical
    ``surface`` column without touching axis values, and ``to_market_source``
    publishes the surface as a canonical ``MarketSource`` for use inside a
    ``MarketSnapshot``.
    """

    name: str
    ref_date: date
    data: pl.LazyFrame
    convention: VolatilityConvention

    @classmethod
    def from_frame(
        cls,
        raw: pl.DataFrame | pl.LazyFrame,
        *,
        name: str,
        ref_date: date,
        convention: VolatilityConvention,
        normalize: bool = True,
    ) -> VolatilitySurface:
        lf = raw.lazy() if isinstance(raw, pl.DataFrame) else raw
        surface = cls(name=name, ref_date=ref_date, data=lf, convention=convention)
        if normalize:
            surface = surface.normalize()
        return surface

    def normalize(self) -> VolatilitySurface:
        schema_names = self.data.collect_schema().names()  # schema-only, NOT a data collect
        prep: list[pl.Expr] = [pl.lit(self.name).alias("surface")]
        for axis in self.convention.axes:
            if axis not in schema_names:
                prep.append(pl.lit(None).alias(axis))
        normalized = self.data.with_columns(prep)
        return replace(self, data=normalized)

    def to_market_source(self) -> MarketSource:
        return MarketSource(name=self.name, data=self.data, schema=VolatilityPoint)

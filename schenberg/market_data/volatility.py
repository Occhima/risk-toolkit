"""Implied-volatility surfaces as declarative market data.

A vol surface is no longer a special abstraction: it is one 2-D instance of the
general :mod:`schenberg.market_data.interpolated` machine, keyed by
``id_indexador`` and interpolated over ``(tenor_days, strike)``. The thin
wrappers here keep the familiar vol-flavoured names and the ``Spec``/``source``
declaration idiom used by the option graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandera.polars as pa
from pandera.typing.polars import LazyFrame

from schenberg.core.columns import ColumnLike
from schenberg.domain.schemas.market_data import VolSurfaceContract
from schenberg.market_data.interpolated import InterpolatedRequirement, InterpolatedSpec
from schenberg.market_data.sources import MarketSource


@dataclass(frozen=True, slots=True)
class VolSurfaceSpec:
    name: str = "vol_surface"

    def implied_vol(
        self,
        *,
        indexer: ColumnLike = "id_indexador",
        tenor: ColumnLike = "payment_days",
        strike: ColumnLike = "strike",
        output: str | None = None,
    ) -> InterpolatedRequirement:
        spec = InterpolatedSpec(self.name, axes=("tenor_days", "strike"))
        return spec.value("implied_vol", output=output, on=(tenor, strike), group_col=indexer)


@dataclass(frozen=True, slots=True)
class VolSurfaces:
    data: LazyFrame[VolSurfaceContract]
    name: str = "vol_surface"

    @classmethod
    @pa.check_types(lazy=True)
    def build(
        cls,
        data: LazyFrame[VolSurfaceContract],
        *,
        name: str = "vol_surface",
    ) -> VolSurfaces:
        return cls(data=data, name=name)

    def source(self) -> MarketSource:
        return MarketSource(name=self.name, data=self.data, schema=VolSurfaceContract)

    def spec(self) -> VolSurfaceSpec:
        return VolSurfaceSpec(name=self.name)

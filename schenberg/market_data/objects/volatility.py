from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandera.polars as pa
import polars as pl

from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.objects.conventions import VolatilityConvention
from schenberg.market_data.sources import MarketSource


class VolatilityPoint(SchenbergDataFrameModel):
    surface: str
    expiry: date
    tenor: date = pa.Field(nullable=True)
    strike: float = pa.Field(nullable=True)
    moneyness: float = pa.Field(nullable=True)
    delta: float = pa.Field(nullable=True)
    volatility: float


@dataclass(frozen=True, slots=True)
class VolatilitySurface:
    name: str
    ref_date: date
    data: pl.DataFrame
    convention: VolatilityConvention

    def __post_init__(self) -> None:
        df = self.data
        if "surface" not in df.columns:
            df = df.with_columns(pl.lit(self.name).alias("surface"))
        for col, dtype in [
            ("tenor", pl.Date),
            ("strike", pl.Float64),
            ("moneyness", pl.Float64),
            ("delta", pl.Float64),
        ]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
        object.__setattr__(self, "data", VolatilityPoint.validate(df))

    @classmethod
    def from_frame(
        cls,
        raw: pl.DataFrame | pl.LazyFrame,
        *,
        name: str,
        ref_date: date,
        convention: VolatilityConvention,
    ) -> VolatilitySurface:
        df = raw.collect() if isinstance(raw, pl.LazyFrame) else raw
        return cls(name=name, ref_date=ref_date, data=df, convention=convention)

    def to_market_source(self) -> MarketSource:
        return MarketSource(name=self.name, data=self.data.lazy(), schema=VolatilityPoint)

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import pandera.polars as pa
import polars as pl

from schenberg.domain.base import SchenbergDataFrameModel
from schenberg.market_data.objects.conventions import CurveConvention
from schenberg.market_data.sources import MarketSource


class CurvePoint(SchenbergDataFrameModel):
    curve: str
    ref_date: date = pa.Field(nullable=True)
    tenor: date
    business_days: int = pa.Field(nullable=True)
    year_fraction: float = pa.Field(nullable=True)
    rate: float = pa.Field(nullable=True)
    factor: float = pa.Field(nullable=True)
    unit_price: float = pa.Field(nullable=True)


@dataclass(frozen=True, slots=True)
class ForwardCurve:
    name: str
    ref_date: date
    data: pl.DataFrame
    convention: CurveConvention

    def __post_init__(self) -> None:
        df = self.data
        if "curve" not in df.columns:
            df = df.with_columns(pl.lit(self.name).alias("curve"))
        if "ref_date" not in df.columns:
            df = df.with_columns(pl.lit(self.ref_date).alias("ref_date"))
        df = df.with_columns(
            self.convention.normalize_exprs(ref_date_col="ref_date", tenor_col="tenor")
        )
        for col in ("rate", "factor", "unit_price"):
            if col not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
        object.__setattr__(self, "data", CurvePoint.validate(df))

    @classmethod
    def from_frame(
        cls,
        raw: pl.DataFrame | pl.LazyFrame,
        *,
        name: str,
        ref_date: date,
        convention: CurveConvention,
    ) -> ForwardCurve:
        df = cast(pl.DataFrame, raw.collect()) if isinstance(raw, pl.LazyFrame) else raw
        return cls(name=name, ref_date=ref_date, data=df, convention=convention)

    def to_market_source(self) -> MarketSource:
        return MarketSource(name=self.name, data=self.data.lazy(), schema=CurvePoint)

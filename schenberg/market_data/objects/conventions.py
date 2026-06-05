from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import polars as pl

from schenberg.market_data.objects.compounding import Compounding


@runtime_checkable
class Calendar(Protocol):
    name: str
    base_days: int

    def business_days_between_expr(self, start_col: str, end_col: str) -> pl.Expr: ...


class QuoteKind(StrEnum):
    RATE = "rate"
    FACTOR = "factor"
    UNIT_PRICE = "unit_price"


class VolQuoteKind(StrEnum):
    LOGNORMAL = "lognormal"
    NORMAL = "normal"
    BLACK = "black"


class InterpolationKind(StrEnum):
    LINEAR = "linear"
    BILINEAR = "bilinear"


@dataclass(frozen=True, slots=True)
class InterpolationPolicy:
    kind: InterpolationKind
    axes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CurveConvention:
    calendar: Calendar
    compounding: Compounding
    quote_kind: QuoteKind

    def normalize_exprs(self, *, ref_date_col: str, tenor_col: str) -> list[pl.Expr]:
        bus_days = self.calendar.business_days_between_expr(ref_date_col, tenor_col)
        year_frac = bus_days.cast(pl.Float64) / pl.lit(float(self.calendar.base_days))
        exprs: list[pl.Expr] = [
            bus_days.alias("business_days"),
            year_frac.alias("year_fraction"),
        ]
        match self.quote_kind:
            case QuoteKind.RATE:
                # factor computed from rate; NEVER reconstruct rate from factor here
                factor = self.compounding.factor_expr(pl.col("rate"), year_frac).alias("factor")
                exprs.append(factor)
            case QuoteKind.FACTOR:
                # rate derived from supplied factor; factor column is not touched
                exprs.append(self.compounding.rate_expr(pl.col("factor"), year_frac).alias("rate"))
            case QuoteKind.UNIT_PRICE:
                pass
        return exprs


@dataclass(frozen=True, slots=True)
class VolatilityConvention:
    axes: tuple[str, ...]
    quote_kind: VolQuoteKind
    interpolation: InterpolationPolicy | None = None

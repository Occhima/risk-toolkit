from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl


class CompoundingKind(StrEnum):
    """How a quoted rate maps to a discount/capitalization factor.

    - ``LINEAR``        ``factor = 1 + r * t``
    - ``EXPONENTIAL``   ``factor = (1 + r) ** t`` (e.g. BUS/252 Brazilian rates)
    - ``CONTINUOUS``    ``factor = exp(r * t)``
    - ``DISCOUNT_FACTOR`` factor is the primary quote; rates are derived from it
    - ``UNIT_PRICE``    neither rate nor factor is the canonical quote; a unit
                        price column is supplied directly (e.g. inflation index)
    """

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CONTINUOUS = "continuous"
    DISCOUNT_FACTOR = "discount_factor"
    UNIT_PRICE = "unit_price"


@dataclass(frozen=True, slots=True)
class Compounding:
    """Lazy conversion between rates and capitalization factors.

    Stateless. All methods return Polars expressions and never call
    ``.collect()``. ``DISCOUNT_FACTOR`` and ``UNIT_PRICE`` are passthrough
    quote kinds: ``factor_expr`` and ``rate_expr`` return the input column
    unchanged so existing factors / unit prices are not recomputed.
    """

    kind: CompoundingKind

    def factor_expr(self, rate: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
        """Compute a capitalization factor from a rate and a year fraction."""
        match self.kind:
            case CompoundingKind.LINEAR:
                return pl.lit(1.0) + rate * year_fraction
            case CompoundingKind.EXPONENTIAL:
                return (pl.lit(1.0) + rate) ** year_fraction
            case CompoundingKind.CONTINUOUS:
                return (rate * year_fraction).exp()
            case CompoundingKind.DISCOUNT_FACTOR | CompoundingKind.UNIT_PRICE:
                # Passthrough: the factor (or unit price) is the primary quote.
                return rate

    def rate_expr(self, factor: pl.Expr, year_fraction: pl.Expr) -> pl.Expr:
        """Compute a rate from a capitalization factor and a year fraction."""
        match self.kind:
            case CompoundingKind.LINEAR:
                return (factor - pl.lit(1.0)) / year_fraction
            case CompoundingKind.EXPONENTIAL:
                return factor ** (pl.lit(1.0) / year_fraction) - pl.lit(1.0)
            case CompoundingKind.CONTINUOUS:
                return factor.log() / year_fraction
            case CompoundingKind.DISCOUNT_FACTOR | CompoundingKind.UNIT_PRICE:
                return factor

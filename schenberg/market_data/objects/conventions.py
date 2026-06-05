from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import polars as pl

from schenberg.market_data.objects.compounding import Compounding, CompoundingKind


@runtime_checkable
class Calendar(Protocol):
    """Structural protocol for a business-day calendar.

    Concrete calendars (e.g. :class:`schenberg.market_data.calendar.Calendar`)
    satisfy this protocol by exposing a ``name``, a ``base_days`` integer
    (business days per year used as the year-fraction denominator), and a
    method returning a Polars expression that counts business days between
    two date columns.
    """

    name: str
    base_days: int

    def business_days_between_expr(self, start_col: str, end_col: str) -> pl.Expr: ...


class QuoteKind(StrEnum):
    """How a curve quote is supplied at the boundary.

    - ``RATE``       the quote is a rate; the factor is derived via compounding.
    - ``FACTOR``     the quote is already a capitalization / discount factor.
    - ``UNIT_PRICE`` the quote is a unit price (e.g. inflation index); the
                     curve does not derive a rate or factor from it.
    """

    RATE = "rate"
    FACTOR = "factor"
    UNIT_PRICE = "unit_price"


class VolQuoteKind(StrEnum):
    """How a volatility quote is expressed."""

    LOGNORMAL = "lognormal"
    NORMAL = "normal"
    BLACK = "black"


class InterpolationKind(StrEnum):
    """Interpolation policy hint stored as metadata on a vol surface."""

    LINEAR = "linear"
    BILINEAR = "bilinear"


@dataclass(frozen=True, slots=True)
class InterpolationPolicy:
    """Lightweight metadata describing how a vol surface should be interpolated.

    The actual interpolation lives in :mod:`schenberg.market_data.interpolated`.
    This object only records the policy so downstream layers can pick the
    appropriate interpolator without inspecting columns.
    """

    kind: InterpolationKind
    axes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CurveConvention:
    """Conventions a forward / zero curve obeys.

    ``calendar`` provides business-day counting. ``compounding`` converts
    between rates and factors. ``quote_kind`` says which of rate / factor /
    unit_price is the primary quote on the input frame.
    """

    calendar: Calendar
    compounding: Compounding
    quote_kind: QuoteKind

    def normalize_exprs(
        self,
        *,
        ref_date_col: str,
        tenor_col: str,
    ) -> list[pl.Expr]:
        """Return the lazy expressions that canonicalize a raw curve frame.

        The returned list of expressions is consumed by
        :meth:`polars.LazyFrame.with_columns`. It computes (in order):
        ``business_days``, ``year_fraction``, then one of (``factor``,
        ``rate``) depending on the quote kind. ``unit_price`` quotes are
        passed through without rate / factor derivation.
        """
        business_days = self.calendar.business_days_between_expr(
            ref_date_col, tenor_col
        ).alias("business_days")
        year_fraction = (
            pl.col("business_days").cast(pl.Float64) / pl.lit(float(self.calendar.base_days))
        ).alias("year_fraction")

        exprs: list[pl.Expr] = [business_days, year_fraction]

        match self.quote_kind:
            case QuoteKind.RATE:
                # ``rate`` is the primary input; derive ``factor`` lazily.
                factor = self.compounding.factor_expr(
                    pl.col("rate"), pl.col("year_fraction")
                ).alias("factor")
                exprs.append(factor)
            case QuoteKind.FACTOR:
                # ``factor`` is the primary input; derive ``rate`` lazily
                # from it. We never recompute factor from a rate column when
                # the convention says factors lead - that is the contract
                # test 5 enforces.
                rate = self.compounding.rate_expr(
                    pl.col("factor"), pl.col("year_fraction")
                ).alias("rate")
                exprs.append(rate)
            case QuoteKind.UNIT_PRICE:
                # Pass through: unit_price quotes don't get a rate / factor.
                pass

        return exprs


@dataclass(frozen=True, slots=True)
class VolatilityConvention:
    """Conventions a volatility matrix / surface obeys.

    ``axes`` names the surface coordinates in order (e.g. ``("expiry",
    "strike")`` or ``("expiry", "moneyness")`` or ``("expiry", "tenor",
    "strike")``). ``quote_kind`` says how the volatility number is expressed.
    ``interpolation`` is optional metadata used by downstream interpolators.
    """

    axes: tuple[str, ...]
    quote_kind: VolQuoteKind
    interpolation: InterpolationPolicy | None = None

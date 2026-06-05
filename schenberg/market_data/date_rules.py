"""Pure Polars-expression helpers for deriving market-data join-key dates.

Usage pattern::

    from schenberg.market_data.date_rules import start_of_tenor_year

    prepared = legs.with_columns(
        start_of_tenor_year(tenor_col="tenor_date", output_col="pca_fixing_date")
    )

The resulting column is used as a join key in a MarketRequirement; it is never
read by the graph formulas themselves.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import polars as pl


def nth_business_day_of_following_month(
    period: pl.Expr,
    *,
    n: int,
    holidays: Iterable[date] = (),
) -> pl.Expr:
    """The ``n``-th business day of the month *after* a ``"YYYY-MM"`` period.

    Energy forwards fix/settle a few business days into the month following
    delivery. ``period`` is a delivery-month expression (e.g.
    ``cols(EnergyForwardLeg).delivery_period.expr()``); the result is the date of
    the ``n``-th business day of the next month (``n=1`` is the first business
    day), skipping weekends and ``holidays`` (pass ``ANBIMA_HOLIDAYS`` for the
    Brazilian calendar). The anchor rolls forward onto a business day first.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    first_of_following = period.str.to_date("%Y-%m").dt.offset_by("1mo")
    return first_of_following.dt.add_business_days(n - 1, holidays=list(holidays), roll="forward")


def start_of_tenor_year(
    *,
    tenor_col: str = "tenor_date",
    output_col: str = "fixing_date",
) -> pl.Expr:
    """Map each row's tenor date to January 1st of the same year."""
    year = pl.col(tenor_col).dt.year()
    return pl.date(year, 1, 1).alias(output_col)


def first_day_of_tenor_month(
    *,
    tenor_col: str = "tenor_date",
    output_col: str = "fixing_date",
) -> pl.Expr:
    """Map each row's tenor date to the first day of the same month."""
    tenor = pl.col(tenor_col)
    return pl.date(tenor.dt.year(), tenor.dt.month(), 1).alias(output_col)


_MONTH_MIN = 1
_MONTH_MAX = 12


def constant_month_of_tenor_year(
    *,
    month: int,
    tenor_col: str = "tenor_date",
    output_col: str = "fixing_date",
) -> pl.Expr:
    """Map each row's tenor date to the first day of ``month`` in the same year."""
    if not _MONTH_MIN <= month <= _MONTH_MAX:
        raise ValueError(f"month must be between 1 and 12, got {month}")
    year = pl.col(tenor_col).dt.year()
    return pl.date(year, month, 1).alias(output_col)


def copy_date(
    *,
    source_col: str,
    output_col: str = "fixing_date",
) -> pl.Expr:
    """Alias an existing date column under a new name for use as a join key."""
    return pl.col(source_col).alias(output_col)


def with_date_rule(lf: pl.LazyFrame, expr: pl.Expr) -> pl.LazyFrame:
    """Convenience wrapper: apply a date-rule expression to a LazyFrame."""
    return lf.with_columns(expr)


# ---- contract-rule date helpers -----------------------------------------
# These return un-aliased expressions so ContractRule.apply can alias them.


def same_day(anchor: str, *, output_col: str | None = None) -> pl.Expr:
    """Reference the anchor date column unchanged."""
    expr = pl.col(anchor)
    return expr if output_col is None else expr.alias(output_col)


def add_days(anchor: str, days: int, *, output_col: str | None = None) -> pl.Expr:
    """Anchor date plus ``days`` calendar days."""
    expr = pl.col(anchor) + pl.duration(days=days)
    return expr if output_col is None else expr.alias(output_col)


def previous_day(anchor: str, *, output_col: str | None = None) -> pl.Expr:
    """One calendar day before the anchor date."""
    expr = pl.col(anchor) - pl.duration(days=1)
    return expr if output_col is None else expr.alias(output_col)

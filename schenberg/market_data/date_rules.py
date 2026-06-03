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


# Energy forwards settle on the 6th business day (ANBIMA calendar) after the
# last calendar day of the delivery month.
_ENERGY_SETTLE_BUSINESS_DAYS = 6


def energy_settlement_date(
    *,
    period_col: str = "delivery_period",
    output_col: str = "fixing_date",
    business_days_after_month_end: int = _ENERGY_SETTLE_BUSINESS_DAYS,
    holidays: Iterable[date] = (),
) -> pl.Expr:
    """Map an energy delivery period to its settlement / fixing date.

    The delivery period is a ``"YYYY-MM"`` string; the fixing is the
    ``business_days_after_month_end``-th business day *after* the last calendar
    day of that month, skipping weekends and any ``holidays`` supplied. Pass the
    ANBIMA holiday set to honour the full calendar — with no holidays only
    weekends are skipped.

    Example: ``"2029-06"`` ends on Sat 2029-06-30; with weekends only the 6th
    business day after is 2029-07-09.
    """
    if business_days_after_month_end < 1:
        raise ValueError(
            f"business_days_after_month_end must be >= 1, got {business_days_after_month_end}"
        )
    month_end = pl.format("{}-01", pl.col(period_col)).str.to_date("%Y-%m-%d").dt.month_end()
    # roll="backward" anchors on the last business day on-or-before month-end, so
    # the count is "Nth business day after the month" whether or not the month
    # ends on a weekend/holiday.
    return month_end.dt.add_business_days(
        business_days_after_month_end,
        holidays=list(holidays),
        roll="backward",
    ).alias(output_col)


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
